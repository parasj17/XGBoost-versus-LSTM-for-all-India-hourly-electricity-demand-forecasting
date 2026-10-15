"""LSTM with a two-output quantile head (5% and 95%) for the uncertainty extension (protocol deviation log, 2026-10-15).

Architecture of the L4_full_A point model (one LSTM layer + small MLP) with a two-output head trained on the pinball loss; stage-1 hyper-parameters
of each horizon (hidden size, learning rate, dropout, weight decay); target = percent deviation from the 24 h anchor divided by the training
standard deviation; early stopping on the validation pinball loss (max 30 epochs, patience 5); seeds 0, 1, 2; quantile predictions averaged over seeds.
Validation is therefore used for early stopping and later for conformal calibration (mildly optimistic).
--check restricts everything to the validation period (the test period is not loaded), one horizon, one seed, 2 epochs, nothing saved.
Outputs: data/interim/predictions/lstm_quantile_h{h}.npz, reports/tables/lstm_quantile_stats.json.   Usage: py scripts/run_quantile_lstm.py [--check]
"""
import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
rl = importlib.import_module("run_lstm")
rt = importlib.import_module("run_test_evaluation")

TAU = torch.tensor([0.05, 0.95])
MAX_EPOCHS, PATIENCE = 30, 5


class NetQ(nn.Module):
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x, s):
        o, _ = self.lstm(x)
        return self.head(torch.cat([o[:, -1], s], dim=1))


def pinball(q, z):
    d = z.unsqueeze(1) - q
    return torch.maximum(TAU * d, (TAU - 1.0) * d).mean()


def predict_q(net, data, n, bs=1024):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            x, s, _ = data.batch(np.arange(i, min(i + bs, n)))
            out.append(net(x, s).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    _, _, lstm_cfg = rt.frozen_configs()
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp")
    ev_end = rb.VAL_END if args.check else rt.TEST_END
    df = df.loc[:ev_end]
    horizons, seeds, max_ep = ([24], [0], 2) if args.check else (rb.HORIZONS, rt.SEEDS, MAX_EPOCHS)
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    stats = {}
    for h in horizons:
        b = rb.build(df, h)
        Xs = {n: rb.design(b, rel, full) for n, (rel, full) in
              {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
        need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna()
        if h <= 168:
            need &= b["anchors"]["a168"].notna()
        valid = need & b["target_obs"]
        Tt = b["T"]
        in_tr = valid & (Tt <= rb.TRAIN_END)
        in_va = valid & (Tt >= rb.VAL_START) & (Tt <= rb.VAL_END)
        in_te = valid & (Tt >= rt.TEST_START) & (Tt <= ev_end)
        a24 = b["anchors"]["a24"]
        a24v = a24.fillna(1.0).to_numpy()
        pos = np.arange(len(df))
        idx_tr, idx_va, idx_te = (pos[m.to_numpy()] for m in (in_tr, in_va, in_te))
        z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
        zsd = float(np.nanstd(z_all[idx_tr]))
        zt = np.nan_to_num(z_all / zsd)
        C = rl.step_channels(df, True, True, rb.TRAIN_END)
        S = rl.static_features(df, h, b, True)
        tr, va = rl.Data(y_all, a24v, C, S, zt, idx_tr), rl.Data(y_all, a24v, C, S, zt, idx_va)
        te = rl.Data(y_all, a24v, C, S, zt, idx_te if not args.check else idx_va)
        y_va = b["target"].to_numpy()[idx_va]
        zva = torch.from_numpy(zt[idx_va].astype(np.float32))
        hp = lstm_cfg[(h, rt.HEADLINE_L, 0)][0]
        qva, qte, per_seed = [], [], []
        t_all = time.perf_counter()
        for sd in seeds:
            torch.manual_seed(sd)
            rng = np.random.default_rng(sd)
            net = NetQ(C.shape[1] + 1, S.shape[1], hp["hidden"], hp["dropout"])
            opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            best, best_state, bad, ep_used = np.inf, None, 0, 0
            t0 = time.perf_counter()
            for ep in range(max_ep):
                net.train()
                perm = rng.permutation(len(idx_tr))
                for i in range(0, len(perm), rl.BATCH):
                    x, s, z = tr.batch(perm[i:i + rl.BATCH])
                    loss = pinball(net(x, s), z)
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    vl = float(pinball(torch.from_numpy(predict_q(net, va, len(idx_va))), zva))
                ep_used = ep + 1
                if vl < best - 1e-6:
                    best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            fit_s = time.perf_counter() - t0
            net.load_state_dict(best_state)
            conv = lambda q, idx: (1 + np.sort(q, axis=1) * zsd / 100.0) * a24v[idx][:, None]
            qv = conv(predict_q(net, va, len(idx_va)), idx_va)
            t1 = time.perf_counter()
            qt = conv(predict_q(net, te, len(te.idx)), te.idx)
            inf_s = time.perf_counter() - t1
            qva.append(qv); qte.append(qt)
            per_seed.append({"seed": sd, "epochs": ep_used, "fit_s": round(fit_s, 1), "val_pinball_z": best, "raw_val_coverage": float(np.mean((y_va >= qv[:, 0]) & (y_va <= qv[:, 1]))),
                             "inference_s": round(inf_s, 2)})
            print(f"h={h} seed {sd}: epochs {ep_used}, fit {fit_s:.0f} s, raw validation coverage {100 * per_seed[-1]['raw_val_coverage']:.1f}%", flush=True)
        qva, qte = np.mean(qva, axis=0), np.mean(qte, axis=0)
        stats[h] = {"seeds": per_seed, "mean_raw_val_coverage_of_seed_average": float(np.mean((y_va >= qva[:, 0]) & (y_va <= qva[:, 1]))),
                    "total_fit_s": round(time.perf_counter() - t_all, 1)}
        if not args.check:
            np.savez_compressed(ROOT / "data" / "interim" / "predictions" / f"lstm_quantile_h{h}.npz", q05_va=qva[:, 0], q95_va=qva[:, 1], q05_te=qte[:, 0], q95_te=qte[:, 1])
    if not args.check:
        (ROOT / "reports" / "tables" / "lstm_quantile_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
