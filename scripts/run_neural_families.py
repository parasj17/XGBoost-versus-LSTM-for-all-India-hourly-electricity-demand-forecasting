"""Part P3 (b)-(d) of the completion batch: GRU, TCN and a small Transformer encoder instead of the LSTM (docs/experiment_protocol.md, entry of 2026-10-19).

Same inputs, target (ratio to the 24 h anchor), valid samples and training procedure as the LSTM L4: the last 168 hours before the origin plus the static target-time features, a recurrent/convolutional/attention
encoder, the same MLP head (TCN: 3-hour average pooling and 4 residual blocks; Transformer: 6-hour patches). Tasks (as amended on 2026-10-20 because of the measured cost): horizons 1 and 24 h and the pooled day-ahead task (issue 10:00 on D-1, leads 14-37 h). 12 random trials per task of the stage-3 space (at most 10 epochs, patience 3), selection
on validation, then 3 seeds with early stopping on validation (at most 30 epochs, patience 5).
  py scripts/run_neural_families.py --model gru|tcn|transformer               PHASE 1 (validation only): writes reports/tables/neural_<model>_config.json and reports/neural_<model>_validation.md
  py scripts/run_neural_families.py --model gru|tcn|transformer --with-test   PHASE 2: single test evaluation (lock reports/tables/NEURAL_<MODEL>_TEST_DONE.lock) -> reports/neural_<model>_test.md,
                                                                              data/interim/predictions/neural_<model>.npz
"""
import argparse
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
r3 = importlib.import_module("run_lstm_stage3")
rda = importlib.import_module("run_dayahead")
T = ROOT / "reports" / "tables"
PRED = ROOT / "data" / "interim" / "predictions"
HORIZONS, SEEDS, N_TRIALS = [1, 24], [0, 1, 2], 12


class _Head(nn.Module):
    def __init__(self, hidden, n_static, dropout):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, h_last, s):
        return self.head(torch.cat([h_last, s], dim=1)).squeeze(1)


class GRUNet(nn.Module):
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, batch_first=True)
        self.out = _Head(hidden, n_static, dropout)

    def forward(self, x, s):
        o, _ = self.rnn(x)
        return self.out(o[:, -1], s)


class _Block(nn.Module):
    def __init__(self, c_in, c_out, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.c1, self.c2 = nn.Conv1d(c_in, c_out, 3, dilation=dilation), nn.Conv1d(c_out, c_out, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + self.res(x))


class TCNNet(nn.Module):
    """The 168 hourly steps are average-pooled to 56 steps of 3 hours, then 4 residual blocks of dilated causal convolutions (dilations 1, 2, 4, 8; receptive field 61 steps = 183 hours), last time step."""
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.pool = nn.AvgPool1d(3)
        chans = [n_in] + [hidden] * 4
        self.blocks = nn.Sequential(*[_Block(chans[i], chans[i + 1], 2 ** i, dropout) for i in range(4)])
        self.out = _Head(hidden, n_static, dropout)

    def forward(self, x, s):
        y = self.blocks(self.pool(x.transpose(1, 2)))
        return self.out(y[:, :, -1], s)


class TransformerNet(nn.Module):
    """Patching: the 168 hourly steps are grouped into 28 patches of 6 hours (linear patch embedding), learned position embedding, 2 encoder layers (4 heads), last-token pooling."""
    PATCH = 6

    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.proj = nn.Linear(n_in * self.PATCH, hidden)
        self.pos = nn.Parameter(torch.zeros(1, rl.L // self.PATCH, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=2 * hidden, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.out = _Head(hidden, n_static, dropout)

    def forward(self, x, s):
        o = self.enc(self.proj(x.reshape(x.shape[0], rl.L // self.PATCH, -1)) + self.pos)
        return self.out(o[:, -1], s)


FACTORY = {"gru": GRUNet, "tcn": TCNNet, "transformer": TransformerNet}


def build_task(df, y_all, C, h, with_test):
    b = rb.build(df, h)
    Xs = {n: rb.design(b, rel, full) for n, (rel, full) in {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
    need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna() & b["anchors"]["a168"].notna()
    valid = need & b["target_obs"]
    Tt = b["T"]
    pos = np.arange(len(df))
    ms = {"tr": valid & (Tt <= rb.TRAIN_END), "va": valid & (Tt >= rb.VAL_START) & (Tt <= rb.VAL_END)}
    if with_test:
        ms["te"] = valid & (Tt >= rt.TEST_START) & (Tt <= rt.TEST_END)
    idx = {k: pos[m.to_numpy()] for k, m in ms.items()}
    a24 = b["anchors"]["a24"]
    a24v = a24.fillna(1.0).to_numpy()
    z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
    zsd = float(np.nanstd(z_all[idx["tr"]]))
    zt = np.nan_to_num(z_all / zsd)
    S = rl.static_features(df, h, b, True)
    data = {k: rl.Data(y_all, a24v, C, S, zt, ix) for k, ix in idx.items()}
    rows = {k: {"y": b["target"].to_numpy()[ix], "a24": a24v[ix], "T": b["T"].to_numpy()[ix].astype("datetime64[ns]").astype("int64")} for k, ix in idx.items()}
    return data, rows, zsd, S.shape[1]


def build_dayahead(df, y_all, C, with_test):
    D = rda.collect(df, rda.LEADS, with_test)
    splits = ["tr", "va"] + (["te"] if with_test else [])
    cat = lambda f, s: np.concatenate([f(D[h], D[h]["pos"][s]) for h in rda.LEADS])
    R = {s: {"pos": cat(lambda d, p: p, s), "a24": cat(lambda d, p: d["a24v"][p], s), "y": cat(lambda d, p: d["target"][p], s), "zx": cat(lambda d, p: d["zx"].to_numpy()[p], s),
             "T": cat(lambda d, p: d["T"].to_numpy()[p].astype("datetime64[ns]").astype("int64"), s), "S": np.concatenate([D[h]["S"][D[h]["pos"][s]] for h in rda.LEADS])} for s in splits}
    zsd = float(np.nanstd((R["tr"]["zx"] - 1.0) * 100))
    data = {s: rda.Pooled(y_all, C, R[s]["pos"], R[s]["a24"], R[s]["S"], np.nan_to_num(((R[s]["zx"] - 1.0) * 100) / zsd)) for s in splits}
    return data, {s: {"y": R[s]["y"], "a24": R[s]["a24"], "T": R[s]["T"]} for s in splits}, zsd, R["tr"]["S"].shape[1]


def main():
    global HORIZONS, N_TRIALS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(FACTORY), required=True)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    name = args.model
    rl.Net = FACTORY[name]                         # the shared training code (run_lstm.train, run_lstm_stage3.trial_fit, run_test_evaluation.lstm_fit_fixed) builds rl.Net
    CONFIG, LOCK = T / f"neural_{name}_config.json", T / f"NEURAL_{name.upper()}_TEST_DONE.lock"
    commit, dirty = rb.git_state()
    if args.smoke:
        HORIZONS, N_TRIALS = [24], 1
    if args.with_test:
        if LOCK.exists():
            sys.exit(f"The {name} test evaluation was already run.")
        if dirty:
            sys.exit("Working tree is not clean: commit first.")
        cfg_all = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:(rt.TEST_END if args.with_test else rb.VAL_END)]
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    C = rl.step_channels(df, True, True, rb.TRAIN_END)
    n_in = C.shape[1] + 1
    ev = "te" if args.with_test else "va"
    tasks = [str(h) for h in HORIZONS] + ([] if args.smoke else ["dayahead"])
    md = ["# " + (f"{name.upper() if name != 'transformer' else 'Transformer'}: single test evaluation" if args.with_test else f"{name.upper() if name != 'transformer' else 'Transformer'}: validation") + f" (generated by `scripts/run_neural_families.py --model {name}`; part P3 of the completion batch)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. LSTM L4 inputs, target and training procedure with the {name} encoder; {N_TRIALS} random trials per task (stage-3 space, at most 10 epochs, patience 3), selection on validation, 3 seeds with early stopping on validation. MAE in MW. "
          + ("Models fitted on the training period only with the validation-chosen configuration and epochs." if args.with_test else "Hyper-parameters chosen on validation (mildly optimistic)."), "",
          "| task | " + ("test" if args.with_test else "validation") + " MAE (MW) | selected hidden / lr / dropout / weight decay | best epochs (3 seeds) |", "|---|---:|---|---|"]
    cfg_out, preds = {}, {}
    t0 = time.perf_counter()
    for task in tasks:
        data, rows, zsd, n_st = build_dayahead(df, y_all, C, args.with_test) if task == "dayahead" else build_task(df, y_all, C, int(task), args.with_test)
        trD, eD, E = data["tr"], data[ev], rows[ev]
        if not args.with_test:
            r_ = np.random.default_rng(9800 + (int(task) if task.isdigit() else 0))
            best = None
            for _ in range(N_TRIALS):
                hp = r3.sample_hp(r_)
                m_, *_ = r3.trial_fit(trD, eD, n_in, n_st, hp, zsd, E["a24"], E["y"], None)
                if best is None or m_ < best[0]:
                    best = (m_, hp)
            epochs, ps = {}, []
            for sd in SEEDS:
                net, m_, hist, *_ = rl.train(trD, eD, n_in, n_st, best[1], sd, 2 if args.smoke else 30, 5, zsd, E["a24"], E["y"])
                epochs[str(sd)] = int(np.argmin(hist)) + 1
                ps.append((1 + rl.predict(net, eD, len(E["y"])) * zsd / 100.0) * E["a24"])
            hp, ep = best[1], epochs
            cfg_out[task] = {"params": hp, "epochs": epochs}
        else:
            hp, ep = cfg_all[task]["params"], cfg_all[task]["epochs"]
            ps = [(1 + rl.predict(rt.lstm_fit_fixed(trD, n_in, n_st, hp, sd, ep[str(sd)]), eD, len(E["y"])) * zsd / 100.0) * E["a24"] for sd in SEEDS]
        p = np.mean(ps, axis=0)
        preds[task], preds[task + "_y"], preds[task + "_T"] = p, E["y"], E["T"]
        mae = float(np.mean(np.abs(p - E["y"])))
        md.append(f"| {task if task == 'dayahead' else task + ' h'} | {mae:,.0f} | {hp['hidden']} / {hp['lr']:.4g} / {hp['dropout']:.2f} / {hp['weight_decay']:.2g} | {list(ep.values())} |")
        print(f"{name} {task}: {ev} MAE {mae:.0f}; {time.perf_counter() - t0:.0f} s", flush=True)
    md += ["", f"Run time {time.perf_counter() - t0:,.0f} s.", ""]
    if args.smoke:
        print(chr(10).join(md))
        print("smoke finished (nothing written)")
        return
    np.savez_compressed(PRED / (f"neural_{name}.npz" if args.with_test else f"neural_{name}_val.npz"), **preds)
    if args.with_test:
        (ROOT / "reports" / f"neural_{name}_test.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
        LOCK.write_text(f"{name} test run completed at commit {commit}" + chr(10), encoding="utf-8")
    else:
        CONFIG.write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")
        (ROOT / "reports" / f"neural_{name}_validation.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
