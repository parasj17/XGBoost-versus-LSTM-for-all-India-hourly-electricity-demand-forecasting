"""Part P2 (c) of the completion batch: a 24-output LSTM for the day-ahead task (docs/experiment_protocol.md, entry of 2026-10-19).

One network per day: input = the last 168 hours before the 10:00 origin (demand relative to the mean of the last 24 hours, the L4 weather and solar/wind channels, hour/weekday encodings); static input = the
day-level calendar and holiday features of day D (weekday, month, weekend, holiday, major-festival and festival-distance features) and last week's relative anchors of the 24 target hours; 24 outputs = the
ratio-to-anchor targets of the 24 target hours (anchors as in the pooled day-ahead models), trained with a mask for invalid hours. 30 random trials of the stage-3 space (at most 40 epochs, patience 5, batch 32),
3 seeds, early stopping on validation (mean absolute error in MW over the valid hours).
  py scripts/run_dayahead_lstm24.py               PHASE 1 (validation only): writes reports/tables/dayahead_lstm24_config.json, reports/dayahead_lstm24_validation.md, data/interim/predictions/dayahead_lstm24_val.npz
  py scripts/run_dayahead_lstm24.py --with-test   PHASE 2: single test evaluation (lock reports/tables/DAYAHEAD_LSTM24_TEST_DONE.lock) -> reports/dayahead_lstm24_test.md, dayahead_lstm24.npz
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
r3 = importlib.import_module("run_lstm_stage3")
rda = importlib.import_module("run_dayahead")
T = ROOT / "reports" / "tables"
CONFIG, LOCK = T / "dayahead_lstm24_config.json", T / "DAYAHEAD_LSTM24_TEST_DONE.lock"
PRED = ROOT / "data" / "interim" / "predictions"
LEADS, SEEDS, N_TRIALS, MAX_EP, PAT, BS = rda.LEADS, rda.SEEDS, 30, 40, 5, 32


class Net24(nn.Module):
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 24))

    def forward(self, x, s):
        o, _ = self.lstm(x)
        return self.head(torch.cat([o[:, -1], s], dim=1))


class DayData:
    def __init__(self, y, ref, C, p, S, z, mask):
        self.y, self.ref, self.C, self.p, self.S, self.z, self.mask = y, ref, C, p, S, z, mask
        self.n = len(p)
        self.off = np.arange(-(rl.L - 1), 1)

    def batch(self, rows):
        p = self.p[rows]
        w = p[:, None] + self.off[None, :]
        dem = ((self.y[w] / self.ref[p][:, None]) - 1.0) / 0.2
        x = np.concatenate([dem[:, :, None].astype(np.float32), self.C[w]], axis=2)
        return torch.from_numpy(x), torch.from_numpy(self.S[rows]), torch.from_numpy(self.z[rows]), torch.from_numpy(self.mask[rows])


def predict(net, d):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, d.n, 256):
            x, s, _, _ = d.batch(np.arange(i, min(i + 256, d.n)))
            out.append(net(x, s).numpy())
    return np.concatenate(out)


def fit(tr, va, hp, seed, epochs, patience, n_in, n_st, zsd, a_va, y_va, m_va):
    """Train with the masked squared error; if va is given, early stopping on the validation MAE (MW) over valid cells; returns (net, best_epoch, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net24(n_in, n_st, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr.n)
        for i in range(0, tr.n, BS):
            x, s, z, m = tr.batch(perm[i:i + BS])
            loss = (((net(x, s) - z) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        if va is not None:
            pr = (1 + predict(net, va) * zsd / 100.0) * a_va
            mae = float(np.abs(pr - y_va)[m_va].mean())
            hist.append(mae)
            if mae < best - 1e-6:
                best, best_state, bad = mae, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, (int(np.argmin(hist)) + 1 if hist else epochs), hist


def main():
    global N_TRIALS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    commit, dirty = rb.git_state()
    leads = LEADS
    if args.smoke:
        N_TRIALS = 1
    if args.with_test:
        if LOCK.exists():
            sys.exit("The 24-output LSTM test evaluation was already run.")
        if dirty:
            sys.exit("Working tree is not clean: commit first.")
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:(rt.TEST_END if args.with_test else rb.VAL_END)]
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    ref = pd.Series(y_all).rolling(24, min_periods=1).mean().to_numpy()
    D = rda.collect(df, leads, args.with_test)
    ev = "te" if args.with_test else "va"
    splits = ["tr", "va"] + (["te"] if args.with_test else [])
    C = rl.step_channels(df, True, True, rb.TRAIN_END)
    n_in = C.shape[1] + 1
    S_all = {h: rl.static_features(df, h, D[h]["b"], True) for h in leads}
    mid = leads[len(leads) // 2]                      # lead 26: noon of day D, for the day-level calendar columns

    def day_set(split):
        orig = np.unique(np.concatenate([D[h]["pos"][split] for h in leads]))
        ix = {p: i for i, p in enumerate(orig)}
        n = len(orig)
        z, mask, a24, yy = np.zeros((n, 24), np.float32), np.zeros((n, 24), bool), np.ones((n, 24)), np.zeros((n, 24))
        s_rel = np.zeros((n, 24), np.float32)
        for k, h in enumerate(leads):
            pos = D[h]["pos"][split]
            ii = np.array([ix[p] for p in pos])
            zz = (D[h]["zx"].to_numpy()[pos] - 1.0) * 100.0
            z[ii, k], mask[ii, k], a24[ii, k], yy[ii, k] = zz, True, D[h]["a24v"][pos], D[h]["target"][pos]
        for k, h in enumerate(leads):
            s_rel[:, k] = S_all[h][orig, -1]
        Sd = np.concatenate([S_all[mid][orig][:, 2:-1], s_rel], axis=1).astype(np.float32)
        return orig, ix, z, mask, a24, yy, Sd

    DS = {s: day_set(s) for s in splits}
    ztr_all = DS["tr"][2][DS["tr"][3]]
    zsd = float(np.std(ztr_all))
    n_st = DS["tr"][6].shape[1]
    mk = lambda s: DayData(y_all, ref, C, DS[s][0], DS[s][6], np.nan_to_num(DS[s][2] / zsd).astype(np.float32), DS[s][3])
    trD, eD = mk("tr"), mk(ev)
    _, _, _, m_e, a_e, y_e, _ = DS[ev]
    n_ = {s: int(DS[s][3].sum()) for s in splits}
    print("origins:", {s: len(DS[s][0]) for s in splits}, "valid cells:", n_, flush=True)
    t0 = time.perf_counter()
    if not args.with_test:
        r2, cands = np.random.default_rng(9700), []
        for _ in range(N_TRIALS):
            hp = r3.sample_hp(r2)
            net, ep, hist = fit(trD, eD, hp, 0, MAX_EP, PAT, n_in, n_st, zsd, a_e, y_e, m_e)
            cands.append((min(hist), hp))
        sel = min(cands, key=lambda x: x[0])
        epochs, ps = {}, []
        for sd in SEEDS:
            net, ep, hist = fit(trD, eD, sel[1], sd, MAX_EP, PAT, n_in, n_st, zsd, a_e, y_e, m_e)
            epochs[sd] = ep
            ps.append((1 + predict(net, eD) * zsd / 100.0) * a_e)
        cfg_out = {"params": sel[1], "epochs": {str(k): v for k, v in epochs.items()}, "zsd": zsd}
        print("selected", sel[1], epochs, flush=True)
    else:
        ps = []
        for sd in SEEDS:
            net, _, _ = fit(trD, None, cfg["params"], sd, cfg["epochs"][str(sd)], PAT, n_in, n_st, zsd, a_e, y_e, m_e)
            ps.append((1 + predict(net, eD) * zsd / 100.0) * a_e)
    P = np.mean(ps, axis=0)
    # back to the lead-major row order of the other day-ahead models
    ix = DS[ev][1]
    rows = np.concatenate([P[[ix[p] for p in D[h]["pos"][ev]], k] for k, h in enumerate(leads)])
    y_rows = np.concatenate([D[h]["target"][D[h]["pos"][ev]] for h in leads])
    zref = np.load(PRED / ("dayahead.npz" if args.with_test else "dayahead_val.npz"), allow_pickle=True)
    if not args.smoke:
        assert np.allclose(zref["y"], y_rows), "row order differs from the day-ahead run"
    mae = float(np.mean(np.abs(rows - y_rows)))
    ref_mae = {k: (float(np.mean(np.abs(zref[k] - zref["y"]))) if not args.smoke else 0.0) for k in ("ridge", "xgb", "lstm")}
    md = ["# " + ("24-output LSTM, day-ahead: single test evaluation" if args.with_test else "24-output LSTM, day-ahead: validation") + " (generated by `scripts/run_dayahead_lstm24.py`; part P2 (c) of the completion batch)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. One network per day with 24 outputs; same (day, hour) samples as the other day-ahead models ({len(y_rows):,}); training origins {len(DS['tr'][0])}, evaluation origins {len(DS[ev][0])}. "
          + ("Fitted on the training period only with the validation-chosen configuration." if args.with_test else "30 random trials; selection and early stopping on validation (mildly optimistic)."), "",
          "| model | " + ("test" if args.with_test else "validation") + " MAE (MW) |", "|---|---:|", f"| 24-output LSTM | {mae:,.0f} |", f"| LSTM L4, pooled single-output (day-ahead model) | {ref_mae['lstm']:,.0f} |",
          f"| XGBoost S4, pooled (day-ahead model) | {ref_mae['xgb']:,.0f} |", f"| ridge (day-ahead model) | {ref_mae['ridge']:,.0f} |", ""]
    if not args.with_test:
        md += [f"Selected: hidden {sel[1]['hidden']}, lr {sel[1]['lr']:.4g}, dropout {sel[1]['dropout']:.2f}, weight decay {sel[1]['weight_decay']:.2g}; best epochs {list(epochs.values())}.", ""]
    md += [f"Run time {time.perf_counter() - t0:,.0f} s.", ""]
    print(chr(10).join(md))
    if args.smoke:
        print("smoke finished (nothing written)")
        return
    np.savez_compressed(PRED / ("dayahead_lstm24.npz" if args.with_test else "dayahead_lstm24_val.npz"), y=y_rows, lead=zref["lead"], T=zref["T"], lstm24=rows)
    if args.with_test:
        (ROOT / "reports" / "dayahead_lstm24_test.md").write_text(chr(10).join(md), encoding="utf-8")
        LOCK.write_text(f"24-output LSTM test run completed at commit {commit}" + chr(10), encoding="utf-8")
    else:
        CONFIG.write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")
        (ROOT / "reports" / "dayahead_lstm24_validation.md").write_text(chr(10).join(md), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
