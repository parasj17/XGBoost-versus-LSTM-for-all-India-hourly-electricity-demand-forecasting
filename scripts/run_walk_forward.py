"""Walk-forward evaluation over 2024-01-01 .. 2025-06-24 with quarterly retraining (protocol deviation log, 2026-10-16).

For each quarterly window the models are retrained on all valid samples whose target precedes the window and scored on the window, with the FROZEN settings of each horizon:
ridge (variant and penalty of the test run), XGBoost S4_full_A (stage-2 parameters, validation-chosen number of trees per seed), LSTM L4_full_A (stage-1 parameters,
validation-chosen epochs per seed) and, if reports/tables/lstm_stage3_config.json exists, the equal-budget LSTM. 3 seeds, mean prediction; LSTM scaling statistics come from
the training part of each window. Nothing is selected from these results.
Windows 2024-Q1 and 2024-Q2 lie in the validation period on which the hyper-parameters were tuned (NOT independent); the windows from 2024-Q3 overlap the test period and
are an additional, exploratory view of it.
Resumable: results are appended to reports/tables/walk_forward.jsonl. Output: reports/walk_forward.md.
  py scripts/run_walk_forward.py [--check]    (--check: one window inside the validation period, horizon 24, one seed, 2 epochs, nothing written)
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")
rx = importlib.import_module("run_xgboost")
rl = importlib.import_module("run_lstm")
rt = importlib.import_module("run_test_evaluation")
T = ROOT / "reports" / "tables"
LOG = T / "walk_forward.jsonl"
WINDOWS = [("2024-Q1", "2024-01-01", "2024-03-31 23:00", False), ("2024-Q2", "2024-04-01", "2024-06-30 23:00", False),
           ("2024-Q3", "2024-07-01", "2024-09-30 23:00", True), ("2024-Q4", "2024-10-01", "2024-12-31 23:00", True),
           ("2025-Q1", "2025-01-01", "2025-03-31 23:00", True), ("2025-Q2", "2025-04-01", "2025-06-24 23:00", True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rx.MAX_TREES = 3000
    xgb_params, xgb_trees, lstm_cfg = rt.frozen_configs()
    ridge_cfg = {r["horizon"]: tuple(r["ridge"]) for r in rt.read_jsonl("test_results.jsonl")}
    s3 = json.loads((T / "lstm_stage3_config.json").read_text(encoding="utf-8")) if (T / "lstm_stage3_config.json").exists() else None
    windows, horizons, seeds = (WINDOWS[:1], [24], [0]) if args.check else (WINDOWS, rb.HORIZONS, rt.SEEDS)
    last = pd.Timestamp(windows[-1][2])
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:last]
    if args.check:
        assert df.index.max() <= rb.VAL_END
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    done = {(r["window"], r["horizon"]) for r in rt.read_jsonl("walk_forward.jsonl")} if (LOG.exists() and not args.check) else set()
    commit, dirty = rb.git_state()
    pos = np.arange(len(df))
    for h in horizons:
        b = rb.build(df, h)
        Xs = {n: rb.design(b, rel, full) for n, (rel, full) in
              {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
        need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna()
        if h <= 168:
            need &= b["anchors"]["a168"].notna()
        valid = need & b["target_obs"]
        Tt = b["T"]
        a24 = b["anchors"]["a24"]
        a24v = a24.fillna(1.0).to_numpy()
        rname, alpha = ridge_cfg[h]
        Xd = Xs[rname.replace("ridge_", "")]
        rel_r = rname.endswith("rel")
        Xf = ft.build_features(df, h, ft.FEATURE_SETS["S4_full_A"], True)
        zx = b["target"] / a24
        z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
        C, S = None, rl.static_features(df, h, b, True)
        for wname, ws, we, indep in windows:
            if (wname, h) in done:
                continue
            ws, we = pd.Timestamp(ws), pd.Timestamp(we)
            in_tr = valid & (Tt < ws)
            in_ev = valid & (Tt >= ws) & (Tt <= we)
            idx_tr, idx_ev = pos[in_tr.to_numpy()], pos[in_ev.to_numpy()]
            y_ev, a_ev = b["target"].to_numpy()[idx_ev], a24v[idx_ev]
            res = {"window": wname, "horizon": h, "n_train": int(in_tr.sum()), "n_eval": int(in_ev.sum()), "validation_period_window": not indep, "git_commit": commit}
            res["best_naive"] = min(float(np.mean(np.abs(v - y_ev))) for v in (b["y"].to_numpy()[idx_ev], a24.to_numpy()[idx_ev], b["anchors"]["a168"].to_numpy()[idx_ev]))
            tgt = (b["target"] / a24) if rel_r else b["target"]
            m = rb.ridge_fit(Xd[in_tr], tgt[in_tr], alpha)
            p = rb.ridge_pred(m, Xd[in_ev])
            res["ridge"] = float(np.mean(np.abs((p * a_ev if rel_r else p) - y_ev)))
            xp = []
            for sd in seeds:
                mdl = xgb.XGBRegressor(n_estimators=xgb_trees[(h, "S4_full_A", sd)], objective="reg:squarederror", tree_method="hist", device="cpu",
                                       n_jobs=rx.NJOBS, random_state=sd, eval_metric="mae", **xgb_params[h])
                mdl.fit(Xf[in_tr], (zx[in_tr] - 1.0) * 100.0, verbose=False)
                xp.append((1.0 + mdl.predict(Xf[in_ev]) / 100.0) * a_ev)
            res["xgb"] = float(np.mean(np.abs(np.mean(xp, axis=0) - y_ev)))
            zsd = float(np.nanstd(z_all[idx_tr]))
            zt = np.nan_to_num(z_all / zsd)
            C = rl.step_channels(df, True, True, ws - pd.Timedelta(hours=1))
            trD, evD = rl.Data(y_all, a24v, C, S, zt, idx_tr), rl.Data(y_all, a24v, C, S, zt, idx_ev)
            variants = {"lstm": [(lstm_cfg[(h, rt.HEADLINE_L, sd)][0], lstm_cfg[(h, rt.HEADLINE_L, sd)][1]) for sd in seeds]}
            if s3 is not None:
                variants["lstm_eqbudget"] = [(s3[str(h)]["params"], s3[str(h)]["best_epoch"][str(sd)]) for sd in seeds]
            for vname, cfgs in variants.items():
                lp = []
                for sd, (hp, ep) in zip(seeds, cfgs):
                    net = rt.lstm_fit_fixed(trD, C.shape[1] + 1, S.shape[1], hp, sd, 2 if args.check else ep)
                    lp.append((1 + rl.predict(net, evD, len(idx_ev)) * zsd / 100.0) * a_ev)
                res[vname] = float(np.mean(np.abs(np.mean(lp, axis=0) - y_ev)))
            print(f"{wname} h={h}: naive {res['best_naive']:.0f} ridge {res['ridge']:.0f} xgb {res['xgb']:.0f} lstm {res['lstm']:.0f}" +
                  (f" lstm_eqbudget {res['lstm_eqbudget']:.0f}" if "lstm_eqbudget" in res else ""), flush=True)
            if not args.check:
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + chr(10))
    if args.check:
        print("check finished (nothing written)")
        return
    rows = rt.read_jsonl("walk_forward.jsonl")
    cols = [("best_naive", "best naive"), ("ridge", "ridge"), ("xgb", "XGBoost S4"), ("lstm", "LSTM L4")] + ([("lstm_eqbudget", "LSTM equal budget")] if any("lstm_eqbudget" in r for r in rows) else [])
    md = ["# Walk-forward evaluation, 2024-01-01 to 2025-06-24 (generated by `scripts/run_walk_forward.py`)", "",
          "Quarterly retraining with the frozen hyper-parameters of each horizon (see the script docstring). MAE in MW, mean of the 3 seeds' predictions. "
          "**Windows marked * (2024-Q1, 2024-Q2) lie in the validation period on which the hyper-parameters were tuned and are not independent. "
          "Windows from 2024-Q3 overlap the test period** and are an additional, exploratory view of it; nothing is selected from these results.", ""]
    for h in rb.HORIZONS:
        rr = sorted((r for r in rows if r["horizon"] == h), key=lambda r: r["window"])
        if not rr:
            continue
        md += [f"## Horizon {h} h", "", "| window | train / eval samples | " + " | ".join(c[1] for c in cols) + " | XGBoost minus ridge | LSTM L4 minus XGBoost |", "|---|---:|" + "---:|" * (len(cols) + 2)]
        for r in rr:
            md.append(f"| {r['window']}{'*' if r['validation_period_window'] else ''} | {r['n_train']:,} / {r['n_eval']:,} | " + " | ".join(f"{r[c[0]]:,.0f}" if c[0] in r else "" for c in cols) +
                      f" | {r['xgb'] - r['ridge']:+,.0f} | {r['lstm'] - r['xgb']:+,.0f} |")
        ind = [r for r in rr if not r["validation_period_window"]]
        if ind:
            md.append(f"| mean of windows from 2024-Q3 | | " + " | ".join(f"{np.mean([r[c[0]] for r in ind if c[0] in r]):,.0f}" if any(c[0] in r for r in ind) else "" for c in cols) +
                      f" | {np.mean([r['xgb'] - r['ridge'] for r in ind]):+,.0f} | {np.mean([r['lstm'] - r['xgb'] for r in ind]):+,.0f} |")
        wins = sum(1 for r in rr if r["xgb"] < r["ridge"])
        md += ["", f"XGBoost was better than ridge in {wins} of {len(rr)} windows; the LSTM L4 was better than XGBoost in {sum(1 for r in rr if r['lstm'] < r['xgb'])} of {len(rr)} windows.", ""]
    (ROOT / "reports" / "walk_forward.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("wrote reports/walk_forward.md")


if __name__ == "__main__":
    main()
