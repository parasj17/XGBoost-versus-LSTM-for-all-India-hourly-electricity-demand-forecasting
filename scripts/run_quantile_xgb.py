"""XGBoost quantile regression (5% and 95%) for the interval extension (protocol deviation log, 2026-10-15).

Feature set S4_full_A, target = percent deviation from the 24 h anchor, the stage-2 point-forecast hyper-parameters of each horizon,
objective reg:quantileerror with alpha 0.05 and 0.95, ONE seed (0), fitted on the training period; the number of trees is chosen by early
stopping on the validation pinball loss (so validation is used for early stopping, and later for conformal calibration - mildly optimistic).
Predictions for validation and test are saved to data/interim/predictions/xgb_quantile_h{h}.npz and used by analyze_predictions.py.
--check restricts everything to the validation period (the test period is not loaded) and saves nothing.
Usage: py scripts/run_quantile_xgb.py [--check]
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
rt = importlib.import_module("run_test_evaluation")

QS = np.array([0.05, 0.95])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    xgb_params, _, _ = rt.frozen_configs()
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp")
    ev_end = rb.VAL_END if args.check else rt.TEST_END
    df = df.loc[:ev_end]
    out = ROOT / "data" / "interim" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for h in rb.HORIZONS:
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
        idx_va, idx_te = pos[in_va.to_numpy()], pos[in_te.to_numpy()]
        X = ft.build_features(df, h, ft.FEATURE_SETS["S4_full_A"], True)
        z = ((b["target"] / a24) - 1.0) * 100.0
        prm = dict(xgb_params[h])
        t0 = time.perf_counter()
        m = xgb.XGBRegressor(n_estimators=3000, objective="reg:quantileerror", quantile_alpha=QS, tree_method="hist", device="cpu",
                             n_jobs=rx.NJOBS, random_state=0, early_stopping_rounds=40, **prm)
        m.fit(X[in_tr], z[in_tr], eval_set=[(X[in_va], z[in_va])], verbose=False)
        fit_s = time.perf_counter() - t0
        def conv(P, idx):
            P = np.sort(P, axis=1)                       # lower <= upper (removes crossing)
            return (1.0 + P[:, 0] / 100.0) * a24v[idx], (1.0 + P[:, 1] / 100.0) * a24v[idx]
        l_va, u_va = conv(m.predict(X[in_va]), idx_va)
        y_va = b["target"].to_numpy()[idx_va]
        raw_cov_va = float(np.mean((y_va >= l_va) & (y_va <= u_va)))
        t1 = time.perf_counter()
        P_te = m.predict(X[in_te]) if not args.check else m.predict(X[in_va])
        inf_s = time.perf_counter() - t1
        stats[h] = {"trees": int(m.best_iteration) + 1, "fit_s": round(fit_s, 1), "raw_val_coverage": raw_cov_va, "inference_s_for_eval_set": round(inf_s, 3)}
        print(f"h={h}: trees {stats[h]['trees']}, fit {fit_s:.0f} s, raw validation coverage {100 * raw_cov_va:.1f}% (nominal 90%)", flush=True)
        if not args.check:
            l_te, u_te = conv(P_te, idx_te)
            np.savez_compressed(out / f"xgb_quantile_h{h}.npz", q05_va=l_va, q95_va=u_va, q05_te=l_te, q95_te=u_te, y_te=b["target"].to_numpy()[idx_te])
    if not args.check:
        (ROOT / "reports" / "tables" / "xgb_quantile_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
