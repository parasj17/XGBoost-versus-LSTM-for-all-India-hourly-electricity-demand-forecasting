"""Rolling-origin stability check (protocol section 9; deviation-log entry of 2026-10-15). Neither validation nor test data is loaded.

Folds (expanding window, targets scored on the next half-year):
  fold 1: train targets <= 2022-12-31, evaluate 2023-01-01 .. 2023-06-30
  fold 2: train targets <= 2023-06-30, evaluate 2023-07-01 .. 2023-12-31
  fold 3 (main split): train <= 2023-12-31, evaluate validation 2024-01-01 .. 2024-06-30 - taken from the existing logs.
Models use the frozen configuration of their horizon: ridge variant and penalty as in the test run, XGBoost S4_full_A with the
stage-2 parameters and the validation-chosen number of trees, LSTM L4_full_A with its stage-1 parameters and the validation-chosen
epochs, 3 seeds. LSTM scaling statistics come from the fold's own training period. MAE is the mean over seeds of the per-seed MAE.
Outputs: reports/rolling_origin.md, reports/tables/rolling_origin.jsonl.   Usage: py scripts/run_rolling_origin.py
"""
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
FOLDS = [("fold 1 (train to 2022-12, eval 2023-H1)", pd.Timestamp("2022-12-31 23:00"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30 23:00")),
         ("fold 2 (train to 2023-06, eval 2023-H2)", pd.Timestamp("2023-06-30 23:00"), pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31 23:00"))]
LAST = pd.Timestamp("2023-12-31 23:00")
# fold 3 = main split, per-seed means from the existing validation logs / baseline tables
RIDGE_VAL = {1: 1089, 6: 2941, 24: 3763, 48: 4663}
NAIVE_VAL = {1: 4430, 6: 4931, 24: 4931, 48: 6187}


def ridge_choice():
    out = {}
    for line in open(T / "test_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[r["horizon"]] = tuple(r["ridge"])
    return out


def main():
    rx.MAX_TREES = 3000
    xgb_params, xgb_trees, lstm_cfg = rt.frozen_configs()
    ridge_cfg = ridge_choice()
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:LAST]
    assert df.index.max() <= LAST
    commit, dirty = rb.git_state()
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    log = open(T / "rolling_origin.jsonl", "a", encoding="utf-8")
    rows = {}
    for h in rb.HORIZONS:
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
        pos = np.arange(len(df))
        rname, alpha = ridge_cfg[h]
        Xd = Xs[rname.replace("ridge_", "")]
        rel_r = rname.endswith("rel")
        Xf = ft.build_features(df, h, ft.FEATURE_SETS["S4_full_A"], True)
        zx = b["target"] / a24
        z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
        for fname, tr_end, ev_start, ev_end in FOLDS:
            in_tr = valid & (Tt <= tr_end)
            in_ev = valid & (Tt >= ev_start) & (Tt <= ev_end)
            assert Tt[in_tr].max() <= tr_end
            idx_tr, idx_ev = pos[in_tr.to_numpy()], pos[in_ev.to_numpy()]
            y_ev, a_ev = b["target"].to_numpy()[idx_ev], a24v[idx_ev]
            res = {"n_train": int(in_tr.sum()), "n_eval": int(in_ev.sum())}
            res["best_naive"] = min(float(np.mean(np.abs(v - y_ev))) for v in (b["y"].to_numpy()[idx_ev], a24.to_numpy()[idx_ev], b["anchors"]["a168"].to_numpy()[idx_ev]))
            tgt = (b["target"] / a24) if rel_r else b["target"]
            m = rb.ridge_fit(Xd[in_tr], tgt[in_tr], alpha)
            p = rb.ridge_pred(m, Xd[in_ev])
            res["ridge"] = float(np.mean(np.abs((p * a_ev if rel_r else p) - y_ev)))
            xm = []
            for sd in rt.SEEDS:
                mdl = xgb.XGBRegressor(n_estimators=xgb_trees[(h, "S4_full_A", sd)], objective="reg:squarederror", tree_method="hist", device="cpu",
                                       n_jobs=rx.NJOBS, random_state=sd, eval_metric="mae", **xgb_params[h])
                mdl.fit(Xf[in_tr], (zx[in_tr] - 1.0) * 100.0, verbose=False)
                xm.append(float(np.mean(np.abs((1.0 + mdl.predict(Xf[in_ev]) / 100.0) * a_ev - y_ev))))
            res["xgb"], res["xgb_range"] = float(np.mean(xm)), (min(xm), max(xm))
            zsd = float(np.nanstd(z_all[idx_tr]))
            zt = np.nan_to_num(z_all / zsd)
            C = rl.step_channels(df, True, True, tr_end)
            S = rl.static_features(df, h, b, True)
            trD, evD = rl.Data(y_all, a24v, C, S, zt, idx_tr), rl.Data(y_all, a24v, C, S, zt, idx_ev)
            lm = []
            for sd in rt.SEEDS:
                hp, ep = lstm_cfg[(h, rt.HEADLINE_L, sd)]
                net = rt.lstm_fit_fixed(trD, C.shape[1] + 1, S.shape[1], hp, sd, ep)
                zhat = rl.predict(net, evD, len(idx_ev))
                lm.append(float(np.mean(np.abs((1 + zhat * zsd / 100.0) * a_ev - y_ev))))
            res["lstm"], res["lstm_range"] = float(np.mean(lm)), (min(lm), max(lm))
            rows[(h, fname)] = res
            log.write(json.dumps({"horizon": h, "fold": fname, "git_commit": commit, **res}) + chr(10)); log.flush()
            print(f"h={h} {fname}: naive {res['best_naive']:.0f} ridge {res['ridge']:.0f} xgb {res['xgb']:.0f} lstm {res['lstm']:.0f}", flush=True)
    # fold 3 from the existing validation logs (per-seed means)
    ntr3 = {}
    xv, lv = {}, {}
    for r in rt.read_jsonl("xgboost_validation_wide.jsonl"):
        if r["stage"] == "ablation_fit" and r["feature_set"] == "S4_full_A" and r["relative_target"]:
            xv.setdefault(r["horizon"], []).append(r["val_MAE_MW"]); ntr3[r["horizon"]] = (r["n_train"], r["n_val"])
    for r in rt.read_jsonl("lstm_validation.jsonl"):
        if r["stage"] == "final_fit" and r["feature_set"] == "L4_full_A":
            lv.setdefault(r["horizon"], []).append(r["val_MAE_MW"])
    md = ["# Rolling-origin stability (generated by `scripts/run_rolling_origin.py`)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. Only data up to 2023-12-31 was loaded (no validation or test targets). MAE in MW; XGBoost and LSTM are the mean over 3 seeds of the "
          "per-seed MAE (range in brackets). Frozen configurations of each horizon (including the number of trees/epochs found on the main validation split) are re-used in every fold, so this tests the "
          "stability of the procedure across periods, not a re-tuned model. Fold 3 is the main split (validation 2024-H1), taken from the existing logs.", ""]
    for h in rb.HORIZONS:
        md += [f"## Horizon {h} h", "", "| fold | train / eval samples | best naive | ridge | XGBoost S4 | LSTM L4 | XGBoost vs ridge | LSTM vs XGBoost |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for fname, *_ in FOLDS:
            r = rows[(h, fname)]
            md.append(f"| {fname} | {r['n_train']:,} / {r['n_eval']:,} | {r['best_naive']:,.0f} | {r['ridge']:,.0f} | {r['xgb']:,.0f} ({r['xgb_range'][0]:,.0f}-{r['xgb_range'][1]:,.0f}) | "
                      f"{r['lstm']:,.0f} ({r['lstm_range'][0]:,.0f}-{r['lstm_range'][1]:,.0f}) | {r['xgb'] - r['ridge']:+,.0f} | {r['lstm'] - r['xgb']:+,.0f} |")
        x3, l3 = float(np.mean(xv[h])), float(np.mean(lv[h]))
        md.append(f"| fold 3 = main split (validation 2024-H1) | {ntr3[h][0]:,} / {ntr3[h][1]:,} | {NAIVE_VAL[h]:,} | {RIDGE_VAL[h]:,} | {x3:,.0f} | {l3:,.0f} | {x3 - RIDGE_VAL[h]:+,.0f} | {l3 - x3:+,.0f} |")
        md.append("")
    md += ["Differences are in MW (negative = first model better). Only three folds of half a year each, with one set of frozen hyper-parameters; the table shows whether the ranking of the models is stable, not a significance test."]
    (ROOT / "reports" / "rolling_origin.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    log.close()
    print("wrote reports/rolling_origin.md")


if __name__ == "__main__":
    main()
