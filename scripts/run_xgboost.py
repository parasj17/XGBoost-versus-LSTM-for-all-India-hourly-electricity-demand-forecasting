"""Phase 6: XGBoost on the VALIDATION period (test period is never loaded).

Follows docs/experiment_protocol.md:
  * one direct model per horizon h in {1, 6, 24, 48}
  * primary target = ratio to the 24 h seasonal anchor (D-07, frozen); absolute MW is a sensitivity variant
  * Step 1  random search, 30 trials per horizon, headline feature set S4_full_A, ratio target, selection by
            validation MAE in MW (early stopping on validation, 40 rounds)
  * Step 2  ablation over the cumulative feature sets S0..S5 (features_tabular.FEATURE_SETS) with the tuned
            hyper-parameters of step 1, both target formulations, 3 seeds each
  * all models are scored on the SAME valid samples as the baselines (run_baselines.build/design)
  * every fit is logged (time, memory, model size, inference latency, versions, git commit, dataset hash)

Outputs: reports/tables/xgboost_validation.md and reports/tables/xgboost_validation.jsonl
Usage:   py scripts/run_xgboost.py            (about an hour on a 4-core laptop CPU)
"""
import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")

OUT_MD = ROOT / "reports" / "tables" / "xgboost_validation.md"
OUT_JSONL = ROOT / "reports" / "tables" / "xgboost_validation.jsonl"
N_TRIALS, SEEDS, NJOBS, ES_ROUNDS, MAX_TREES = 30, [0, 1, 2], 4, 40, 1500
HEADLINE = "S4_full_A"


def sample_params(rng):
    return {"learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))),
            "max_depth": int(rng.integers(3, 9)),
            "min_child_weight": float(np.exp(rng.uniform(0, np.log(50)))),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(50)))),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))),
            "gamma": float(rng.uniform(0, 5))}


def fit_predict(X_tr, y_tr, X_va, y_va, params, seed, a_va, relative):
    """Fit with early stopping on validation; return MW-scale prediction, model and timings.

    Ratio targets are trained as z = 100 * (ratio - 1) (percent deviation from the anchor): the same target,
    but on a scale where the split penalty (gamma) and L2 term (lambda) of the search space are meaningful.
    """
    if relative:
        y_tr, y_va = (y_tr - 1.0) * 100.0, (y_va - 1.0) * 100.0
    m = xgb.XGBRegressor(n_estimators=MAX_TREES, objective="reg:squarederror", tree_method="hist", device="cpu",
                         n_jobs=NJOBS, random_state=seed, early_stopping_rounds=ES_ROUNDS, eval_metric="mae", **params)
    proc = psutil.Process()
    t0, c0 = time.perf_counter(), time.process_time()
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
    p = m.predict(X_va)
    p = (1.0 + p / 100.0) * a_va if relative else p
    return p, m, wall, cpu, peak


def inference_latency(m, X_va):
    """Median ms for one full validation batch and for a single row (after warm-up)."""
    m.predict(X_va.iloc[:8])
    batch, single = [], []
    for _ in range(30):
        t = time.perf_counter(); m.predict(X_va); batch.append(time.perf_counter() - t)
        t = time.perf_counter(); m.predict(X_va.iloc[:1]); single.append(time.perf_counter() - t)
    return {"batch_rows": len(X_va), "batch_ms_median": 1000 * float(np.median(batch)),
            "single_row_ms_median": 1000 * float(np.median(single)),
            "single_row_ms_iqr": 1000 * float(np.subtract(*np.percentile(single, [75, 25])))}


def main():
    global N_TRIALS, SEEDS, MAX_TREES, OUT_MD, OUT_JSONL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny run (horizon 24, 2 trials, 1 seed) written to *_smoke files")
    args = ap.parse_args()
    horizons = rb.HORIZONS
    if args.smoke:
        N_TRIALS, SEEDS, MAX_TREES, horizons = 2, [0], 200, [24]
        OUT_MD = OUT_MD.with_name("xgboost_validation_smoke.md")
        OUT_JSONL = OUT_JSONL.with_name("xgboost_validation_smoke.jsonl")
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:rb.VAL_END]
    sha = hashlib.sha256(rb.SRC.read_bytes()).hexdigest()
    commit, dirty = rb.git_state()
    meta = {"git_commit": commit, "git_dirty": dirty, "dataset_sha256": sha, "xgboost": xgb.__version__,
            "python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "n_jobs": NJOBS,
            "gpu": "none used (Intel HD 620; CPU only)"}
    log = open(OUT_JSONL, "a", encoding="utf-8")
    train_peak_thr = df.loc[:rb.TRAIN_END, "electricity_demand_mw"].quantile(0.95)
    md = ["# XGBoost on the validation period (generated by `scripts/run_xgboost.py`)", "",
          f"Dataset sha256 `{sha[:16]}...`, git commit `{(commit or 'n/a')[:10]}` (working tree dirty: {dirty}); xgboost {xgb.__version__}; "
          f"{NJOBS} threads on {meta['logical_cpus']} logical CPUs, CPU only. Validation targets 2024-01-01 to 2024-06-30; "
          "the test period was not loaded. Scenario A (no future weather) unless labelled oracle. Early stopping and the "
          "hyper-parameter search both use the validation period, so these validation numbers are mildly optimistic; "
          "the honest estimate is the one-time test evaluation. The ratio target is trained as 100*(ratio-1), i.e. percent deviation from the 24 h anchor.", ""]
    summary = {}
    for h in horizons:
        b = rb.build(df, h)
        Xs = {n: rb.design(b, rel, full) for n, (rel, full) in
              {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
        allfeat = pd.concat(Xs.values(), axis=1)
        need = allfeat.notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna()
        if h <= 168:
            need &= b["anchors"]["a168"].notna()
        valid = need & b["target_obs"]
        T = b["T"]
        in_tr = valid & (T <= rb.TRAIN_END)
        in_va = valid & (T >= rb.VAL_START) & (T <= rb.VAL_END)
        a24 = b["anchors"]["a24"]
        y_va = b["target"][in_va].to_numpy()
        a_va = a24[in_va].to_numpy()
        feats = {(sname, rel): ft.build_features(df, h, groups, rel)
                 for sname, groups in ft.FEATURE_SETS.items() for rel in (True, False)}
        # ---------- Step 1: tuning (headline set, ratio target)
        Xh = feats[(HEADLINE, True)]
        rng = np.random.default_rng(1000 + h)
        best = None
        t_search = time.perf_counter()
        for trial in range(N_TRIALS):
            prm = sample_params(rng)
            p, m, wall, cpu, peak = fit_predict(Xh[in_tr], (b["target"] / a24)[in_tr], Xh[in_va], (b["target"] / a24)[in_va],
                                                prm, 0, a_va, True)
            mae = float(np.mean(np.abs(p - y_va)))
            log.write(json.dumps({"stage": "tuning_trial", "horizon": h, "trial": trial, "params": prm, "val_MAE_MW": mae,
                                  "best_iteration": int(m.best_iteration), "fit_s": round(wall, 2)}) + "\n")
            if best is None or mae < best[0]:
                best = (mae, prm, trial)
        search_s = time.perf_counter() - t_search
        prm = best[1]
        # ---------- Step 2: ablation over feature sets, both target formulations, 3 seeds
        rows = []
        keep = {}
        for sname in ft.FEATURE_SETS:
            for rel in (True, False):
                X = feats[(sname, rel)]
                tgt = (b["target"] / a24) if rel else b["target"]
                maes, preds = [], []
                for seed in SEEDS:
                    p, m, wall, cpu, peak = fit_predict(X[in_tr], tgt[in_tr], X[in_va], tgt[in_va], prm, seed, a_va, rel)
                    maes.append(float(np.mean(np.abs(p - y_va))))
                    preds.append(p)
                    lat = inference_latency(m, X[in_va]) if seed == SEEDS[0] else None
                    log.write(json.dumps({"stage": "ablation_fit", "horizon": h, "feature_set": sname, "relative_target": rel,
                                          "seed": seed, "n_features": X.shape[1], "n_train": int(in_tr.sum()), "n_val": int(in_va.sum()),
                                          "val_MAE_MW": maes[-1], "best_iteration": int(m.best_iteration), "train_wall_s": round(wall, 2),
                                          "train_cpu_s": round(cpu, 2), "peak_working_set_mb": round(peak / 1e6, 1),
                                          "model_bytes": len(m.get_booster().save_raw()), "inference": lat, "params": prm, **meta}) + "\n")
                    log.flush()
                rows.append({"set": sname, "rel": rel, "n_features": X.shape[1], "mae_mean": float(np.mean(maes)),
                             "mae_sd": float(np.std(maes)), "pred_mean": np.mean(preds, axis=0)})
                if sname == HEADLINE and rel:
                    keep["headline_pred"] = np.mean(preds, axis=0)
        # ---------- reference: best ridge on the same samples (refit, same code as the baseline run)
        ridge_best = None
        for name, Xd in {"ridge_full_rel": Xs["full_rel"], "ridge_g01_rel": Xs["g01_rel"], "ridge_full_abs": Xs["full_abs"], "ridge_g01_abs": Xs["g01_abs"]}.items():
            rel = name.endswith("rel")
            tgt = (b["target"] / a24) if rel else b["target"]
            for a in rb.ALPHAS:
                mdl = rb.ridge_fit(Xd[in_tr], tgt[in_tr], a)
                p = rb.ridge_pred(mdl, Xd[in_va])
                p = p * a_va if rel else p
                mae = float(np.mean(np.abs(p - y_va)))
                if ridge_best is None or mae < ridge_best[0]:
                    ridge_best = (mae, name, p)
        res_h = metrics_row = rb.metrics(y_va, keep["headline_pred"])
        d = np.abs(keep["headline_pred"] - y_va) - np.abs(ridge_best[2] - y_va)
        lo, hi = rb.block_ci(d, np.random.default_rng(rb.SEED))
        best_naive = min(float(np.mean(np.abs(v - y_va))) for v in
                         (b["y"][in_va].to_numpy(), a24[in_va].to_numpy(), b["anchors"]["a168"][in_va].to_numpy()))
        summary[h] = {"rows": rows, "params": prm, "search_s": search_s, "best_trial": best[2], "ridge": ridge_best[:2],
                      "headline": res_h, "ci": (lo, hi), "naive": best_naive, "n_tr": int(in_tr.sum()), "n_va": int(in_va.sum()),
                      "y_va": y_va, "pred": keep["headline_pred"], "season": b["season_T"][in_va].to_numpy()}
        print(f"h={h}: headline MAE {res_h['MAE_MW']:.0f} MW vs best ridge {ridge_best[0]:.0f} ({ridge_best[1]})", flush=True)
    # ---------- report
    for h, s in summary.items():
        md += [f"## Horizon {h} h (train samples {s['n_tr']:,}; validation samples {s['n_va']:,})", "",
               f"Tuning: {N_TRIALS} random trials in {s['search_s'] / 60:.1f} min; best trial #{s['best_trial']}; params " +
               ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s["params"].items()), "",
               "| feature set | #features | MAE ratio-target (MW, mean of 3 seeds +- sd) | MAE absolute-target (MW) |", "|---|---:|---:|---:|"]
        for sname in ft.FEATURE_SETS:
            r_rel = next(r for r in s["rows"] if r["set"] == sname and r["rel"])
            r_abs = next(r for r in s["rows"] if r["set"] == sname and not r["rel"])
            md.append(f"| {sname}{' (oracle weather at t+h)' if sname == 'S5_oracle_B' else ''} | {r_rel['n_features']} | "
                      f"{r_rel['mae_mean']:,.0f} +- {r_rel['mae_sd']:,.0f} | {r_abs['mae_mean']:,.0f} +- {r_abs['mae_sd']:,.0f} |")
        hl = s["headline"]
        yv, pv, se = s["y_va"], s["pred"], s["season"]
        pk = yv >= train_peak_thr
        md += ["", f"Headline ({HEADLINE}, ratio target, mean of 3 seeds): MAE {hl['MAE_MW']:,.0f} MW, RMSE {hl['RMSE_MW']:,.0f} MW, "
               f"MAPE {hl['MAPE_pct']:.2f}%, WAPE {hl['WAPE_pct']:.2f}%, R2 {hl['R2']:.3f}.",
               f"Best naive baseline MAE {s['naive']:,.0f} MW; best ridge ({s['ridge'][1]}) {s['ridge'][0]:,.0f} MW. "
               f"Paired MAE difference XGBoost minus best ridge: {np.mean(np.abs(pv - yv)) - s['ridge'][0]:+,.0f} MW "
               f"(95% block-bootstrap CI {s['ci'][0]:+,.0f} to {s['ci'][1]:+,.0f}; negative = XGBoost better).",
               "MAE by season of the target hour: " + "; ".join(
                   f"{sn}: {np.mean(np.abs(pv[se == sn] - yv[se == sn])):,.0f} MW" for sn in ["winter", "pre_monsoon_summer", "monsoon"] if (se == sn).any()),
               f"Peak hours (target >= training 95th percentile, {train_peak_thr / 1000:.0f} GW, n={int(pk.sum())}): MAE {np.mean(np.abs(pv[pk] - yv[pk])):,.0f} MW.", ""]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.close()
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
