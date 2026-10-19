"""Part P3 (a) of the completion batch: LightGBM on feature set S4 (docs/experiment_protocol.md, entry of 2026-10-19).

Tasks: horizons 1, 6, 24, 48, 72, 168 h (same samples, ratio-to-anchor target and valid-sample rule as the main study / the long-horizon extension) and the pooled day-ahead task (issue 10:00 on D-1, leads 14-37 h).
30 random trials per task of a search space equivalent to the XGBoost stage-1 space, early stopping on validation (40 rounds, L1 metric), 3 seeds, mean prediction.
  py scripts/run_lightgbm.py               PHASE 1 (validation only; test not loaded): writes reports/tables/lightgbm_config.json and reports/lightgbm_validation.md
  py scripts/run_lightgbm.py --with-test   PHASE 2: single test evaluation (lock reports/tables/LIGHTGBM_TEST_DONE.lock) -> reports/lightgbm_test.md, data/interim/predictions/lightgbm.npz
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.append(r"D:\pylibs_ext")          # LightGBM was installed outside C: (appended so the system numpy/pandas keep precedence)
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")
rt = importlib.import_module("run_test_evaluation")
rda = importlib.import_module("run_dayahead")
T = ROOT / "reports" / "tables"
CONFIG, LOCK = T / "lightgbm_config.json", T / "LIGHTGBM_TEST_DONE.lock"
PRED = ROOT / "data" / "interim" / "predictions"
HORIZONS, SEEDS, N_TRIALS = [1, 6, 24, 48, 72, 168], [0, 1, 2], 30
NJOBS = 4


def sample_params(rng):
    return {"learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))), "num_leaves": int(rng.integers(4, 129)), "min_child_samples": int(np.exp(rng.uniform(np.log(5), np.log(200)))),
            "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.5, 1.0)), "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(50)))),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))), "min_split_gain": float(rng.uniform(0, 5))}


def model(params, seed, n_estimators):
    return lgb.LGBMRegressor(n_estimators=n_estimators, objective="regression", subsample_freq=1, random_state=seed, n_jobs=NJOBS, verbose=-1, deterministic=True, force_row_wise=True, **params)


def fit_es(Xtr, ytr, Xva, yva, params, seed):
    m = model(params, seed, 3000)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1", callbacks=[lgb.early_stopping(40, verbose=False)])
    return m, int(m.best_iteration_)


def task_arrays(df, h, with_test):
    """Rows (S4 features, ratio target, anchors, actual) of the training, validation and test samples of horizon h."""
    b = rb.build(df, h)
    Xs = {n: rb.design(b, rel, full) for n, (rel, full) in {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
    need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna() & b["anchors"]["a168"].notna()
    valid = need & b["target_obs"]
    Tt = b["T"]
    masks = {"tr": valid & (Tt <= rb.TRAIN_END), "va": valid & (Tt >= rb.VAL_START) & (Tt <= rb.VAL_END)}
    if with_test:
        masks["te"] = valid & (Tt >= rt.TEST_START) & (Tt <= rt.TEST_END)
    Xf = ft.build_features(df, h, ft.FEATURE_SETS["S4_full_A"], True)
    a24 = b["anchors"]["a24"]
    zx = (b["target"] / a24).to_numpy()
    pos = np.arange(len(df))
    out = {}
    for s, m in masks.items():
        p = pos[m.to_numpy()]
        out[s] = {"X": Xf.iloc[p], "zx": zx[p], "a24": a24.fillna(1.0).to_numpy()[p], "y": b["target"].to_numpy()[p]}
    return out


def main():
    global HORIZONS, N_TRIALS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    commit, dirty = rb.git_state()
    if args.smoke:
        HORIZONS, N_TRIALS = [24], 2
    if args.with_test:
        if LOCK.exists():
            sys.exit("The LightGBM test evaluation was already run.")
        if dirty:
            sys.exit("Working tree is not clean: commit first.")
        cfg_all = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:(rt.TEST_END if args.with_test else rb.VAL_END)]
    ev = "te" if args.with_test else "va"
    tasks = {}
    for h in HORIZONS:
        tasks[str(h)] = task_arrays(df, h, args.with_test)
    if not args.smoke:                                   # pooled day-ahead task
        D = rda.collect(df, rda.LEADS, args.with_test)
        dtask = {}
        for s in (["tr", "va"] + (["te"] if args.with_test else [])):
            dtask[s] = {"X": pd.concat([D[h]["Xf"].iloc[D[h]["pos"][s]] for h in rda.LEADS]), "zx": np.concatenate([D[h]["zx"].to_numpy()[D[h]["pos"][s]] for h in rda.LEADS]),
                        "a24": np.concatenate([D[h]["a24v"][D[h]["pos"][s]] for h in rda.LEADS]), "y": np.concatenate([D[h]["target"][D[h]["pos"][s]] for h in rda.LEADS])}
        tasks["dayahead"] = dtask
    md = ["# " + ("LightGBM (S4): single test evaluation" if args.with_test else "LightGBM (S4): validation") + " (generated by `scripts/run_lightgbm.py`; part P3 (a) of the completion batch)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. Feature set S4 (scenario A), ratio target, same samples as the XGBoost models of each task, 30 random trials per task (2 in a smoke run), early stopping on validation (40 rounds), 3 seeds, mean prediction. MAE in MW. "
          + ("Models fitted on the training period only with the validation-chosen configuration and trees." if args.with_test else "Hyper-parameters chosen on validation; early stopping on validation (mildly optimistic)."), "",
          "| task | training samples | " + ("test" if args.with_test else "validation") + " samples | LightGBM MAE (MW) | reference: XGBoost S4 (same task) |", "|---|---:|---:|---:|---:|"]
    cfg_out, preds = {}, {}
    rng = np.random.default_rng(0)
    for name, t in tasks.items():
        tr, E = t["tr"], t[ev]
        ztr = (tr["zx"] - 1.0) * 100.0
        zva = (t["va"]["zx"] - 1.0) * 100.0
        if not args.with_test:
            r_, best = np.random.default_rng(8800 + (int(name) if name.isdigit() else 0)), None
            for _ in range(N_TRIALS):
                prm = sample_params(r_)
                m, _n = fit_es(tr["X"], ztr, t["va"]["X"], zva, prm, 0)
                mae = float(np.mean(np.abs((1.0 + m.predict(t["va"]["X"]) / 100.0) * t["va"]["a24"] - t["va"]["y"])))
                if best is None or mae < best[0]:
                    best = (mae, prm)
            trees, ps = {}, []
            for sd in SEEDS:
                m, n = fit_es(tr["X"], ztr, t["va"]["X"], zva, best[1], sd)
                trees[str(sd)] = n
                ps.append((1.0 + m.predict(t["va"]["X"]) / 100.0) * t["va"]["a24"])
            cfg_out[name] = {"params": best[1], "trees": trees}
        else:
            c = cfg_all[name]
            ps = []
            for sd in SEEDS:
                m = model(c["params"], sd, c["trees"][str(sd)])
                m.fit(tr["X"], ztr)
                ps.append((1.0 + m.predict(E["X"]) / 100.0) * E["a24"])
        p = np.mean(ps, axis=0)
        preds[name] = p
        preds[name + "_y"] = E["y"]
        mae = float(np.mean(np.abs(p - E["y"])))
        print(f"{name}: LightGBM {ev} MAE {mae:.0f} (n={len(E['y'])})", flush=True)
        ref_file = PRED / ("dayahead.npz" if args.with_test else "dayahead_val.npz") if name == "dayahead" else None
        if name == "dayahead" and not args.smoke:
            zr = np.load(ref_file, allow_pickle=True)
            assert np.allclose(zr["y"], E["y"])
            ref = float(np.mean(np.abs(zr["xgb"] - zr["y"])))
        elif name.isdigit() and int(name) <= 48 and args.with_test:
            ref = float(np.mean(np.abs(np.load(PRED / f"h{name}.npz", allow_pickle=True)["te__xgb_S4_full_A"] - np.load(PRED / f"h{name}.npz", allow_pickle=True)["y_te"])))
        elif name.isdigit() and int(name) > 48 and args.with_test:
            zr = np.load(PRED / f"long_h{name}.npz", allow_pickle=True)
            ref = float(np.mean(np.abs(zr["XGBoost_S4"] - zr["y_te"])))
        else:
            ref = float("nan")
        md.append(f"| {name if name == 'dayahead' else name + ' h'} | {len(tr['y']):,} | {len(E['y']):,} | {mae:,.0f} | " + (f"{ref:,.0f}" if ref == ref else "see the validation reports") + " |")
    md.append("")
    if args.smoke:
        print(chr(10).join(md))
        print("smoke finished (nothing written)")
        return
    np.savez_compressed(PRED / ("lightgbm.npz" if args.with_test else "lightgbm_val.npz"), **preds)
    if args.with_test:
        (ROOT / "reports" / "lightgbm_test.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
        LOCK.write_text(f"LightGBM test run completed at commit {commit}" + chr(10), encoding="utf-8")
    else:
        CONFIG.write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")
        (ROOT / "reports" / "lightgbm_validation.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
