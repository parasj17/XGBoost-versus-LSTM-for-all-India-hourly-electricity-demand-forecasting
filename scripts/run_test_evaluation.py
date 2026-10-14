"""One-time TEST-period evaluation of the frozen configuration (docs/experiment_protocol.md section 15).

  py scripts/run_test_evaluation.py --check   dry run of the identical code on the VALIDATION period
                                              (the test period is not loaded; all horizons and feature sets, one seed, at most 2 LSTM epochs)
  py scripts/run_test_evaluation.py           the real, single test run (refuses to run twice; needs a clean git tree)

Fitting rule: train period only (targets <= 2023-12-31); trees/epochs fixed from the validation runs; seeds 0,1,2;
headline prediction = mean of the three seeds. Hyper-parameters come from the logged validation runs; nothing is re-tuned.
Outputs: reports/test_results.md and reports/tables/test_results.jsonl (or *_check files in --check mode).
"""
import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")
rx = importlib.import_module("run_xgboost")
rl = importlib.import_module("run_lstm")

T = ROOT / "reports" / "tables"
LOCK = T / "TEST_RUN_DONE.lock"
TEST_START, TEST_END = pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2025-06-24 23:00")
SEEDS = [0, 1, 2]
XGB_SETS = list(ft.FEATURE_SETS)
HEADLINE_X, HEADLINE_L = "S4_full_A", "L4_full_A"
SEASONS = ["winter", "pre_monsoon_summer", "monsoon", "post_monsoon"]


def read_jsonl(name):
    return [json.loads(l) for l in open(T / name, encoding="utf-8")]


def frozen_configs():
    xw = read_jsonl("xgboost_validation_wide.jsonl")
    xgb_params, xgb_trees = {}, {}
    for r in xw:
        if r["stage"] == "ablation_fit" and r["relative_target"]:
            xgb_params[r["horizon"]] = r["params"]
            xgb_trees[(r["horizon"], r["feature_set"], r["seed"])] = int(r["best_iteration"]) + 1
    lstm = {}
    for fname, tag in (("lstm_validation.jsonl", HEADLINE_L), ("lstm_validation_stage2_L1.jsonl", "L1_history_calendar")):
        for r in read_jsonl(fname):
            if r["stage"] == "final_fit" and r["feature_set"] == tag:
                lstm[(r["horizon"], tag, r["seed"])] = (r["params"], int(r["best_epoch"]))
    return xgb_params, xgb_trees, lstm


def dm_test(d, lag):
    n = len(d)
    dc = d - d.mean()
    var = float(np.dot(dc, dc) / n)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * float(np.dot(dc[k:], dc[:-k]) / n)
    z = float(d.mean() / math.sqrt(max(var, 1e-12) / n))
    return z, math.erfc(abs(z) / math.sqrt(2))


def lstm_fit_fixed(tr, n_in, n_st, hp, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = rl.Net(n_in, n_st, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    ntr = len(tr.idx)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(ntr)
        for i in range(0, ntr, rl.BATCH):
            x, s, z = tr.batch(perm[i:i + rl.BATCH])
            loss = ((net(x, s) - z) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="unchanged re-run that saves test and validation predictions to data/interim/predictions and compares "
                         "the metrics with the first run (protocol deviation log, 2026-10-15); official numbers stay those of the first run")
    args = ap.parse_args()
    rx.MAX_TREES = 3000
    commit, dirty = rb.git_state()
    if args.check:
        ev_start, ev_end, horizons, seeds, out_md, out_jsonl = rb.VAL_START, rb.VAL_END, rb.HORIZONS, [0], ROOT / "reports" / "test_results_check.md", T / "test_results_check.jsonl"
    elif args.export:
        if not LOCK.exists():
            sys.exit("--export is only for re-running after the single test run has been completed.")
        if dirty:
            sys.exit("Working tree is not clean: commit first so the run records an exact commit.")
        ev_start, ev_end, horizons, seeds, out_md, out_jsonl = TEST_START, TEST_END, rb.HORIZONS, SEEDS, ROOT / "reports" / "test_results_rerun.md", T / "test_results_rerun.jsonl"
    else:
        if LOCK.exists():
            sys.exit(f"The test evaluation was already run ({LOCK}). Re-running would invalidate the single-use test set; "
                     "any change must be logged as a post-hoc deviation in the protocol.")
        if dirty:
            sys.exit("Working tree is not clean: commit first so the run records an exact commit.")
        ev_start, ev_end, horizons, seeds, out_md, out_jsonl = TEST_START, TEST_END, rb.HORIZONS, SEEDS, ROOT / "reports" / "test_results.md", T / "test_results.jsonl"
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp")
    df = df.loc[:ev_end]
    assert args.check == (df.index.max() <= rb.VAL_END), "the check mode must not load the test period"
    sha = hashlib.sha256(rb.SRC.read_bytes()).hexdigest()
    xgb_params, xgb_trees, lstm_cfg = frozen_configs()
    meta = {"git_commit": commit, "git_dirty": dirty, "dataset_sha256": sha, "xgboost": xgb.__version__, "torch": torch.__version__,
            "python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "mode": "check_on_validation" if args.check else ("TEST_EXPORT_RERUN" if args.export else "TEST")}
    log = open(out_jsonl, "a", encoding="utf-8")
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    train_peak_thr = df.loc[:rb.TRAIN_END, "electricity_demand_mw"].quantile(0.95)
    md = [f"# {'CHECK on the validation period (not a result)' if args.check else ('Test-period results: UNCHANGED RE-RUN that saves predictions (official numbers = first run, test_results.md)' if args.export else 'Test-period results (single run)')} - generated by `scripts/run_test_evaluation.py`", "",
          f"Evaluation targets {ev_start:%Y-%m-%d} to {ev_end:%Y-%m-%d}. Git commit `{(commit or 'n/a')[:10]}`, dataset sha256 `{sha[:16]}...`. "
          "Frozen configuration: `docs/experiment_protocol.md` section 15. Models are fitted on the training period only; trees and epochs fixed from "
          "validation; headline = mean prediction of 3 seeds. Tuning budgets differ between models (XGBoost 90 trials per horizon, LSTM 12 + 12). "
          "Scenario A except where labelled oracle.", ""]
    all_names = {}
    for h in horizons:
        t_h = time.perf_counter()
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
        in_te = valid & (Tt >= ev_start) & (Tt <= ev_end)
        assert not (in_tr & (Tt > rb.TRAIN_END)).any()
        ntr, nte = int(in_tr.sum()), int(in_te.sum())
        n_te_all = int(((Tt >= ev_start) & (Tt <= ev_end)).sum())
        a24 = b["anchors"]["a24"]
        a24v = a24.fillna(1.0).to_numpy()
        pos = np.arange(len(df))
        idx_tr, idx_te = pos[in_tr.to_numpy()], pos[in_te.to_numpy()]
        y_te, a_te = b["target"].to_numpy()[idx_te], a24v[idx_te]
        idx_va = pos[in_va.to_numpy()]
        y_va, a_va = b["target"].to_numpy()[idx_va], a24v[idx_va]
        preds, secs = {}, {}
        preds_va = {"persistence": b["y"].to_numpy()[idx_va], "snaive24": a24.to_numpy()[idx_va], "snaive168": b["anchors"]["a168"].to_numpy()[idx_va]}
        # ---- naive
        preds["persistence"] = b["y"].to_numpy()[idx_te]
        preds["snaive24"] = a24.to_numpy()[idx_te]
        preds["snaive168"] = b["anchors"]["a168"].to_numpy()[idx_te]
        # ---- ridge: variant and penalty selected on validation (same code as the baseline run), fitted on train
        best = None
        for rname, Xd in {"ridge_full_rel": Xs["full_rel"], "ridge_g01_rel": Xs["g01_rel"], "ridge_full_abs": Xs["full_abs"], "ridge_g01_abs": Xs["g01_abs"]}.items():
            rel = rname.endswith("rel")
            tgt = (b["target"] / a24) if rel else b["target"]
            for al in rb.ALPHAS:
                m = rb.ridge_fit(Xd[in_tr], tgt[in_tr], al)
                p = rb.ridge_pred(m, Xd[in_va]); p = p * a24v[pos[in_va.to_numpy()]] if rel else p
                mae = float(np.mean(np.abs(p - b["target"].to_numpy()[pos[in_va.to_numpy()]])))
                if best is None or mae < best[0]:
                    best = (mae, rname, al, Xd, rel, tgt)
        _, rname, al, Xd, rel, tgt = best
        t0 = time.perf_counter()
        m = rb.ridge_fit(Xd[in_tr], tgt[in_tr], al)
        p = rb.ridge_pred(m, Xd[in_te]); preds["ridge_best"] = p * a_te if rel else p
        secs["ridge_best"] = time.perf_counter() - t0
        pv = rb.ridge_pred(m, Xd[in_va]); preds_va["ridge_best"] = pv * a_va if rel else pv
        # ---- XGBoost, all feature sets (ratio target), fixed trees from validation
        zx = b["target"] / a24
        for sname in XGB_SETS:
            Xf = ft.build_features(df, h, ft.FEATURE_SETS[sname], True)
            sp, spv, tt = [], [], 0.0
            for sd in seeds:
                t0 = time.perf_counter()
                mdl = xgb.XGBRegressor(n_estimators=xgb_trees[(h, sname, sd)], objective="reg:squarederror", tree_method="hist", device="cpu",
                                       n_jobs=rx.NJOBS, random_state=sd, eval_metric="mae", **xgb_params[h])
                mdl.fit(Xf[in_tr], (zx[in_tr] - 1.0) * 100.0, verbose=False)
                tt += time.perf_counter() - t0
                sp.append((1.0 + mdl.predict(Xf[in_te]) / 100.0) * a_te)
                spv.append((1.0 + mdl.predict(Xf[in_va]) / 100.0) * a_va)
            preds_va[f"xgb_{sname}"] = np.mean(spv, axis=0)
            preds[f"xgb_{sname}"] = np.mean(sp, axis=0)
            preds[f"xgb_{sname}__perseed"] = np.array(sp)
            secs[f"xgb_{sname}"] = tt
        # ---- LSTM (full features L4 and history+calendar L1), epochs fixed from validation
        z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
        zsd = float(np.nanstd(z_all[idx_tr]))
        zt = np.nan_to_num(z_all / zsd)
        for lname, w, g, hol in ((HEADLINE_L, True, True, True), ("L1_history_calendar", False, False, False)):
            C = rl.step_channels(df, w, g, rb.TRAIN_END)
            S = rl.static_features(df, h, b, hol)
            tr = rl.Data(y_all, a24v, C, S, zt, idx_tr)
            te = rl.Data(y_all, a24v, C, S, zt, idx_te)
            vaD = rl.Data(y_all, a24v, C, S, zt, idx_va)
            sp, spv, tt = [], [], 0.0
            for sd in seeds:
                hp, ep = lstm_cfg[(h, lname, sd)]
                if args.check:
                    ep = min(ep, 2)
                t0 = time.perf_counter()
                net = lstm_fit_fixed(tr, C.shape[1] + 1, S.shape[1], hp, sd, ep)
                tt += time.perf_counter() - t0
                zhat = rl.predict(net, te, len(idx_te))
                sp.append((1 + zhat * zsd / 100.0) * a_te)
                spv.append((1 + rl.predict(net, vaD, len(idx_va)) * zsd / 100.0) * a_va)
            preds_va[f"lstm_{lname}"] = np.mean(spv, axis=0)
            preds[f"lstm_{lname}"] = np.mean(sp, axis=0)
            preds[f"lstm_{lname}__perseed"] = np.array(sp)
            secs[f"lstm_{lname}"] = tt
        # ---- metrics
        names = [k for k in preds if not k.endswith("__perseed")]
        res = {k: rb.metrics(y_te, preds[k]) for k in names}
        best_naive = min(("persistence", "snaive24", "snaive168"), key=lambda k: res[k]["MAE_MW"])
        md += [f"## Horizon {h} h", "", f"Train samples {ntr:,}; evaluation targets {n_te_all:,}, valid {nte:,}. Best naive: {best_naive}; ridge selected on validation: {rname}, alpha {al}.", "",
               "| model | MAE (MW) | RMSE (MW) | MAPE % | WAPE % | R2 | skill vs best naive | per-seed MAE (min-max) | refit time (s) |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
        for k in names:
            r = res[k]
            sk = "" if k in ("persistence", "snaive24", "snaive168") else f"{100 * (1 - r['MAE_MW'] / res[best_naive]['MAE_MW']):+.1f}%"
            ps = ""
            if f"{k}__perseed" in preds:
                pm = [float(np.mean(np.abs(q - y_te))) for q in preds[f"{k}__perseed"]]
                ps = f"{min(pm):,.0f}-{max(pm):,.0f}"
            md.append(f"| {k}{' (oracle weather)' if k.endswith('S5_oracle_B') else ''} | {r['MAE_MW']:,.0f} | {r['RMSE_MW']:,.0f} | {r['MAPE_pct']:.2f} | {r['WAPE_pct']:.2f} | {r['R2']:.3f} | {sk} | {ps} | {secs.get(k, 0):,.0f} |")
        # ---- paired comparisons
        lx, ll1, ll4 = f"xgb_{HEADLINE_X}", "lstm_L1_history_calendar", f"lstm_{HEADLINE_L}"
        pairs = [(lx, "ridge_best"), (lx, best_naive), (ll4, lx), (ll1, lx), (ll4, "ridge_best"), (ll1, "ridge_best")]
        md += ["", "Paired comparisons (MAE of first minus second; negative = first better; 95% block-bootstrap CI; DM = Diebold-Mariano on |error| differences, Newey-West lag h-1, a rough guide because hourly errors are strongly autocorrelated):", "",
               "| first | second | difference (MW) | 95% CI | DM z | DM p |", "|---|---|---:|---|---:|---:|"]
        rng = np.random.default_rng(0)
        for a_, b_ in pairs:
            if a_ not in preds or b_ not in preds:
                continue
            d = np.abs(preds[a_] - y_te) - np.abs(preds[b_] - y_te)
            lo, hi = rb.block_ci(d, rng)
            z, pv = dm_test(d, max(h - 1, 0))
            md.append(f"| {a_} | {b_} | {d.mean():+,.0f} | {lo:+,.0f} to {hi:+,.0f} | {z:+.2f} | {pv:.3f} |")
        # ---- breakdowns
        dft = df.shift(-h)
        season = b["season_T"].to_numpy()[idx_te]
        hol = (dft["is_holiday"].to_numpy()[idx_te] == 1)
        wkend = (dft["is_weekend"].to_numpy()[idx_te] == 1)
        daytype = np.where(hol, "holiday", np.where(wkend, "weekend", "weekday"))
        fw = ft.build_features(df, h, ["g2"], True).iloc[idx_te]
        near = ((fw["d_next_major"] <= 1) | (fw["d_since_major"] <= 1)).to_numpy()
        peak = y_te >= train_peak_thr
        show = [k for k in (lx, ll4, ll1, "ridge_best", best_naive) if k in preds]
        groups = [(f"season: {s}", season == s) for s in SEASONS] + [(f"day type: {d_}", daytype == d_) for d_ in ("weekday", "weekend", "holiday")] + \
                 [("within 1 day of a major festival", near), (f"peak hours (>= training 95th pct, {train_peak_thr / 1000:.0f} GW)", peak)]
        md += ["", "MAE (MW) by subset:", "", "| subset | n | " + " | ".join(show) + " |", "|---|---:|" + "---:|" * len(show)]
        for gname, mask in groups:
            if mask.sum() == 0:
                continue
            md.append(f"| {gname} | {int(mask.sum()):,} | " + " | ".join(f"{np.mean(np.abs(preds[k][mask] - y_te[mask])):,.0f}" for k in show) + " |")
        md += ["", f"Share of evaluation hours at or above the training 95th percentile: {100 * peak.mean():.1f}%.", ""]
        if args.export:
            pdir = ROOT / "data" / "interim" / "predictions"
            pdir.mkdir(parents=True, exist_ok=True)
            bad = (df["demand_imputed"].astype(bool) | (df["demand_quality_flag"] != "ok")).astype(int).rolling(168, min_periods=1).max().to_numpy()
            stuck = df["demand_stuck_flag"].astype(bool).astype(int).rolling(168, min_periods=1).max().to_numpy()
            arrs = {"y_te": y_te, "y_va": y_va, "a24_te": a_te, "a24_va": a_va, "origin_te": df.index.to_numpy()[idx_te].astype("datetime64[ns]").astype("int64"),
                    "target_time_te": Tt.to_numpy()[idx_te].astype("datetime64[ns]").astype("int64"),
                    "target_time_va": Tt.to_numpy()[idx_va].astype("datetime64[ns]").astype("int64"),
                    "origin_va": df.index.to_numpy()[idx_va].astype("datetime64[ns]").astype("int64"),
                    "window_has_imputed_te": bad[idx_te], "window_has_stuck_te": stuck[idx_te], "season_te": b["season_T"].to_numpy()[idx_te].astype(str)}
            for k, v in preds.items():
                arrs[f"te__{k}"] = v
            for k, v in preds_va.items():
                arrs[f"va__{k}"] = v
            np.savez_compressed(pdir / f"h{h}.npz", **arrs)
            first = [json.loads(l) for l in open(T / "test_results.jsonl", encoding="utf-8")]
            first = next(r for r in first if r["horizon"] == h)
            diffs = {k: res[k]["MAE_MW"] - first["metrics"][k]["MAE_MW"] for k in res if k in first["metrics"]}
            md += [f"Reproducibility versus the first run (MAE difference in MW, re-run minus first run): max |difference| {max(abs(v) for v in diffs.values()):.2f}; " +
                   "; ".join(f"{k} {v:+.2f}" for k, v in diffs.items() if abs(v) > 0.5) + (" (all others within 0.5 MW)" if any(abs(v) > 0.5 for v in diffs.values()) else " (all models within 0.5 MW)"), ""]
            print(f"h={h} reproducibility: max |MAE diff vs first run| = {max(abs(v) for v in diffs.values()):.2f} MW", flush=True)
        log.write(json.dumps({"horizon": h, "n_train": ntr, "n_eval": nte, "metrics": res, "ridge": [rname, al], "refit_seconds": secs,
                              "horizon_wall_s": round(time.perf_counter() - t_h, 1), **meta}) + chr(10)); log.flush()
        print(f"h={h} done: " + "; ".join(f"{k} {res[k]['MAE_MW']:.0f}" for k in (best_naive, 'ridge_best', lx, ll4, ll1) if k in res), flush=True)
    out_md.write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    log.close()
    if not args.check and not args.export:
        LOCK.write_text(f"test run completed {pd.Timestamp.now().isoformat(timespec='seconds')} at commit {commit}" + chr(10), encoding="utf-8")
    print("wrote", out_md)


if __name__ == "__main__":
    main()
