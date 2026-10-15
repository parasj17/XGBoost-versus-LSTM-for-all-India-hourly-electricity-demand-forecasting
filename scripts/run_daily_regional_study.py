"""Regional daily study (extension 9.2; design declared in docs/experiment_protocol.md before any model was fitted).

Target: all-India daily energy met (MU), horizons 1 and 7 days, Zenodo state-level data 2013-01-01 .. 2024-04-28.
Split by target date: train <= 2020-12-31, validation 2021-01-01 .. 2022-12-31, test 2023-01-01 .. 2024-04-28 (single use).
Models: naive (same weekday last week), ridge, XGBoost direct (national series), XGBoost bottom-up (one model per grid region, summed, plus a naive forecast for
the AMNSIL/DVC remainder). Ratio-to-weekly-anchor target; 20 random trials for the direct XGBoost on validation, the best hyper-parameters reused for the regional
models; early stopping on validation; 3 seeds; mean prediction.
  py scripts/run_daily_regional_study.py               validation only (test period not loaded)
  py scripts/run_daily_regional_study.py --with-test   adds the test period, runs ONCE (lock reports/tables/DAILY_TEST_DONE.lock)
Output: reports/regional_daily_study.md (validation) or reports/regional_daily_study_test.md
"""
import argparse
import importlib
import math
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
rx = importlib.import_module("run_xgboost")
crw = importlib.import_module("compute_region_weights")

ZEN = ROOT / "data" / "raw" / "zenodo_state_daily"
LOCK = ROOT / "reports" / "tables" / "DAILY_TEST_DONE.lock"
TRAIN_END, VAL_START, VAL_END = pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")
TEST_START, TEST_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-04-28")
HORIZONS, SEEDS, N_TRIALS, BLOCK = [1, 7], [0, 1, 2], 20, 28
LAGS = [0, 1, 2, 3, 7, 14, 28]
WCOLS = {"t_mean": "2m_temperature_mean", "t_min": "2m_temperature_min", "t_max": "2m_temperature_max", "dew": "2m_dewpoint_temperature_mean",
         "solar": "surface_solar_radiation_downwards_mean", "cloud": "total_cloud_cover_mean"}


def block_ci(d, rng, B=1000):
    n = len(d)
    cs = np.concatenate([[0], np.cumsum(d)])
    nb = math.ceil(n / BLOCK)
    starts = rng.integers(0, n - BLOCK + 1, size=(B, nb))
    m = (cs[starts + BLOCK] - cs[starts]).sum(1) / (nb * BLOCK)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_energy(last):
    e = pd.read_csv(ZEN / "corrected_daily_energy_met_MU.csv", parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    e = e.reindex(pd.date_range(e.index.min(), min(e.index.max(), last), freq="D"))
    return e


def region_series(e):
    out, mapped = {}, set()
    for r, states in crw.REGIONS.items():
        cols = []
        for s in states:
            cols += ["Dadra and Nagar Haveli", "Daman and Diu"] if s == "Dadra and Nagar Haveli and Daman and Diu" else [s]
        cols = [c for c in cols if c in e.columns]
        mapped |= set(cols)
        out[r] = e[cols].sum(axis=1, min_count=1)
    out["Other"] = e[[c for c in ("AMNSIL", "Damodar Valley Corporation") if c in e.columns]].sum(axis=1, min_count=1)
    return out


def weather_by_region(idx):
    pop = pd.read_csv(ROOT / "data" / "raw" / "population" / "census2011_state_population.csv").set_index("state_or_ut")["population_2011"]
    regs = {}
    for r, states in crw.REGIONS.items():
        parts, ws = [], []
        for s in states:
            f = ZEN / (s.replace(" ", "_") + ".csv")
            if s == "Dadra and Nagar Haveli and Daman and Diu":
                f = ZEN / "Dadra_and_Nagar_Haveli.csv"
            if s == "Delhi":
                f = ZEN / "NCT_of_Delhi.csv"
            if s == "Andaman and Nicobar Islands" or not f.exists():
                continue
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            w = pd.DataFrame({k: d[c] for k, c in WCOLS.items()})
            w["wind"] = np.sqrt(d["10m_u_component_of_wind_mean"] ** 2 + d["10m_v_component_of_wind_mean"] ** 2)
            parts.append(w.reindex(idx)); ws.append(float(pop[s]))
        ws = np.array(ws) / np.sum(ws)
        regs[r] = sum(p * wi for p, wi in zip(parts, ws))
    nat = sum(regs.values()) / len(regs)                       # equal weight over the five regions (decision D-03)
    return regs, nat


def features(y, W, h, hol):
    m = math.ceil(h / 7)
    a = y.shift(7 * m - h)
    X = pd.DataFrame(index=y.index)
    for k in LAGS:
        X[f"lag{k}"] = y.shift(k) / a
    X["roll7"], X["roll28"] = y.rolling(7, min_periods=5).mean() / a, y.rolling(28, min_periods=20).mean() / a
    T = y.index + pd.Timedelta(days=h)
    X["dow_T"], X["month_T"], X["doy_T"] = T.dayofweek, T.month, T.dayofyear
    X["holiday_T"] = np.array([1 if t in hol else 0 for t in T])
    for c in W.columns:
        X[f"w_{c}"] = W[c].to_numpy()
    X["w_tmean_roll7"] = W["t_mean"].rolling(7, min_periods=5).mean().to_numpy()
    X["w_tmean_lag1"] = W["t_mean"].shift(1).to_numpy()
    return X, a, pd.Series(T, index=y.index)


def ridge_fit_predict(Xtr, ytr, Xe_list, alphas):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1)
    Z = (Xtr - mu) / sd
    ym = ytr.mean()
    out = {}
    for al in alphas:
        w = np.linalg.solve((Z.T @ Z + al * np.eye(Z.shape[1])).to_numpy(), (Z.T @ (ytr - ym)).to_numpy())
        out[al] = [((X - mu) / sd).to_numpy() @ w + ym for X in Xe_list]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-test", action="store_true")
    args = ap.parse_args()
    if args.with_test and LOCK.exists():
        sys.exit("The daily-study test evaluation was already run.")
    commit, dirty = rb.git_state()
    if args.with_test and dirty:
        sys.exit("Working tree is not clean: commit first.")
    last = TEST_END if args.with_test else VAL_END
    e = load_energy(last)
    idx = e.index
    reg_y = region_series(e)
    reg_w, nat_w = weather_by_region(idx)
    nat = e["Total"]
    hol = holidays.India(years=range(2013, 2026))
    md = ["# Regional daily study (generated by `scripts/run_daily_regional_study.py`)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. Zenodo state-level daily energy met (MU); validation 2021-2022" + ("; test 2023-01-01 to 2024-04-28 (single run)" if args.with_test else "; the test period was not loaded") +
          ". Target in every model: ratio to the same-weekday anchor. MAE in MU, MAPE and WAPE in %. Paired intervals: 95% moving-block bootstrap, block 28 days.", ""]
    rng = np.random.default_rng(0)
    for h in HORIZONS:
        Xn, a_n, T = features(nat, nat_w, h, hol)
        target = nat.shift(-h)
        z = (target / a_n - 1) * 100
        base_ok = target.notna() & a_n.notna() & Xn.notna().all(axis=1)
        regs = {}
        for r in list(crw.REGIONS) + ["Other"]:
            Xr, ar, _ = features(reg_y[r], reg_w[r] if r != "Other" else nat_w, h, hol)
            regs[r] = (Xr, ar, reg_y[r].shift(-h))
            base_ok &= ar.notna()
        valid = base_ok
        in_tr, in_va = valid & (T <= TRAIN_END), valid & (T >= VAL_START) & (T <= VAL_END)
        in_te = valid & (T >= TEST_START) & (T <= TEST_END) if args.with_test else None
        y_va, a_va = target[in_va].to_numpy(), a_n[in_va].to_numpy()
        preds_va, preds_te = {"naive": a_va}, {}
        y_te = a_te = None
        if args.with_test:
            y_te, a_te = target[in_te].to_numpy(), a_n[in_te].to_numpy()
            preds_te["naive"] = a_te
        # ridge
        Xr_tr = Xn[in_tr]
        outs = ridge_fit_predict(Xr_tr, z[in_tr], [Xn[in_va]] + ([Xn[in_te]] if args.with_test else []), [0.1, 1, 10, 100, 1000])
        best_al = min(outs, key=lambda al: np.mean(np.abs((1 + outs[al][0] / 100) * a_va - y_va)))
        preds_va["ridge"] = (1 + outs[best_al][0] / 100) * a_va
        if args.with_test:
            preds_te["ridge"] = (1 + outs[best_al][1] / 100) * a_te
        # xgboost direct: random search on validation, then 3 seeds
        r_ = np.random.default_rng(5000 + h)
        best = None
        for _ in range(N_TRIALS):
            prm = rx.sample_params(r_)
            p, *_ = rx.fit_predict(Xn[in_tr], (target / a_n)[in_tr], Xn[in_va], (target / a_n)[in_va], prm, 0, a_va, True)
            mae = float(np.mean(np.abs(p - y_va)))
            if best is None or mae < best[0]:
                best = (mae, prm)
        prm = best[1]
        pv, pt = [], []
        for sd in SEEDS:
            p, m, *_ = rx.fit_predict(Xn[in_tr], (target / a_n)[in_tr], Xn[in_va], (target / a_n)[in_va], prm, sd, a_va, True)
            pv.append(p)
            if args.with_test:
                pt.append((1 + m.predict(Xn[in_te]) / 100) * a_te)
        preds_va["xgb_direct"] = np.mean(pv, axis=0)
        if args.with_test:
            preds_te["xgb_direct"] = np.mean(pt, axis=0)
        # bottom-up
        bu_va, bu_te, reg_mae = np.zeros(len(y_va)), (np.zeros(len(y_te)) if args.with_test else None), {}
        for r, (Xr, ar, yr_t) in regs.items():
            if r == "Other":
                bu_va += ar[in_va].to_numpy()
                if args.with_test:
                    bu_te += ar[in_te].to_numpy()
                continue
            zr = (yr_t / ar)
            pvr, ptr = [], []
            for sd in SEEDS:
                p, m, *_ = rx.fit_predict(Xr[in_tr], zr[in_tr], Xr[in_va], zr[in_va], prm, sd, ar[in_va].to_numpy(), True)
                pvr.append(p)
                if args.with_test:
                    ptr.append((1 + m.predict(Xr[in_te]) / 100) * ar[in_te].to_numpy())
            bu_va += np.mean(pvr, axis=0)
            if args.with_test:
                bu_te += np.mean(ptr, axis=0)
                reg_mae[r] = (float(np.mean(np.abs(np.mean(ptr, axis=0) - yr_t[in_te].to_numpy()))), float(np.mean(np.abs(ar[in_te].to_numpy() - yr_t[in_te].to_numpy()))))
        preds_va["xgb_bottom_up"] = bu_va
        if args.with_test:
            preds_te["xgb_bottom_up"] = bu_te
        def table(preds, y, title):
            lines = [f"| {title} model | MAE (MU) | MAPE % | WAPE % |", "|---|---:|---:|---:|"]
            for k, p in preds.items():
                err = np.abs(p - y)
                lines.append(f"| {k} | {err.mean():,.1f} | {100 * np.mean(err / y):.2f} | {100 * err.sum() / y.sum():.2f} |")
            return lines
        md += [f"## Horizon {h} day{'s' if h > 1 else ''}", "", f"Samples: train {int(in_tr.sum()):,}, validation {int(in_va.sum()):,}" + (f", test {int(in_te.sum()):,}" if args.with_test else "") +
               f". Direct XGBoost best of {N_TRIALS} trials (validation MAE {best[0]:.1f} MU); ridge alpha {best_al}.", ""]
        md += table(preds_va, y_va, "Validation") + [""]
        if args.with_test:
            md += table(preds_te, y_te, "TEST") + [""]
            md += ["Paired differences on the test period (MAE of first minus second, MU; negative = first better):", "", "| first | second | difference | 95% CI |", "|---|---|---:|---|"]
            for a_, b_ in (("xgb_bottom_up", "xgb_direct"), ("xgb_direct", "ridge"), ("xgb_direct", "naive"), ("xgb_bottom_up", "naive")):
                d = np.abs(preds_te[a_] - y_te) - np.abs(preds_te[b_] - y_te)
                lo, hi = block_ci(d, rng)
                md.append(f"| {a_} | {b_} | {d.mean():+,.1f} | {lo:+,.1f} to {hi:+,.1f} |")
            md += ["", "Regional test MAE (MU) of the XGBoost regional model / naive anchor: " + "; ".join(f"{r}: {a:,.1f} / {b:,.1f}" for r, (a, b) in reg_mae.items()), ""]
        print(f"h={h} done: validation " + "; ".join(f"{k} {np.mean(np.abs(p - y_va)):.1f}" for k, p in preds_va.items()), flush=True)
    out = ROOT / "reports" / ("regional_daily_study_test.md" if args.with_test else "regional_daily_study.md")
    out.write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    if args.with_test:
        LOCK.write_text(f"daily study test run completed at commit {commit}" + chr(10), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
