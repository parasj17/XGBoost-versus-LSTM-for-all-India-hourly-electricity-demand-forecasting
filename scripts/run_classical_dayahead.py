"""Part P4 of the completion batch: classical references on the day-ahead task (docs/experiment_protocol.md, entry of 2026-10-19). No hyper-parameter is tuned; the specifications are fixed in advance.

Task and samples as in scripts/run_dayahead.py: origin 10:00 on day D-1, the 24 hourly values of day D, the same (day, hour) samples. Both models use only information known at the origin
(demand up to 10:00 on D-1; the calendar of the target hours, which is known in advance); there is no weather input.
  sarimax  Regression with ARMA errors on the 24-hour-differenced demand (GW): order (1,0,1) x (1,0,1)_24, exogenous = the 24-hour differences of six weekday dummies, the holiday flag and the major-festival flag;
           refitted every 7 days on the preceding 12 weeks (84 days), updated every day with the newest observations without refitting (res.apply), forecasts integrated back to demand levels.
  prophet  Prophet with daily and weekly seasonality (no yearly term), the national holidays and major festivals as holidays, default trend settings, refitted every 7 days on the preceding 52 weeks;
           forecasts for the days of the following week come from that fit.
  py scripts/run_classical_dayahead.py --model sarimax|prophet               validation days (sanity check; nothing is tuned) -> reports/classical_<model>_validation.md
  py scripts/run_classical_dayahead.py --model sarimax|prophet --with-test   test days, single evaluation (lock reports/tables/CLASSICAL_<MODEL>_TEST_DONE.lock) -> reports/classical_<model>_test.md,
                                                                             data/interim/predictions/classical_<model>.npz
"""
import argparse
import importlib
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.append(r"D:\pylibs_ext")          # statsmodels and prophet were installed outside C:
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
rt = importlib.import_module("run_test_evaluation")
T = ROOT / "reports" / "tables"
PRED = ROOT / "data" / "interim" / "predictions"
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
SARIMAX_WINDOW_D, PROPHET_WINDOW_D, REFIT_D = 84, 364, 7


def calendar_exog(idx, df):
    X = pd.DataFrame({f"dow{k}": (idx.dayofweek == k).astype(float) for k in range(6)}, index=idx)
    X["hol"] = df["is_holiday"].reindex(idx).astype(float).fillna(0).to_numpy()
    X["major"] = df["is_major_festival"].reindex(idx).astype(float).fillna(0).to_numpy()
    return X


def run_sarimax(df, days):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = df["electricity_demand_mw"] / 1000.0
    yl = y.ffill()
    X = calendar_exog(df.index, df)
    Xd = X - X.shift(24)
    yd = y - y.shift(24)
    out, res, last_refit, n_fail, n_noconv = {}, None, None, 0, 0
    for D in days:
        t0 = D - pd.Timedelta(hours=14)            # D 00:00 minus 14 h = 10:00 on D-1
        if res is None or (t0 - last_refit) >= pd.Timedelta(days=REFIT_D):
            w = slice(t0 - pd.Timedelta(hours=24 * SARIMAX_WINDOW_D - 1), t0)
            try:
                m = SARIMAX(yd.loc[w], exog=Xd.loc[w], order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
                new = m.fit(disp=False, maxiter=100)
                n_noconv += int(not new.mle_retvals.get("converged", True))
                res, last_refit = new, t0
            except Exception:
                n_fail += 1
                if res is None:
                    raise
        w = slice(t0 - pd.Timedelta(hours=24 * SARIMAX_WINDOW_D - 1), t0)
        r2 = res.apply(yd.loc[w], exog=Xd.loc[w], refit=False)
        fi = pd.date_range(t0 + pd.Timedelta(hours=1), periods=37, freq="h")
        dh = np.asarray(r2.forecast(37, exog=Xd.loc[fi]))
        lev = list(yl.loc[t0 - pd.Timedelta(hours=23):t0].to_numpy())
        yh = []
        for k in range(1, 38):
            base = lev[k - 1] if k <= 24 else yh[k - 25]
            yh.append(base + dh[k - 1])
        for k in range(14, 38):
            out[fi[k - 1]] = yh[k - 1] * 1000.0
    return out, {"refit_failures": n_fail, "not_converged": n_noconv}


def run_prophet(df, days):
    from prophet import Prophet
    hol = pd.concat([pd.DataFrame({"holiday": "holiday", "ds": df.index[df["is_holiday"].astype(bool)].normalize().unique()}),
                     pd.DataFrame({"holiday": "major_festival", "ds": df.index[df["is_major_festival"].astype(bool)].normalize().unique()})])
    y = (df["electricity_demand_mw"] / 1000.0).dropna()
    out, n_fail = {}, 0
    days = list(days)
    i = 0
    while i < len(days):
        D0 = days[i]
        t_r = D0 - pd.Timedelta(hours=14)
        chunk = [d for d in days[i:] if (d - D0) < pd.Timedelta(days=REFIT_D)]
        train = y.loc[t_r - pd.Timedelta(hours=24 * PROPHET_WINDOW_D - 1):t_r]
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, holidays=hol)
        m.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
        ds = pd.DatetimeIndex(sorted({d + pd.Timedelta(hours=k) for d in chunk for k in range(24)}))
        fc = m.predict(pd.DataFrame({"ds": ds}))
        for t, v in zip(ds, fc["yhat"].to_numpy()):
            out[t] = float(v) * 1000.0
        i += len(chunk)
    return out, {"refit_failures": n_fail}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["sarimax", "prophet"], required=True)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    name = args.model
    LOCK = T / f"CLASSICAL_{name.upper()}_TEST_DONE.lock"
    commit, dirty = rb.git_state()
    if args.with_test:
        if LOCK.exists():
            sys.exit(f"The {name} test evaluation was already run.")
        if dirty:
            sys.exit("Working tree is not clean: commit first.")
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:(rt.TEST_END if args.with_test else rb.VAL_END)]
    z = np.load(PRED / ("dayahead.npz" if args.with_test else "dayahead_val.npz"), allow_pickle=True)
    Tn, y = np.asarray(z["T"], dtype="int64"), z["y"].astype(float)
    tt = pd.to_datetime(Tn.astype("datetime64[ns]"))
    days = sorted(pd.DatetimeIndex(tt.normalize()).unique())
    if args.smoke:
        days = days[:10]
    t0 = time.perf_counter()
    out, info = (run_sarimax if name == "sarimax" else run_prophet)(df, days)
    p = np.array([out.get(t, np.nan) for t in tt])
    ok = ~np.isnan(p)
    if args.smoke:
        sel = np.asarray(tt.normalize().isin(days))
        ok = ok & sel
    mae = float(np.mean(np.abs(p[ok] - y[ok])))
    ref = {k: float(np.mean(np.abs(z[k][ok] - y[ok]))) for k in ("naive_a24", "ridge", "xgb", "lstm")}
    ev = "test" if args.with_test else "validation"
    md = [f"# {'SARIMAX' if name == 'sarimax' else 'Prophet'} on the day-ahead task: {ev} (generated by `scripts/run_classical_dayahead.py --model {name}`; part P4 of the completion batch)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. Specification fixed in advance (docs/experiment_protocol.md, entry of 2026-10-19); nothing was tuned. Refitted every {REFIT_D} days on the preceding "
          + (f"{SARIMAX_WINDOW_D} days, updated daily without refitting" if name == "sarimax" else f"{PROPHET_WINDOW_D} days") + "; calendar information only, no weather. "
          f"{int(ok.sum()):,} (day, hour) samples ({len(days)} days). MAE in MW. Diagnostics: {info}. Run time {time.perf_counter() - t0:,.0f} s.", "",
          f"| model | {ev} MAE (MW) | MAPE % |", "|---|---:|---:|", f"| {name} | {mae:,.0f} | {100 * np.mean(np.abs(p[ok] - y[ok]) / y[ok]):.2f} |"]
    md += [f"| naive (same hour on the latest earlier day) | {ref['naive_a24']:,.0f} | |", f"| ridge, day-ahead | {ref['ridge']:,.0f} | |", f"| XGBoost S4, day-ahead | {ref['xgb']:,.0f} | |", f"| LSTM L4, day-ahead | {ref['lstm']:,.0f} | |", ""]
    print(chr(10).join(md))
    if args.smoke:
        print("smoke finished (nothing written)")
        return
    np.savez_compressed(PRED / (f"classical_{name}.npz" if args.with_test else f"classical_{name}_val.npz"), pred=p, y=y, T=Tn)
    if args.with_test:
        (ROOT / "reports" / f"classical_{name}_test.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
        LOCK.write_text(f"{name} test run completed at commit {commit}" + chr(10), encoding="utf-8")
    else:
        (ROOT / "reports" / f"classical_{name}_validation.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
