"""Independent re-computation of the data statistics, sample counts and leakage checks quoted in the paper.

Reads only: data/interim/parsed/*.csv.gz (one file per source workbook, written by inspect_electricity_data.py), data/interim/hourly_audit_view.csv.gz,
data/processed/india_hourly_canonical.csv, data/raw/weather/*.csv, data/raw/calendar/*.csv, data/processed/calendar_daily.csv, and the feature builders of
run_baselines.py (for the sample counts). It does not write to any data file. Output: reports/tables/integrity_audit_stats.json (printed as well).
Usage: py -I -X utf8 scripts/audit_integrity.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_baselines as rb  # noqa: E402

out = {}
# ---------------------------------------------------------------- A. source files (parsed from the 29 workbooks)
rows = []
for f in sorted((ROOT / "data" / "interim" / "parsed").glob("*.csv.gz")):
    d = pd.read_csv(f)
    if "Timestamp" in d.columns:         # layout B (2024-2025 hourly workbook): text stamps dd-mm-YYYY HH:MM:SS, plus footer rows that do not parse
        d = d.rename(columns={"Timestamp": "Time", "Demand (MW)": "NLDC_DEMAND|P"})[["Time", "NLDC_DEMAND|P"]]
        d["Time"] = pd.to_datetime(d["Time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        d["NLDC_DEMAND|P"] = pd.to_numeric(d["NLDC_DEMAND|P"], errors="coerce")
        footer = int(d["Time"].isna().sum())
        d = d.dropna(subset=["Time"])
    else:
        d = d[["Time", "NLDC_DEMAND|P"]]
        d["Time"] = pd.to_datetime(d["Time"])
        footer = 0
    step = d["Time"].diff().dropna().dt.total_seconds()
    rows.append({"file": f.name.replace(".csv.gz", ""), "rows": len(d), "first": str(d["Time"].min()), "last": str(d["Time"].max()),
                 "median_step_s": float(step.median()), "dup_ts": int(d["Time"].duplicated().sum()), "nan_demand": int(d["NLDC_DEMAND|P"].isna().sum()),
                 "zero_demand": int((d["NLDC_DEMAND|P"] == 0).sum()), "unparsed_footer_rows": footer, "max_demand": float(d["NLDC_DEMAND|P"].max())})
src = pd.DataFrame(rows)
out["source_files"] = {"n_files": len(src), "rows_total": int(src.rows.sum()), "first": src["first"].min(), "last": src["last"].max(),
                       "files_by_median_step_s": src.groupby("median_step_s").size().to_dict(), "dup_ts_total": int(src.dup_ts.sum()),
                       "nan_demand_total": int(src.nan_demand.sum()), "zero_demand_total": int(src.zero_demand.sum()), "unparsed_footer_rows": int(src.unparsed_footer_rows.sum()), "per_file": rows,
                       "raw_max_demand_mw": float(src.max_demand.max())}
# ---------------------------------------------------------------- B. hourly grid
av = pd.read_csv(ROOT / "data" / "interim" / "hourly_audit_view.csv.gz", parse_dates=["hour"])
full = pd.date_range("2021-09-01 00:00", "2025-06-24 23:00", freq="h")
out["hourly_grid"] = {"expected_hours": len(full), "rows_in_audit_view": len(av), "duplicate_hours": int(av["hour"].duplicated().sum()),
                      "missing_hours_vs_grid": int(len(full.difference(av["hour"]))), "hours_by_samples_per_hour": av["n_samples"].value_counts().sort_index().to_dict()}
# ---------------------------------------------------------------- C. canonical vs raw snapshot, interpolation, split counts of flags
cn = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", parse_dates=["timestamp"]).set_index("timestamp")
cn = cn.reindex(full)
snap = av.set_index("hour")["demand_mw_first"].reindex(full)
mean = av.set_index("hour")["demand_mw_mean"].reindex(full)
ok = cn["demand_quality_flag"].eq("ok") & ~cn["demand_imputed"].astype(bool)
out["canonical_vs_snapshot"] = {"hours_ok_and_equal_to_snapshot": int((ok & (cn["electricity_demand_mw"] - snap).abs().lt(1e-3)).sum()),
                                "hours_ok": int(ok.sum()), "hours_flagged": int((~ok).sum()), "flag_counts": cn["demand_quality_flag"].value_counts().to_dict(),
                                "imputed": int(cn["demand_imputed"].astype(bool).sum()), "missing_after_cleaning": int(cn["electricity_demand_mw"].isna().sum())}
imp = cn["demand_imputed"].astype(bool)
lin = []                                 # every imputed value must lie between the neighbours that bracket its gap
for t in cn.index[imp]:
    a = cn["electricity_demand_mw"][:t - pd.Timedelta(hours=1)].last_valid_index()
    after = cn["electricity_demand_mw"][t + pd.Timedelta(hours=1):]
    b = after[~imp.reindex(after.index)].first_valid_index()
    lo, hi = sorted([cn.at[a, "electricity_demand_mw"], cn.at[b, "electricity_demand_mw"]])
    lin.append(lo - 1 <= cn.at[t, "electricity_demand_mw"] <= hi + 1)
out["interpolation_values_between_bracketing_neighbours"] = {"n_imputed": int(imp.sum()), "all_between": bool(all(lin))}
sp = {"train": (cn.index <= rb.TRAIN_END), "validation": (cn.index >= rb.VAL_START) & (cn.index <= rb.VAL_END), "test": cn.index >= "2024-07-01"}
out["flags_by_split"] = {k: {"hours": int(m.sum()), "not_ok": int((~ok & m).sum()), "imputed": int((imp & m).sum()), "missing": int(cn["electricity_demand_mw"][m].isna().sum())} for k, m in sp.items()}
# ---------------------------------------------------------------- D. hour-start snapshot versus hourly mean (2021-09 to 2023-12, both inside the plausibility bounds)
m = (full <= rb.TRAIN_END) & snap.between(80_000, 260_000).to_numpy() & mean.between(80_000, 260_000).to_numpy()
dd = (snap - mean)[m]
out["snapshot_vs_mean_2021_2023"] = {"hours": int(m.sum()), "mae_mw": float(dd.abs().mean()), "mae_pct_of_mean_demand": float(dd.abs().mean() / mean[m].mean() * 100),
                                     "p99_abs_mw": float(dd.abs().quantile(0.99)), "mean_diff_mw": float(dd.mean())}
# ---------------------------------------------------------------- E. weather aggregation: national series = equal-weight mean of the 12 cities?
cities = []
for f in sorted((ROOT / "data" / "raw" / "weather").glob("era5_*_2021-09-01_2025-06-24.csv")):
    w = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
    cities.append(w)
temp = pd.concat([c["temperature_2m"] for c in cities], axis=1)
out["weather"] = {"n_cities": len(cities), "regions": pd.concat([c[["city", "grid_region"]].iloc[:1] for c in cities]).set_index("city")["grid_region"].to_dict(),
                  "rows_each": [len(c) for c in cities][:3], "max_abs_diff_national_temp_vs_equal_mean_C": float((cn["temp_c"] - temp.mean(axis=1).reindex(cn.index)).abs().max()),
                  "missing_temp_hours": int(temp.isna().any(axis=1).sum())}
# ---------------------------------------------------------------- F. holidays and festivals
cal = pd.read_csv(ROOT / "data" / "processed" / "calendar_daily.csv", parse_dates=["date"])
hol = pd.read_csv(ROOT / "data" / "raw" / "calendar" / "dopt_central_gazetted_holidays.csv")
out["calendar"] = {"holiday_rows_in_raw_csv": len(hol), "holiday_days_in_calendar": int(cal.is_holiday.sum()), "major_festival_days": int(cal.is_major_festival.sum()),
                   "festival_names": cal.festival_name.dropna().unique().tolist(), "holiday_days_in_test_period": int(cal[cal.date >= "2024-07-01"].is_holiday.sum())}
# ---------------------------------------------------------------- G. sample counts per horizon and where the 567 hours go (scenario-A feature set, ratio target)
acc = {}
for h in (1, 6, 24, 48, 72, 168):
    b = rb.build(cn.reset_index().rename(columns={"index": "timestamp"}).set_index("timestamp"), h)
    X = rb.design(b, rel=True, full=True)
    T = b["T"]
    tgt_ok = b["target"].notna() & b["target_obs"]
    fok = X.notna().all(axis=1)
    r = {}
    for k, mk in (("train", T <= rb.TRAIN_END), ("validation", (T >= rb.VAL_START) & (T <= rb.VAL_END)), ("test", T >= pd.Timestamp("2024-07-01"))):
        mk = mk & (T <= cn.index[-1])
        r[k] = {"target_hours": int(mk.sum()), "target_missing_or_imputed": int((mk & ~tgt_ok).sum()), "inputs_incomplete": int((mk & tgt_ok & ~fok).sum()),
                "valid": int((mk & tgt_ok & fok).sum())}
    acc[h] = r
out["sample_counts_by_horizon"] = acc
# ---------------------------------------------------------------- H. solar statistic
s = cn["solar_generation_mw"]
tr_max = s[sp["train"]].max()
test = sp["test"]
out["solar"] = {"train_max_mw": float(tr_max), "share_test_hours_above": float((s[test] > tr_max).mean() * 100), "n_test_hours_above": int((s[test] > tr_max).sum()),
                "test_hours": int(test.sum()), "mean_train": float(s[sp["train"]].mean()), "mean_test": float(s[test].mean()),
                "mean_test_over_train_pct": float((s[test].mean() / s[sp["train"]].mean() - 1) * 100)}
# ---------------------------------------------------------------- I. interpolation look-ahead: origins whose input window uses a value interpolated from a later hour
gap_end = {}
run_start = None
for t in cn.index:
    if imp[t] and run_start is None:
        run_start = t
    if run_start is not None and not imp[t]:
        for u in pd.date_range(run_start, t - pd.Timedelta(hours=1), freq="h"):
            gap_end[u] = t - pd.Timedelta(hours=1)
        run_start = None
affected = sorted({t for u, e in gap_end.items() for t in pd.date_range(u, e, freq="h")})   # origins t with an imputed hour u<=t whose gap ends at or after t
out["interpolation_lookahead"] = {"imputed_hours": len(gap_end), "origin_hours_whose_window_contains_a_value_interpolated_from_a_later_hour": len(affected),
                                  "of_which_in_test": int(sum(t >= pd.Timestamp("2024-07-01") for t in affected))}
(ROOT / "reports" / "tables").mkdir(exist_ok=True)
(ROOT / "reports" / "tables" / "integrity_audit_stats.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
