"""Phase 2: build the cleaned all-India hourly demand/solar/wind table.

Input  : data/interim/hourly_audit_view.csv.gz   (made by inspect_electricity_data.py)
         It holds, per clock hour, the reading at HH:00 (`*_first`) and the
         plain mean of the hour (`*_mean`) for the 29 raw workbooks. For the
         5-min / 10-s files `*_first` is the raw record stamped HH:00:00; for
         the hourly file it is the single value. Raw workbooks are never modified.
Output : data/processed/india_hourly_demand_clean.csv
         reports/cleaning_log.md   (counts per rule, per year)

Cleaning rules (all thresholds are constants below; they were chosen from
physical plausibility - India's record demand is about 250 GW - not tuned to
any forecasting result):
  R1 zero_outage   demand == 0 (SCADA dropout; wind is 0 at the same time)
  R2 implausible   demand outside [DEMAND_MIN_MW, DEMAND_MAX_MW]
  R3 dip_spike     one-hour jump of more than JUMP_FRAC that returns to within
                   RETURN_FRAC of the previous level within MAX_RETURN_HOURS
  R3b isolated_point  a value that differs by more than ISOLATED_DEV_FRAC from the
                   mean of its two neighbours (catches glitches inside a ramp;
                   99.9% of normal hours deviate by < 5%)
  R4 imputed       NaN runs of at most MAX_IMPUTE_HOURS filled linearly;
                   longer runs stay missing (never filled)
  Stuck values (identical to the previous hour) are only flagged, not removed.
Solar: negative values clipped to 0 (sensor offset). The night-time baseline
drift is NOT corrected here (needs weather; Phase 3).

Usage (project root):  py scripts/build_hourly_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "interim" / "hourly_audit_view.csv.gz"
OUT = ROOT / "data" / "processed" / "india_hourly_demand_clean.csv"
LOG = ROOT / "reports" / "cleaning_log.md"

DEMAND_MIN_MW = 80_000
DEMAND_MAX_MW = 260_000
JUMP_FRAC = 0.15
RETURN_FRAC = 0.10
MAX_RETURN_HOURS = 3
MAX_IMPUTE_HOURS = 3
ISOLATED_DEV_FRAC = 0.10
EXPECTED_START, EXPECTED_END = "2021-09-01 00:00", "2025-06-24 23:00"


def md_table(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    rows = ["| year | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for y, r in t.iterrows():
        rows.append(f"| {y} | " + " | ".join(str(int(r[c])) for c in cols) + " |")
    return chr(10).join(rows)


def find_dips(d: pd.Series) -> pd.Series:
    """True for hours that belong to a temporary jump (down or up)."""
    v = d.to_numpy(dtype=float)
    flag = np.zeros(len(v), dtype=bool)
    for t in range(1, len(v) - 1):
        prev = v[t - 1]
        if np.isnan(prev) or np.isnan(v[t]):
            continue
        down = v[t] < prev * (1 - JUMP_FRAC)
        up = v[t] > prev * (1 + JUMP_FRAC)
        if not (down or up):
            continue
        for k in range(1, MAX_RETURN_HOURS + 1):
            if t + k >= len(v):
                break
            nxt = v[t + k]
            if not np.isnan(nxt) and abs(nxt - prev) <= prev * RETURN_FRAC:
                flag[t:t + k] = True
                break
    return pd.Series(flag, index=d.index)


def impute_short_gaps(s: pd.Series, max_len: int):
    """Linear fill of NaN runs of length <= max_len that have valid neighbours."""
    out = s.copy()
    filled = pd.Series(False, index=s.index)
    isna = s.isna().to_numpy()
    i, n = 0, len(s)
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            if (j - i) <= max_len and i > 0 and j < n:
                idx = s.index[i:j]
                out.loc[idx] = np.interp(np.arange(i, j), [i - 1, j],
                                         [s.iloc[i - 1], s.iloc[j]])
                filled.loc[idx] = True
            i = j
        else:
            i += 1
    return out, filled


def main():
    if not SRC.exists():
        sys.exit(f"Missing {SRC}. Run: py scripts/inspect_electricity_data.py")
    h = pd.read_csv(SRC, parse_dates=["hour"]).sort_values("hour").set_index("hour")
    full = pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")
    assert h.index.is_unique and h.index.equals(full), "hourly grid is not complete/unique"

    df = pd.DataFrame(index=h.index)
    df.index.name = "timestamp"
    df["demand_raw_mw"] = h["demand_mw_first"]
    df["source_regime"] = np.select(
        [h["source_file"].str.startswith("January 2024"), h.index < "2022-10-01"],
        ["hourly", "5min"], default="10s")

    d = df["demand_raw_mw"].copy()
    flag = pd.Series("ok", index=df.index, dtype=object)
    r1 = d == 0
    r2 = (d > 0) & ((d < DEMAND_MIN_MW) | (d > DEMAND_MAX_MW))
    d = d.mask(r1 | r2)
    flag[r1] = "zero_outage"
    flag[r2] = "implausible"
    r3 = find_dips(d) & d.notna()
    d = d.mask(r3)
    flag[r3] = "dip_spike"
    nb = (d.shift(1) + d.shift(-1)) / 2          # neighbours must both be valid
    r3b = ((d - nb).abs() / nb > ISOLATED_DEV_FRAC) & d.notna()
    d = d.mask(r3b)
    flag[r3b] = "isolated_point"
    n_missing_before = int(d.isna().sum())
    d, filled = impute_short_gaps(d, MAX_IMPUTE_HOURS)
    # flag keeps the REASON the raw value was rejected; imputation is a separate column
    df["electricity_demand_mw"] = d
    df["demand_quality_flag"] = flag
    df["demand_imputed"] = filled
    df["demand_stuck_flag"] = (df["demand_raw_mw"].diff() == 0) & (df["demand_raw_mw"] > 0)

    # Wind/solar/total: blank where the whole channel dropped out (demand 0 in the raw data)
    dropout = r1
    for out_name, src in (("wind_generation_mw", "wind_mw_first"),
                          ("solar_generation_mw", "solar_mw_first"),
                          ("total_generation_mw", "total_gen_mw_first")):
        x = h[src].copy()
        if out_name == "solar_generation_mw":
            df["solar_raw_mw"] = x
            x = x.clip(lower=0)
        df[out_name] = x.mask(dropout)

    cols = ["electricity_demand_mw", "demand_raw_mw", "demand_quality_flag", "demand_imputed", "demand_stuck_flag",
            "solar_generation_mw", "solar_raw_mw", "wind_generation_mw", "total_generation_mw",
            "source_regime"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, float_format="%.3f")

    # ---- assertions ----
    assert len(df) == 33432
    assert df.index.is_unique and df.index.is_monotonic_increasing
    valid = df["electricity_demand_mw"].dropna()
    assert valid.between(DEMAND_MIN_MW, DEMAND_MAX_MW).all()
    assert (df["solar_generation_mw"].dropna() >= 0).all()

    # ---- log ----
    yr = df.index.year
    tab = pd.crosstab(yr, df["demand_quality_flag"])
    tab_imp = df.groupby(yr)["demand_imputed"].sum().astype(int)
    still_missing = df["electricity_demand_mw"].isna()
    runs = (still_missing != still_missing.shift()).cumsum()
    gaps = still_missing.groupby(runs).agg(["first", "size"])
    longest = gaps[gaps["first"]].sort_values("size", ascending=False).head(5)
    gap_rows = []
    for rid in longest.index:
        idx = df.index[runs == rid]
        gap_rows.append(f"| {idx[0]} | {idx[-1]} | {len(idx)} |")
    lines = [
        "# Cleaning log (generated by `scripts/build_hourly_dataset.py`)", "",
        f"Rows: {len(df):,} hourly (2021-09-01 00:00 to 2025-06-24 23:00 IST).", "",
        "## Parameters",
        f"- demand bounds: {DEMAND_MIN_MW:,} to {DEMAND_MAX_MW:,} MW",
        f"- dip/spike rule: jump > {JUMP_FRAC:.0%} that returns to within {RETURN_FRAC:.0%} of the previous level within {MAX_RETURN_HOURS} h",
        f"- isolated-point rule: deviation from neighbours' mean > {ISOLATED_DEV_FRAC:.0%}",
        f"- imputation: linear for NaN runs of at most {MAX_IMPUTE_HOURS} h only", "",
        "## Hours per rule (all years)", "",
        "| reason raw value rejected | hours | of which imputed (<= 3 h gaps) | left missing |",
        "|---|---:|---:|---:|",
    ]
    for k, g in df.groupby("demand_quality_flag"):
        lines.append(f"| {k} | {len(g):,} | {int(g['demand_imputed'].sum())} | {int(g['electricity_demand_mw'].isna().sum())} |")
    lines += ["", "## Hours per rule per year", "", md_table(tab), "",
              "Imputed hours per year: " + ", ".join(f"{y}: {n}" for y, n in tab_imp.items()), "",
              f"Stuck-value hours (flagged only, kept): {int(df['demand_stuck_flag'].sum())}", "",
              f"Hours still missing after cleaning: {int(still_missing.sum())} "
              f"({still_missing.mean():.3%}); before imputation: {n_missing_before}", "",
              "## Longest remaining gaps", "", "| start | end | hours |", "|---|---|---:|"] + gap_rows
    lines += ["", f"Negative raw solar hours clipped to 0: {int((df['solar_raw_mw'] < 0).sum())}",
              f"Raw demand maxima before cleaning: {df['demand_raw_mw'].max():,.0f} MW; after: {valid.max():,.0f} MW"]
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}\nwrote {LOG}")
    print(df["demand_quality_flag"].value_counts().to_string())


if __name__ == "__main__":
    main()
