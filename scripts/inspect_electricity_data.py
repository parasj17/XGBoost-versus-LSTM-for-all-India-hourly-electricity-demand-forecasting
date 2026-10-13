"""Audit the raw Mendeley electricity workbooks (read-only on the raw files).

What it does
------------
1. Finds every .xlsx under data/raw/electricity_mendeley/.
2. For each workbook: lists sheet names/dimensions and the header rows.
3. Parses the main data sheet ("Sheet1" or "Report") and caches it as a
   gzip CSV in data/interim/parsed/ (so re-runs are fast). Raw files are
   never modified.
4. Computes per-file statistics: first/last timestamp, row count, sampling
   interval distribution, duplicate timestamps, missing/non-numeric cells,
   min/max of key columns.
5. Writes everything to data/interim/electricity_audit.json, which feeds
   reports/dataset_audit.md.

Usage (from the project root):
    py scripts/inspect_electricity_data.py            # full run (slow first time)
    py scripts/inspect_electricity_data.py --refresh  # ignore the parse cache
"""
import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "electricity_mendeley"
PARSED_DIR = ROOT / "data" / "interim" / "parsed"
OUT_JSON = ROOT / "data" / "interim" / "electricity_audit.json"

DATA_SHEETS = ("Sheet1", "Report")  # sheets that hold the time series


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")


def sheet_overview(wb):
    """Sheet name, dimensions and first 3 rows (long cells truncated)."""
    out = []
    for ws in wb.worksheets:
        head = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= 3:
                break
            head.append([str(c)[:60] if c is not None else None for c in row])
        out.append({"sheet": ws.title, "max_row": ws.max_row,
                    "max_col": ws.max_column, "head": head})
    return out


def parse_data_sheet(ws):
    """Return (DataFrame, tag_row, name_row). Handles both layouts.

    Layout A (2021-2023): row0 = SCADA tag ids, row1 = signal names, data from row2.
    Layout B (2024-25):   row0 = column names, data from row1.
    """
    rows = ws.iter_rows(values_only=True)
    r0 = next(rows)
    r1 = next(rows)
    if r0[0] == "Time" and r1[0] is None:  # layout A
        labels, tags = [], {}
        for i, (tag, name) in enumerate(zip(r0, r1)):
            if i == 0:
                labels.append("Time")
                continue
            label = name if name else f"col_{i}"
            labels.append(label)
            tags[label] = tag
        data = list(rows)
        layout = "A"
    else:  # layout B
        labels = list(r0)
        tags = {}
        data = [r1] + list(rows)
        layout = "B"
    df = pd.DataFrame(data, columns=labels)
    return df, tags, layout


def to_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # Layout B stores text such as '01-01-2024 00:00:00' (day-first).
    out = pd.to_datetime(s, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    if out.isna().all():  # cached CSVs of Layout A hold ISO strings
        out = pd.to_datetime(s, errors="coerce")
    return out


def summarize(df: pd.DataFrame, tcol: str) -> dict:
    ts = to_timestamp(df[tcol])
    n_bad_ts = int(ts.isna().sum())
    footer_labels = df.loc[ts.isna(), tcol].astype(str).tolist()[:10]
    keep = ts.notna()          # rows with an unparseable timestamp (e.g. the
    df = df[keep]              # Min/Max/Average/Sum footer rows) are excluded
    ts_ok = ts[keep]           # from all statistics below, but counted
    diffs = ts_ok.sort_values().diff().dropna()
    vc = diffs.dt.total_seconds().value_counts().head(6)
    num_cols = [c for c in df.columns if c != tcol]
    cols = {}
    for c in num_cols:
        raw = df[c]
        num = pd.to_numeric(raw, errors="coerce")
        non_numeric_text = raw[num.isna() & raw.notna()].astype(str)
        cols[c] = {
            "null": int(raw.isna().sum()),
            "non_numeric": int(len(non_numeric_text)),
            "non_numeric_examples": non_numeric_text.unique()[:3].tolist(),
            "min": None if num.dropna().empty else float(num.min()),
            "max": None if num.dropna().empty else float(num.max()),
            "mean": None if num.dropna().empty else float(num.mean()),
            "n_negative": int((num < 0).sum()),
            "n_zero": int((num == 0).sum()),
        }
    return {
        "rows": int(len(df)),
        "bad_or_missing_timestamps": n_bad_ts,
        "dropped_row_labels": footer_labels,
        "first_ts": str(ts_ok.min()) if len(ts_ok) else None,
        "last_ts": str(ts_ok.max()) if len(ts_ok) else None,
        "duplicate_timestamps": int(ts_ok.duplicated().sum()),
        "not_monotonic_increasing": bool(not ts_ok.is_monotonic_increasing),
        "interval_seconds_top": {str(k): int(v) for k, v in vc.items()},
        "min_interval_s": float(diffs.dt.total_seconds().min()) if len(diffs) else None,
        "max_interval_s": float(diffs.dt.total_seconds().max()) if len(diffs) else None,
        "columns": cols,
    }


KEY_COLS = {  # canonical name -> (layout A column, layout B column)
    "demand_mw": ("NLDC_DEMAND|P", "Demand (MW)"),
    "wind_mw": ("ALL_INDIA_WIND|P", "Wind (MW)"),
    "solar_mw": ("ALL_IND_SOLAR|P", "Solar (MW)"),
    "total_gen_mw": ("Total", "Total Generation (MW)"),
}


def hourly_view(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """Per clock-hour: value at HH:00 (instantaneous) and mean of the hour.

    For hourly files the single value is returned for both. n_samples is the
    number of raw records in that hour; n_demand_zero counts demand==0 rows.
    """
    ts = to_timestamp(df.iloc[:, 0])
    out = pd.DataFrame({"timestamp": ts})
    idx = 0 if layout == "A" else 1
    for name, cols in KEY_COLS.items():
        out[name] = pd.to_numeric(df[cols[idx]], errors="coerce") if cols[idx] in df else float("nan")
    out = out.dropna(subset=["timestamp"])
    out["hour"] = out["timestamp"].dt.floor("h")
    g = out.groupby("hour")
    res = g[list(KEY_COLS)].mean().add_suffix("_mean")
    first = g[list(KEY_COLS)].first().add_suffix("_first")
    res = res.join(first)
    res["n_samples"] = g.size()
    res["n_demand_zero"] = g["demand_mw"].apply(lambda x: int((x == 0).sum()))
    return res.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-parse even if cached")
    args = ap.parse_args()

    files = sorted(RAW_DIR.rglob("*.xlsx"))
    if not files:
        sys.exit(f"No .xlsx files found under {RAW_DIR}. Extract the Mendeley zip there first.")
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    audit = {}
    hourly_parts = []
    for f in files:
        print(f"[{f.name}] ...", flush=True)
        cache = PARSED_DIR / f"{safe_name(f)}.csv.gz"
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        entry = {"size_bytes": f.stat().st_size, "sheets": sheet_overview(wb)}
        sheet = next((ws for ws in wb.worksheets if ws.title in DATA_SHEETS), None)
        if sheet is None:
            entry["error"] = "no data sheet (Sheet1/Report) found"
            audit[f.name] = entry
            continue
        if cache.exists() and not args.refresh:
            df = pd.read_csv(cache, low_memory=False)
            side = cache.with_suffix(".meta.json")
            meta = json.loads(side.read_text()) if side.exists() else {}
            tags, layout = meta.get("tags", {}), meta.get("layout", "cached")
        else:
            # re-open: sheet_overview consumed iterators only on a fresh generator,
            # but parse from a new workbook handle to be safe.
            wb2 = openpyxl.load_workbook(f, read_only=True, data_only=True)
            df, tags, layout = parse_data_sheet(wb2[sheet.title])
            df.to_csv(cache, index=False, compression="gzip")
            cache.with_suffix(".meta.json").write_text(
                json.dumps({"tags": tags, "layout": layout}))
            wb2.close()
        entry["data_sheet"] = sheet.title
        entry["layout"] = layout
        entry["columns"] = list(df.columns)
        if tags:
            entry["scada_tags"] = tags
        entry["summary"] = summarize(df, df.columns[0])
        lay = "A" if "NLDC_DEMAND|P" in df.columns else "B"
        hourly_parts.append(hourly_view(df, lay).assign(source_file=f.name))
        audit[f.name] = entry
        wb.close()
        s = entry["summary"]
        print(f"    rows={s['rows']} {s['first_ts']} -> {s['last_ts']} "
              f"intervals={s['interval_seconds_top']}", flush=True)

    hourly = pd.concat(hourly_parts, ignore_index=True).sort_values("hour")
    hourly.to_csv(ROOT / "data" / "interim" / "hourly_audit_view.csv.gz", index=False,
                  compression="gzip")
    OUT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
