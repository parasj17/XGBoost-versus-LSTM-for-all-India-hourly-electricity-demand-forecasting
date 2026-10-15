"""Download archived NWP forecasts (scenario C) and aggregate them to a national series.

Source: Open-Meteo Previous Runs API (GFS, model gfs_seamless): the value for time T taken from the forecast run issued 1 day (previous_day1)
or 2 days (previous_day2) earlier. Same 12 cities, variables, IST time stamps and equal-weight region aggregation as the ERA5 weather.
Raw CSV per city: data/raw/weather_forecast/gfs_<city>.csv.  National aggregate: data/interim/forecast_weather_national.csv with columns
f1_<var> (day-1 forecast) and f2_<var> (day-2 forecast). Missing values stay NaN (the archive has gaps, notably 2023-12 to 2024-01).
Usage: py scripts/fetch_forecast_weather.py          (about 200 requests; resumes from existing city files)
"""
import importlib
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
iw = importlib.import_module("inspect_weather_data")
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
RAW = ROOT / "data" / "raw" / "weather_forecast"
OUT = ROOT / "data" / "interim" / "forecast_weather_national.csv"
START, END = "2021-09-01", "2025-06-24"
VARS = iw.HOURLY
LEADS = {"f1": "previous_day1", "f2": "previous_day2"}


def fetch_chunk(lat, lon, a, b):
    hourly = ",".join(f"{v}_{s}" for v in VARS for s in LEADS.values())
    params = {"latitude": lat, "longitude": lon, "start_date": a, "end_date": b, "hourly": hourly, "timezone": "Asia/Kolkata", "models": "gfs_seamless"}
    for attempt in range(6):
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 200:
            df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        time.sleep(20 * (attempt + 1))
    r.raise_for_status()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    quarters = pd.period_range(START, END, freq="Q")
    frames = {}
    for region, city, lat, lon in iw.CITIES:
        f = RAW / f"gfs_{city}.csv"
        if f.exists():
            frames[(region, city)] = pd.read_csv(f, parse_dates=["timestamp"])
            print(f"{city}: cached", flush=True)
            continue
        parts = []
        for q in quarters:
            a = max(q.start_time.normalize(), pd.Timestamp(START)).strftime("%Y-%m-%d")
            b = min(q.end_time.normalize(), pd.Timestamp(END)).strftime("%Y-%m-%d")
            parts.append(fetch_chunk(lat, lon, a, b))
            time.sleep(1.0)
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(f, index=False)
        frames[(region, city)] = df
        miss = {k: int(df[k].isna().sum()) for k in df.columns if k != "timestamp" and df[k].isna().any()}
        print(f"{city}: {len(df)} rows; columns with missing values: {len(miss)}; e.g. {dict(list(miss.items())[:2])}", flush=True)
    # national: mean of cities within a region, then equal-weight mean of the five regions (decision D-03)
    cols = [f"{v}_{s}" for v in VARS for s in LEADS.values()]
    long = pd.concat([d.assign(grid_region=reg) for (reg, _), d in frames.items()], ignore_index=True)
    reg = long.groupby(["timestamp", "grid_region"])[cols].mean()          # NaN-skipping mean over cities
    nat = reg.groupby("timestamp").mean()
    nat.columns = [c.replace("_previous_day1", "").replace("_previous_day2", "") for c in nat.columns]
    nat = pd.DataFrame({f"{p}_{v}": reg.groupby("timestamp").mean()[f"{v}_{s}"] for v in VARS for p, s in LEADS.items()})
    full = pd.date_range(START + " 00:00", END + " 23:00", freq="h")
    nat = nat.reindex(full)
    nat.index.name = "timestamp"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nat.to_csv(OUT, float_format="%.4f")
    print("hours:", len(nat), "| missing share per column (max):", round(float(nat.isna().mean().max()), 4))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
