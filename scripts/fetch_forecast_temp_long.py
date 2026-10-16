"""Archived GFS temperature forecasts at longer leads (3 days and 7 days) for the long-horizon scenario-C extension (protocol deviation log, 2026-10-16).

Open-Meteo Previous Runs API (model gfs_seamless), temperature_2m_previous_day3 and _day7 for the same 12 cities; aggregated like the other national series
(mean of cities within a grid region, then equal-weight mean of the five regions). Output: data/raw/weather_forecast/gfs_d3d7_<city>.csv (raw) and
data/interim/forecast_temp_long_national.csv with columns f3_temperature_2m and f7_temperature_2m (IST, 33,432 hourly rows, gaps stay NaN). Resumable.
Usage: py scripts/fetch_forecast_temp_long.py
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
OUT = ROOT / "data" / "interim" / "forecast_temp_long_national.csv"
START, END = "2021-09-01", "2025-06-24"
COLS = ["temperature_2m_previous_day3", "temperature_2m_previous_day7"]


def fetch_chunk(lat, lon, a, b):
    params = {"latitude": lat, "longitude": lon, "start_date": a, "end_date": b, "hourly": ",".join(COLS), "timezone": "Asia/Kolkata", "models": "gfs_seamless"}
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
        f = RAW / f"gfs_d3d7_{city}.csv"
        if f.exists():
            frames[(region, city)] = pd.read_csv(f, parse_dates=["timestamp"])
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
        print(f"{city}: {len(df)} rows", flush=True)
    long = pd.concat([d.assign(grid_region=reg) for (reg, _), d in frames.items()], ignore_index=True)
    reg = long.groupby(["timestamp", "grid_region"])[COLS].mean()
    nat = reg.groupby("timestamp").mean()
    nat.columns = ["f3_temperature_2m", "f7_temperature_2m"]
    nat = nat.reindex(pd.date_range(START + " 00:00", END + " 23:00", freq="h"))
    nat.index.name = "timestamp"
    nat.to_csv(OUT, float_format="%.4f")
    print("hours:", len(nat), "| missing share:", nat.isna().mean().round(4).to_dict())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
