"""Probe / fetch Open-Meteo historical (ERA5) weather for a set of Indian cities.

Design (see reports/data_acquisition_plan.md)
---------------------------------------------
The electricity target is ALL-INDIA, so no single city is representative.
We sample 12 cities spread over India's five grid regions and later aggregate
them (region mean, then weighted national mean). Coordinates below are
approximate city centres; ERA5 is a 0.25 deg (~25 km) grid so they only select
the nearest grid cell. The aggregation weights are NOT decided here: they must
come from a documented source (e.g. Census 2011 population or regional energy
consumption) - see WEIGHTS in the plan. Nothing is weighted by this script.

Modes
-----
  --pilot   Fetch 3 days for every city (36 small requests) and print a
            schema check. Nothing is saved to data/raw. SAFE default.
  --fetch   Fetch the full period and save one raw CSV per city in
            data/raw/weather/. Run only after you approve the city list.

Time zone: requests use timezone=Asia/Kolkata so timestamps are IST,
matching the electricity data (to be confirmed in the audit).

Leakage reminder: these are reanalysis OBSERVATIONS, not forecasts. For
horizons > 0 a deployed model would only have weather FORECASTS. Documented as
an experimental assumption in the methodology.

Usage:
    py scripts/inspect_weather_data.py --pilot
    py scripts/inspect_weather_data.py --fetch --start 2021-09-01 --end 2025-06-24
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "weather"
API = "https://archive-api.open-meteo.com/v1/archive"

# grid_region, city, lat, lon (approximate city centres)
CITIES = [
    ("Northern", "Delhi", 28.61, 77.21), ("Northern", "Lucknow", 26.85, 80.95),
    ("Northern", "Jaipur", 26.91, 75.79),
    ("Western", "Mumbai", 19.08, 72.88), ("Western", "Ahmedabad", 23.02, 72.57),
    ("Western", "Nagpur", 21.15, 79.09),
    ("Southern", "Chennai", 13.08, 80.27), ("Southern", "Bengaluru", 12.97, 77.59),
    ("Southern", "Hyderabad", 17.39, 78.48),
    ("Eastern", "Kolkata", 22.57, 88.36), ("Eastern", "Patna", 25.59, 85.14),
    ("North-Eastern", "Guwahati", 26.14, 91.74),
]
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "wind_speed_10m", "cloud_cover", "shortwave_radiation"]


def fetch_city(lat, lon, start, end):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(HOURLY), "timezone": "Asia/Kolkata", "models": "era5"}
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j["hourly"]).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--fetch", action="store_true")
    ap.add_argument("--start", default="2021-09-01")
    ap.add_argument("--end", default="2025-06-24")
    a = ap.parse_args()

    if a.fetch:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        start, end = a.start, a.end
    else:  # pilot is the default
        start, end = "2023-06-01", "2023-06-03"

    for region, city, lat, lon in CITIES:
        df, j = fetch_city(lat, lon, start, end)
        nulls = int(df[HOURLY].isna().sum().sum())
        print(f"{region:13s} {city:10s} grid=({j['latitude']:.2f},{j['longitude']:.2f}) "
              f"rows={len(df)} nulls={nulls} tz={j['timezone']} units={j['hourly_units']['temperature_2m']}")
        if a.fetch:
            df.insert(1, "city", city)
            df.insert(2, "grid_region", region)
            df.to_csv(OUT_DIR / f"era5_{city.lower()}_{start}_{end}.csv", index=False)
        time.sleep(1)  # be polite to the free API


if __name__ == "__main__":
    main()
