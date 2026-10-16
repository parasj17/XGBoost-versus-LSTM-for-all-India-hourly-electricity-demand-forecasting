"""Task 9.4 (extension D): population-weighted state-level weather as an alternative to the 12-city equal-weight national series.

Downloads hourly ERA5 (Open-Meteo, same variables, IST) for the capital (or main city) of every state/UT, aggregates it with Census-2011 population weights
(national series, and the five regional temperature columns from the states of each grid region), and writes a copy of the canonical dataset in which these
columns replace the 12-city series: data/interim/sensitivity/canonical_statepop.csv. Points, not grids: a state capital is a proxy for its state.
Raw files: data/raw/weather_states/era5_<code>.csv (git-ignored). Resumes from existing files.   Usage: py scripts/build_state_weather_dataset.py
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
crw = importlib.import_module("compute_region_weights")
brh = importlib.import_module("build_regional_holiday_features")
RAW = ROOT / "data" / "raw" / "weather_states"
OUT = ROOT / "data" / "interim" / "sensitivity" / "canonical_statepop.csv"
CAP = {"AN": (11.62, 92.73), "AP": (16.51, 80.52), "AR": (27.08, 93.61), "AS": (26.14, 91.79), "BR": (25.59, 85.14), "CG": (21.25, 81.63), "CH": (30.73, 76.78),
       "DH": (20.40, 72.83), "DL": (28.61, 77.21), "GA": (15.50, 73.83), "GJ": (23.22, 72.65), "HP": (31.10, 77.17), "HR": (30.73, 76.78), "JH": (23.34, 85.31),
       "JK": (34.08, 74.80), "KA": (12.97, 77.59), "KL": (8.52, 76.94), "LA": (34.16, 77.58), "LD": (10.57, 72.64), "MH": (19.08, 72.88), "ML": (25.57, 91.88),
       "MN": (24.82, 93.94), "MP": (23.26, 77.41), "MZ": (23.73, 92.72), "NL": (25.67, 94.11), "OD": (20.30, 85.82), "PB": (30.90, 75.85), "PY": (11.93, 79.83),
       "RJ": (26.91, 75.79), "SK": (27.33, 88.61), "TN": (13.08, 80.27), "TR": (23.83, 91.28), "TS": (17.39, 78.48), "UK": (30.32, 78.03), "UP": (26.85, 80.95),
       "WB": (22.57, 88.36)}
VARS = iw.HOURLY
COLMAP = {"temperature_2m": "temp_c", "relative_humidity_2m": "rel_humidity_pct", "dew_point_2m": "dew_point_c", "precipitation": "precip_mm",
          "wind_speed_10m": "wind_speed_10m_kmh", "cloud_cover": "cloud_cover_pct", "shortwave_radiation": "ghi_wm2"}
REGION_COL = {"Northern": "temp_c_n", "Western": "temp_c_w", "Southern": "temp_c_s", "Eastern": "temp_c_e", "North-Eastern": "temp_c_ne"}


def fetch(lat, lon):
    for attempt in range(6):
        try:
            df, _ = iw.fetch_city(lat, lon, "2021-09-01", "2025-06-24")
            return df
        except Exception:
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("download failed")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    pop = pd.read_csv(ROOT / "data" / "raw" / "population" / "census2011_state_population.csv").set_index("state_or_ut")["population_2011"]
    frames, done = {}, {}
    for code, (lat, lon) in CAP.items():
        f = RAW / f"era5_{code}.csv"
        if not f.exists():
            key = (lat, lon)
            if key in done:
                done[key].to_csv(f, index=False)
            else:
                df = fetch(lat, lon)
                df.to_csv(f, index=False)
                done[key] = df
                time.sleep(1.0)
            print(f"{code}: downloaded", flush=True)
        frames[code] = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
    idx = frames["DL"].index
    w = pd.Series({c: float(pop[brh.CODE2NAME[c]]) for c in CAP})
    w = w / w.sum()
    nat = sum(frames[c][VARS] * w[c] for c in CAP)
    region_t = {}
    name2code = {v: k for k, v in brh.CODE2NAME.items()}
    for r, states in crw.REGIONS.items():
        codes = [name2code[s] for s in states if s in name2code]
        wr = w[codes] / w[codes].sum()
        region_t[REGION_COL[r]] = sum(frames[c]["temperature_2m"] * wr[c] for c in codes)
    can = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", parse_dates=["timestamp"]).set_index("timestamp")
    assert can.index.equals(idx), "weather grid differs from the canonical dataset"
    for v, name in COLMAP.items():
        can[name] = nat[v].to_numpy()
    for name, s in region_t.items():
        can[name] = s.to_numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    can.to_csv(OUT, float_format="%.4f")
    base = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", parse_dates=["timestamp"]).set_index("timestamp")
    print("correlation of the new national series with the 12-city series:", {n: round(float(can[n].corr(base[n])), 4) for n in ("temp_c", "rel_humidity_pct", "ghi_wm2")})
    print("mean temperature 12-city vs state-capital population-weighted:", round(float(base["temp_c"].mean()), 2), round(float(can["temp_c"].mean()), 2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
