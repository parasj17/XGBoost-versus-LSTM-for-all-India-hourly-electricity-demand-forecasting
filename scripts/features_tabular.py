"""Leakage-safe tabular features for the forecasting models (protocol section 4).

Every function takes the canonical hourly table `df` (index = IST hour) and a horizon
`h`, and returns features indexed by the ORIGIN time t. A feature at row t may use
  * measurements (demand, weather, solar, wind) at times <= t only, or
  * deterministic information about the TARGET time t+h (calendar, holidays,
    day/night), which is known in advance,
  * and, only in the labelled oracle scenario B, weather measured at t+h.
tests/test_leakage.py enforces this by perturbing every measurement after a chosen
origin and checking that no earlier feature row changes.

Feature groups: g0 history, g1 calendar, g2 holidays/festival windows, g3a past weather,
g3b oracle weather at t+h (scenario B), g4 past solar/wind.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLIDAYS = ROOT / "data" / "raw" / "calendar" / "dopt_central_gazetted_holidays.csv"
LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]
WEATHER = ["temp_c", "rel_humidity_pct", "dew_point_c", "precip_mm", "wind_speed_10m_kmh", "cloud_cover_pct", "ghi_wm2"]
REGIONAL_T = ["temp_c_n", "temp_c_w", "temp_c_s", "temp_c_e", "temp_c_ne"]
FEATURE_SETS = {                      # cumulative ablation sets
    "S0_history": ["g0"],
    "S1_calendar": ["g0", "g1"],
    "S2_holidays": ["g0", "g1", "g2"],
    "S3_weatherA": ["g0", "g1", "g2", "g3a"],
    "S4_full_A": ["g0", "g1", "g2", "g3a", "g4"],        # headline (scenario A)
    "S5_oracle_B": ["g0", "g1", "g2", "g3a", "g4", "g3b"],  # upper bound, oracle weather at t+h
}


def _major_set():
    spec = importlib.util.spec_from_file_location("bcf", ROOT / "scripts" / "build_calendar_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MAJOR_FESTIVALS


def festival_windows(target_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Signed distances (days) from each target day to the nearest major festival / any holiday."""
    hol = pd.read_csv(HOLIDAYS, parse_dates=["date"])
    major = hol[hol["holiday_name"].isin(_major_set())]["date"].drop_duplicates().sort_values().to_numpy()
    anyh = hol["date"].drop_duplicates().sort_values().to_numpy()
    days = target_days.normalize().to_numpy()
    out = {}
    for name, dates in (("major", major), ("hol", anyh)):
        i = np.searchsorted(dates, days, side="left")           # next date >= day
        nxt = np.where(i < len(dates), (dates[np.minimum(i, len(dates) - 1)] - days) / np.timedelta64(1, "D"), np.nan)
        j = np.searchsorted(dates, days, side="right") - 1      # last date <= day
        prv = np.where(j >= 0, (days - dates[np.maximum(j, 0)]) / np.timedelta64(1, "D"), np.nan)
        out[f"d_next_{name}"] = np.clip(nxt, 0, 30)
        out[f"d_since_{name}"] = np.clip(prv, 0, 30)
    return pd.DataFrame(out)


def anchors(y: pd.Series, h: int) -> dict:
    """Seasonal anchors: latest same-hour value (24 h and 168 h grid) with time <= origin."""
    a = {"a24": y.shift(24 * math.ceil(h / 24) - h)}
    if h <= 168:
        a["a168"] = y.shift(168 - h)
    return a


def build_features(df: pd.DataFrame, h: int, groups, relative: bool) -> pd.DataFrame:
    y = df["electricity_demand_mw"]
    X = pd.DataFrame(index=df.index)
    T = df.index + pd.Timedelta(hours=h)
    a = anchors(y, h)
    scale = a["a24"] if relative else 1.0
    if "g0" in groups:
        for k in LAGS:
            X[f"lag{k}"] = y.shift(k) / scale
        for w in (24, 168):
            r = y.rolling(w, min_periods=w)
            X[f"roll{w}_mean"] = r.mean() / scale
            X[f"roll{w}_std"] = r.std() / scale
        X["anchor24"] = a["a24"] / scale if relative else a["a24"]
        if "a168" in a:
            X["anchor168"] = a["a168"] / scale
        if relative:
            X = X.drop(columns=["anchor24"])            # equals 1 by construction
    if "g1" in groups:
        X["hour_T"], X["dow_T"] = T.hour, T.dayofweek
        X["month_T"], X["doy_T"] = T.month, T.dayofyear
        X["weekend_T"] = (T.dayofweek >= 5).astype(int)
        X["hour_origin"] = df.index.hour
    if "g2" in groups:
        hol = pd.read_csv(HOLIDAYS, parse_dates=["date"])
        major_days = set(hol[hol["holiday_name"].isin(_major_set())]["date"])
        any_days = set(hol["date"])
        td = pd.DatetimeIndex(T).normalize()
        X["is_holiday_T"] = td.isin(any_days).astype(int)
        X["is_major_T"] = td.isin(major_days).astype(int)
        fw = festival_windows(pd.DatetimeIndex(T))
        for c in fw.columns:
            X[c] = fw[c].to_numpy()
    if "g3a" in groups:
        for c in WEATHER:
            X[f"{c}_t"] = df[c]
        for lag in (1, 3, 6, 24):
            X[f"temp_lag{lag}"] = df["temp_c"].shift(lag)
        X["temp_roll24_mean"] = df["temp_c"].rolling(24, min_periods=24).mean()
        X["temp_roll24_max"] = df["temp_c"].rolling(24, min_periods=24).max()
        X["precip_roll24_sum"] = df["precip_mm"].rolling(24, min_periods=24).sum()
        for c in REGIONAL_T:
            X[f"{c}_t"] = df[c]
    if "g3b" in groups:                                   # ORACLE: weather measured at t+h
        for c in WEATHER + REGIONAL_T:
            X[f"{c}_oracleT"] = df[c].shift(-h)
    if "g4" in groups:
        for c, n in (("solar_generation_mw", "solar"), ("wind_generation_mw", "wind")):
            X[f"{n}_t"] = df[c]
            X[f"{n}_lag24"] = df[c].shift(24)
            X[f"{n}_roll24_mean"] = df[c].rolling(24, min_periods=24).mean()
    return X
