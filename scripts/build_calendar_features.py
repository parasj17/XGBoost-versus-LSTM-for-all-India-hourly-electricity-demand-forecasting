"""Build calendar/holiday features from timestamps and the sourced holiday CSV.

Inputs
------
data/raw/calendar/dopt_central_gazetted_holidays.csv
    Hand-transcribed from DoPT (Govt. of India) Office Memoranda; every row
    carries its source document and URL. See references.md.

Outputs
-------
data/processed/calendar_daily.csv   one row per day: date, holiday_name,
                                    holiday_type, is_holiday, festival_name,
                                    is_major_festival (+ is_weekend)
data/processed/calendar_hourly.csv  one row per hour: time features (hour, day,
                                    day_of_week, month, is_weekend, season,
                                    sin/cos cyclical encodings) joined with the
                                    daily holiday flags

Important scope notes
---------------------
* The DoPT list is the *central-government (Delhi) gazetted* list. It is a
  reasonable proxy for pan-India holidays, NOT a list of state/regional
  holidays. Regional holidays are NOT included (needs per-state sources).
* is_major_festival is an analyst-defined flag (MAJOR_FESTIVALS below), not an
  official classification. Change it deliberately and record the change.
* Dates for moon-sighting festivals (Id, Muharram, Milad-un-Nabi) are the
  *planned* dates in the memoranda and may differ by a day from observance.
* No timezone conversion is done: timestamps are treated as naive India
  local time (IST, UTC+05:30, no DST).

Usage:
    py scripts/build_calendar_features.py --start 2021-09-01 --end 2025-06-24
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLIDAY_CSV = ROOT / "data" / "raw" / "calendar" / "dopt_central_gazetted_holidays.csv"
OUT_DIR = ROOT / "data" / "processed"

# Analyst-defined (assumption): festivals expected to shift demand most.
MAJOR_FESTIVALS = {"Holi", "Diwali (Deepavali)", "Dussehra", "Id-ul-Fitr", "Christmas Day"}


def season_imd(month: int) -> str:
    """IMD-style Indian seasons."""
    if month in (1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre_monsoon_summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"  # Oct-Dec


def build_daily(start: str, end: str) -> pd.DataFrame:
    hol = pd.read_csv(HOLIDAY_CSV, parse_dates=["date"])
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    g = hol.groupby("date").agg(
        holiday_name=("holiday_name", "; ".join),
        holiday_type=("holiday_type", lambda s: "; ".join(sorted(set(s)))),
    ).reset_index()
    out = days.merge(g, on="date", how="left")
    out["is_holiday"] = out["holiday_name"].notna().astype(int)
    # festival_name: holidays that are festivals (not fixed national days)
    fest = hol[hol["holiday_type"] == "central_gazetted_festival"]
    fest = fest.groupby("date")["holiday_name"].agg("; ".join).rename("festival_name")
    out = out.merge(fest, on="date", how="left")
    major = hol[hol["holiday_name"].isin(MAJOR_FESTIVALS)]["date"]
    out["is_major_festival"] = out["date"].isin(major).astype(int)
    out["is_weekend"] = (out["date"].dt.dayofweek >= 5).astype(int)
    for c in ("holiday_name", "holiday_type", "festival_name"):
        out[c] = out[c].fillna("")
    return out


def build_hourly(start: str, end: str, daily: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    h = pd.DataFrame({"timestamp": idx})
    h["date"] = h["timestamp"].dt.normalize()
    h["hour"] = h["timestamp"].dt.hour
    h["day"] = h["timestamp"].dt.day
    h["day_of_week"] = h["timestamp"].dt.dayofweek  # Monday=0
    h["day_of_year"] = h["timestamp"].dt.dayofyear
    h["month"] = h["timestamp"].dt.month
    h["is_weekend"] = (h["day_of_week"] >= 5).astype(int)
    h["season"] = h["month"].map(season_imd)
    # Cyclical encodings so 23:00 is close to 00:00, December to January, etc.
    for name, val, period in (("hour", h["hour"], 24), ("dow", h["day_of_week"], 7),
                              ("month", h["month"] - 1, 12),
                              ("doy", h["day_of_year"] - 1, 365.25)):
        h[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        h[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    keep = ["date", "is_holiday", "holiday_name", "festival_name", "is_major_festival"]
    h = h.merge(daily[keep], on="date", how="left").drop(columns="date")
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-09-01")
    ap.add_argument("--end", default="2025-06-24")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = build_daily(a.start, a.end)
    hourly = build_hourly(a.start, a.end, daily)
    daily.to_csv(OUT_DIR / "calendar_daily.csv", index=False)
    hourly.to_csv(OUT_DIR / "calendar_hourly.csv", index=False)
    print(f"daily rows={len(daily)}, holidays in range={int(daily.is_holiday.sum())}")
    print(f"hourly rows={len(hourly)}")


if __name__ == "__main__":
    main()
