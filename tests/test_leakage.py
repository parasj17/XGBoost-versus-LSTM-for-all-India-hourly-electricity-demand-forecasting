"""Leakage test for the tabular feature pipeline (protocol section 4 / task 6.1).

Method: pick an origin time t0, destroy every MEASUREMENT after t0 (demand, weather,
solar, wind, total generation: scaled randomly and partly set to NaN), rebuild the
features, and require that all feature rows with origin <= t0 are unchanged for every
group except the labelled oracle group g3b. A deliberately leaky 'canary' feature
(demand one hour in the future) must be caught by the same machinery, which proves the
test is able to fail. Only data up to the end of the validation period is loaded.

Usage: py tests/test_leakage.py        (exit code 0 = pass)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import features_tabular as ft  # noqa: E402

VAL_END = pd.Timestamp("2024-06-30 23:00")
MEASURED = ["electricity_demand_mw", "demand_raw_mw", "solar_generation_mw", "solar_raw_mw", "wind_generation_mw",
            "total_generation_mw", "temp_c", "rel_humidity_pct", "dew_point_c", "precip_mm", "wind_speed_10m_kmh",
            "cloud_cover_pct", "ghi_wm2"] + ft.REGIONAL_T


def perturb(df: pd.DataFrame, t0: pd.Timestamp, rng) -> pd.DataFrame:
    d = df.copy()
    fut = d.index > t0
    for c in MEASURED:
        v = d.loc[fut, c].to_numpy(dtype=float)
        v = v * rng.uniform(0.5, 1.5, size=v.shape) + rng.normal(0, 5, size=v.shape)
        v[rng.random(v.shape) < 0.05] = np.nan
        d.loc[fut, c] = v
    return d


def changed_rows(df, df2, t0, h, groups, relative) -> int:
    a = ft.build_features(df, h, groups, relative).loc[:t0]
    b = ft.build_features(df2, h, groups, relative).loc[:t0]
    same = (a.eq(b) | (a.isna() & b.isna())).all(axis=1)
    return int((~same).sum())


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", parse_dates=["timestamp"], index_col="timestamp")
    df = df.loc[:VAL_END]
    rng = np.random.default_rng(0)
    failures = []
    safe_groups = ["g0", "g1", "g2", "g3a", "g4"]
    for t0 in (pd.Timestamp("2022-06-15 10:00"), pd.Timestamp("2023-11-03 02:00"), pd.Timestamp("2024-03-20 18:00")):
        df2 = perturb(df, t0, rng)
        for h in (1, 6, 24, 48):
            for relative in (True, False):
                n = changed_rows(df, df2, t0, h, safe_groups, relative)
                status = "ok" if n == 0 else "LEAK"
                print(f"t0={t0}  h={h:2d}  relative={str(relative):5s}  rows changed by future data: {n}  [{status}]")
                if n:
                    failures.append((t0, h, relative, n))
        # the oracle group must react to the future (sanity: it is correctly labelled as using it)
        n_or = changed_rows(df, df2, t0, 24, ["g3b"], True)
        print(f"t0={t0}  oracle group g3b changes (expected > 0): {n_or}")
        if n_or == 0:
            failures.append((t0, "oracle group did not change", None, 0))
    # canary: a deliberately leaky feature must be detected
    leaky = lambda d: d["electricity_demand_mw"].shift(-1)
    t0 = pd.Timestamp("2023-11-03 02:00")
    df2 = perturb(df, t0, rng)
    a, b = leaky(df).loc[:t0], leaky(df2).loc[:t0]
    caught = int((~(a.eq(b) | (a.isna() & b.isna()))).sum())
    print(f"canary leaky feature detected in {caught} rows (expected > 0)")
    if caught == 0:
        failures.append(("canary not detected",))
    # target definition: the target at origin t is the demand at t+h
    for h in (1, 24):
        tgt = df["electricity_demand_mw"].shift(-h)
        i = df.index[1000]
        assert tgt.loc[i] == df["electricity_demand_mw"].loc[i + pd.Timedelta(hours=h)] or np.isnan(tgt.loc[i])
    if failures:
        print("\nFAILED:", failures)
        sys.exit(1)
    print("\nALL LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
