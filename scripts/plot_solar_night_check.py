"""Figure: the growing night-time solar offset and its correction (decision in Phase 3).

Reads  data/processed/india_hourly_canonical.csv
Writes reports/figures/solar_night_baseline.png
Panel A: mean raw (negatives clipped) solar in national-night hours, by year - what the correction removes.
Panel B: mean daily solar profile by year after correction - night is 0, the daytime shape is intact.
Usage: py scripts/plot_solar_night_check.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "india_hourly_canonical.csv"
OUT = ROOT / "reports" / "figures" / "solar_night_baseline.png"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    d = pd.read_csv(SRC, parse_dates=["timestamp"]).set_index("timestamp")
    raw = d["solar_raw_mw"].clip(lower=0)
    night = d["is_national_night"].astype(bool)
    by_year = raw[night].groupby(raw[night].index.year).mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    style(a)
    a.bar([str(y) for y in by_year.index], by_year.values, color=BLUE, width=0.55)
    for i, v in enumerate(by_year.values):
        a.text(i, v + 15, f"{v:,.0f}", ha="center", fontsize=8, color=INK2)
    a.set_ylabel("Mean solar in national-night hours (MW)", color=INK2, fontsize=9)
    a.set_title("A  Night-time solar offset in the raw data (should be 0)", loc="left", fontsize=10, color=INK)
    a.text(0.0, -0.16, "2021 = Sep-Dec only; 2025 = Jan to 24 Jun", transform=a.transAxes, fontsize=8, color=INK2)
    style(b)
    for yr, c, ls in ((2021, BLUE, "-"), (2023, ORANGE, "--"), (2025, AQUA, "-.")):
        s = d.loc[d.index.year == yr, "solar_generation_mw"]
        prof = s.groupby(s.index.hour).mean() / 1000
        b.plot(prof.index, prof.values, color=c, linestyle=ls, linewidth=1.8, label=str(yr))
    b.set_xlabel("Hour of day (IST)", color=INK2, fontsize=8)
    b.set_ylabel("Mean solar after correction (GW)", color=INK2, fontsize=9)
    b.set_title("B  Daily solar profile after correction", loc="left", fontsize=10, color=INK)
    b.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
