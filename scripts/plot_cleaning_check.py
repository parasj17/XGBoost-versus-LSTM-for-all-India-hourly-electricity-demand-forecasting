"""Before/after figure for the demand cleaning (task 2.5).

Reads  data/processed/india_hourly_demand_clean.csv
Writes reports/figures/cleaning_before_after.png

Panel A: whole cleaned series with the rejected hours marked along the bottom.
Panels B-D: three zooms comparing the raw value (orange) with the cleaned value (blue):
  B  8-hour telemetry outage (raw = 0)           2021-10-08/09
  C  glitch inside the morning ramp              2022-02-01
  D  single-hour dip                             2025-04-04
Gaps in the blue line are hours that stay missing; short gaps that were
interpolated are drawn as blue points joined by the line.

Colours: validated categorical palette slots 1 (blue, cleaned) and 2 (orange, raw).
Usage: py scripts/plot_cleaning_check.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "india_hourly_demand_clean.csv"
OUT = ROOT / "reports" / "figures" / "cleaning_before_after.png"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

ZOOMS = [
    ("B  8-hour outage: raw reads 0", "2021-10-07 12:00", "2021-10-10 06:00"),
    ("C  glitch inside the morning ramp", "2022-01-31 18:00", "2022-02-02 12:00"),
    ("D  single-hour dip", "2025-04-03 12:00", "2025-04-05 12:00"),
]


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
    d["clean_gw"] = d["electricity_demand_mw"] / 1000
    d["raw_gw"] = d["demand_raw_mw"] / 1000
    rejected = d["demand_quality_flag"] != "ok"

    fig = plt.figure(figsize=(12, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.plot(d.index, d["clean_gw"], color=BLUE, linewidth=0.5)
    ymin = 80
    ax.scatter(d.index[rejected], [ymin + 4] * int(rejected.sum()), marker="^", s=26,
               color=ORANGE, linewidth=0, zorder=3)
    ax.set_ylim(ymin, 260)
    ax.set_ylabel("Demand (GW)", color=INK2, fontsize=9)
    ax.set_title(f"A  Cleaned all-India hourly demand; orange triangles mark the "
                 f"{int(rejected.sum())} hours whose raw value was rejected (0.18% of hours)",
                 loc="left", fontsize=10, color=INK)

    for i, (title, a, b) in enumerate(ZOOMS):
        ax = fig.add_subplot(gs[1, i])
        style(ax)
        w = d.loc[a:b]
        ax.plot(w.index, w["raw_gw"], color=ORANGE, linewidth=1.4, marker="o", markersize=4,
                label="raw")
        ax.plot(w.index, w["clean_gw"], color=BLUE, linewidth=1.4, marker="o", markersize=4,
                label="cleaned")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        if i == 0:
            ax.set_ylabel("Demand (GW)", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("Cleaning check: rejected hours are isolated glitches; the real daily shape is kept",
                 x=0.01, ha="left", fontsize=12, color=INK)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
