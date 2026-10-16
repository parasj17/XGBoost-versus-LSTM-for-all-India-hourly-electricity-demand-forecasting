"""Figures for the write-up, generated from the saved test-period predictions (data/interim/predictions) and the result tables.
Colour-blind-safe palette (Okabe-Ito). Output: reports/figures/fig*.png.
Usage: py -I scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data" / "interim" / "predictions", ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
HZ = [1, 6, 24, 48, 72, 168]
COL = {"ridge": "#0072B2", "xgb": "#D55E00", "lstm": "#009E73", "naive": "#7F7F7F"}
LAB = {"ridge": "Ridge", "xgb": "XGBoost", "lstm": "LSTM", "naive": "Best naive"}
KEYS = {"short": {"ridge": "te__ridge_best", "xgb": "te__xgb_S4_full_A", "lstm": "te__lstm_L4_full_A",
                  "naive": ["te__persistence", "te__snaive24", "te__snaive168"]},
        "long": {"ridge": "ridge", "xgb": "XGBoost_S4", "lstm": "LSTM_L4",
                 "naive": ["persistence", "same_hour_3_or_7_days_earlier", "snaive168"]}}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})
rng = np.random.default_rng(0)


def load(h):
    z = np.load(P / (f"h{h}.npz" if h <= 48 else f"long_h{h}.npz"), allow_pickle=True)
    k = KEYS["short" if h <= 48 else "long"]
    y = z["y_te"].astype(float)
    pr = {m: z[k[m]].astype(float) for m in ("ridge", "xgb", "lstm")}
    nv = {n: z[n].astype(float) for n in k["naive"]}
    best = min(nv, key=lambda n: np.mean(np.abs(nv[n] - y)))
    pr["naive"] = nv[best]
    return z, y, pr


def boot(d, block=168, B=1000):
    n = len(d)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    m = d[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


data = {h: load(h) for h in HZ}
mae = {h: {m: float(np.mean(np.abs(p - data[h][1]))) for m, p in data[h][2].items()} for h in HZ}

# Fig 1: test MAE by horizon with 95% block-bootstrap intervals
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, m in enumerate(("naive", "ridge", "xgb", "lstm")):
    x = np.arange(len(HZ)) + (i - 1.5) * 0.18
    v = np.array([mae[h][m] for h in HZ])
    ci = np.array([boot(np.abs(data[h][2][m] - data[h][1])) for h in HZ])
    ax.errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", ms=4, capsize=2, color=COL[m], label=LAB[m])
ax.set_xticks(range(len(HZ)), [f"{h} h" for h in HZ])
ax.set_yscale("log")
ax.set_ylabel("Test MAE (MW, log scale)")
ax.set_xlabel("Forecast horizon")
ax.set_title("Test-period accuracy by horizon (95% block-bootstrap intervals)")
ax.legend(frameon=False)
save(fig, "fig01_mae_by_horizon")

# Fig 2: skill vs best naive
fig, ax = plt.subplots(figsize=(6.5, 4))
for m in ("ridge", "xgb", "lstm"):
    ax.plot(range(len(HZ)), [100 * (1 - mae[h][m] / mae[h]["naive"]) for h in HZ], "o-", color=COL[m], label=LAB[m])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(HZ)), [f"{h} h" for h in HZ])
ax.set_ylabel("MAE reduction vs best naive (%)")
ax.set_xlabel("Forecast horizon")
ax.set_title("Skill over the best naive baseline")
ax.legend(frameon=False)
save(fig, "fig02_skill_by_horizon")

# Fig 3: paired differences (forest plot)
pairs = [("xgb", "ridge"), ("lstm", "xgb"), ("lstm", "ridge")]
fig, axes = plt.subplots(1, len(pairs), figsize=(10, 3.6), sharey=True)
for ax, (a, b) in zip(axes, pairs):
    for j, h in enumerate(HZ):
        d = np.abs(data[h][2][a] - data[h][1]) - np.abs(data[h][2][b] - data[h][1])
        lo, hi = boot(d)
        sig = lo > 0 or hi < 0
        ax.errorbar(d.mean(), j, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color=COL[a] if sig else "#999999", capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{LAB[a]} minus {LAB[b]}")
    ax.set_xlabel("MAE difference (MW)")
axes[0].set_yticks(range(len(HZ)), [f"{h} h" for h in HZ])
axes[0].invert_yaxis()
fig.suptitle("Paired MAE differences (negative = first model better; grey = interval includes zero)", y=1.02)
save(fig, "fig03_paired_differences")

# Fig 4: a test week at 24 h
z, y, pr = data[24]
t = np.array(z["target_time_te"], dtype="datetime64[ns]").astype("datetime64[h]")
s = np.searchsorted(t, np.datetime64("2025-03-03T00"))
sl = slice(s, s + 168)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t[sl], y[sl] / 1000, color="k", lw=1.6, label="Actual")
for m in ("ridge", "xgb", "lstm"):
    ax.plot(t[sl], pr[m][sl] / 1000, color=COL[m], lw=1, label=LAB[m])
ax.set_ylabel("Demand (GW)")
ax.set_title("24 h-ahead forecasts, one test week (3-9 March 2025)")
ax.legend(frameon=False, ncol=4)
fig.autofmt_xdate()
save(fig, "fig04_week_24h")
