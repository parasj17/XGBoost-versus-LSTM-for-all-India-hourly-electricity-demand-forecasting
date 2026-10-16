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


# ---------- batch 2: figures read from the generated reports and result tables ----------
import json
import re

import pandas as pd
from matplotlib.patches import FancyBboxPatch

TAB, REP = ROOT / "reports" / "tables", ROOT / "reports"
HZL = [f"{h} h" for h in HZ]


def jl(name):
    return [json.loads(line) for line in open(TAB / name, encoding="utf-8")]


def md_sections(path):
    """{horizon: [table rows as lists of cells]} for reports whose sections start with '## Horizon N'."""
    out, cur = {}, None
    for line in open(REP / path, encoding="utf-8"):
        m = re.match(r"## Horizon (\d+)", line)
        if m:
            cur = int(m.group(1))
            out[cur] = []
        elif cur is not None and line.startswith("|") and not line.startswith("|---"):
            out[cur].append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


num = lambda s: float(s.replace(",", "").replace("%", ""))

# Fig 5: feature-group ablation, MAE relative to the history-only set
SETS = [("S0_history", "S0 history"), ("S1_calendar", "S1 + calendar"), ("S2_holidays", "S2 + holidays"), ("S3_weatherA", "S3 + past weather"), ("S4_full_A", "S4 + solar/wind"), ("S5_oracle_B", "S5 oracle weather")]
abl = {}
for h in (1, 6, 24, 48):
    z, y, _ = data[h]
    abl[h] = {k: float(np.mean(np.abs(z["te__xgb_" + k] - y))) for k, _ in SETS}
for h, rows in md_sections("long_ablation_test.md").items():
    abl[h] = {r[0].split()[0]: num(r[1]) for r in rows if r[0].startswith("S") and len(r) == 2 and r[0][1].isdigit()}
fig, ax = plt.subplots(figsize=(7.5, 4.4))
pal = ["#999999", "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
for (k, lab), c in zip(SETS, pal):
    ax.plot(range(len(HZ)), [100 * (abl[h][k] / abl[h]["S0_history"] - 1) for h in HZ], "o--" if k == "S5_oracle_B" else "o-", color=c, label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(HZ)), HZL)
ax.set_ylabel("Test MAE change vs history-only S0 (%)")
ax.set_xlabel("Forecast horizon")
ax.set_title("Feature-group ablation (XGBoost, test period)")
ax.legend(frameon=False, fontsize=8, ncol=2)
save(fig, "fig05_feature_ablation")

# Fig 6: walk-forward (quarterly retraining), MAE per test quarter
wf = jl("walk_forward.jsonl") + jl("walk_forward_long.jsonl")
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for ax, h in zip(axes.ravel(), HZ):
    rows = sorted([r for r in wf if r["horizon"] == h], key=lambda r: r["window"])
    q = [r["window"] for r in rows]
    for m, key in (("naive", "best_naive"), ("ridge", "ridge"), ("xgb", "xgb"), ("lstm", "lstm")):
        ax.plot(q, [r[key] for r in rows], "o-", ms=3, color=COL[m], label=LAB[m])
    ax.set_title(f"{h} h")
    ax.tick_params(axis="x", rotation=45)
axes[0, 0].legend(frameon=False, fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("MAE (MW)")
fig.suptitle("Walk-forward evaluation: models refitted each quarter, MAE per quarter", y=1.0)
fig.tight_layout()
save(fig, "fig06_walk_forward")

# Fig 7: daily totals, multi-year expanding window
dm = jl("daily_multiyear.jsonl")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, h in zip(axes, (1, 7)):
    rows = sorted([r for r in dm if r["horizon"] == h], key=lambda r: r["year"])
    for m, key in (("naive", "naive"), ("ridge", "ridge"), ("xgb", "xgb"), ("lstm", "lstm")):
        ax.plot([r["year"] for r in rows], [r[key] for r in rows], "o-", ms=3, color=COL[m], label=LAB[m])
    ax.set_title(f"Daily total demand, {h}-day-ahead")
    ax.set_xlabel("Evaluation year (trained on all earlier years)")
axes[0].set_ylabel("MAE of the daily total (MU = GWh per day)")
axes[0].legend(frameon=False)
save(fig, "fig07_daily_multiyear")

# Fig 8: prediction intervals, XGBoost S4 (coverage and width)
iv = {}
for f in ("uncertainty.md", "long_uncertainty.md"):
    for line in open(REP / f, encoding="utf-8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 6 and c[0].isdigit() and c[1] == "XGBoost S4":
            iv.setdefault(c[2], {})[int(c[0])] = (num(c[3]), num(c[4]))
meth = {"split conformal (absolute)": ("Split conformal", "#0072B2"), "rolling conformal (last 720 h of known errors)": ("Rolling conformal", "#009E73"),
        "XGBoost quantile regression, raw 5%-95%": ("Quantile regression, raw", "#D55E00"), "XGBoost quantile regression + CQR": ("Quantile regression + CQR", "#CC79A7")}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for k, (lab, c) in meth.items():
    axes[0].plot(range(len(HZ)), [iv[k][h][0] for h in HZ], "o-", color=c, label=lab)
    axes[1].plot(range(len(HZ)), [iv[k][h][1] / 1000 for h in HZ], "o-", color=c, label=lab)
axes[0].axhline(90, color="k", ls="--", lw=0.8)
axes[0].text(2.6, 90.4, "nominal 90%", fontsize=8)
axes[0].set_ylabel("Coverage of the 90% interval (%)")
axes[1].set_ylabel("Mean interval width (GW)")
for ax in axes:
    ax.set_xticks(range(len(HZ)), HZL)
    ax.set_xlabel("Forecast horizon")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Prediction intervals on the test period (XGBoost S4)", y=1.02)
save(fig, "fig08_prediction_intervals")

# Fig 9: computational cost (training time per model)
tr = {r["horizon"]: r["refit_seconds"] for r in jl("test_results.jsonl")}
H4 = [1, 6, 24, 48]
mods = [("ridge", "ridge_best"), ("xgb", "xgb_S4_full_A"), ("lstm", "lstm_L4_full_A")]
fig, ax = plt.subplots(figsize=(6.8, 4))
for i, (m, k) in enumerate(mods):
    ax.bar(np.arange(len(H4)) + (i - 1) * 0.26, [tr[h][k] for h in H4], 0.26, color=COL[m], label=LAB[m])
ax.set_yscale("log")
ax.set_xticks(range(len(H4)), [f"{h} h" for h in H4])
ax.set_ylabel("Final-fit time (s, log scale)")
ax.set_xlabel("Forecast horizon")
ax.set_title("Training cost of the final models (as logged in the test run)")
ax.legend(frameon=False)
save(fig, "fig09_training_cost")

# Fig 10: learning curves (validation period)
lc = {}
for h, rows in md_sections("learning_curve.md").items():
    lc[h] = [(num(r[1]), num(r[3]), num(r[4])) for r in rows if r[0].split()[0] in ("12", "18", "full")]
for h, rows in md_sections("long_learning_curve.md").items():
    lc[h] = [(num(r[1]), num(r[2]), num(r[3])) for r in rows if r[0].split()[0] in ("12", "18", "full")]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, h in zip(axes.ravel(), HZ):
    n = [r[0] for r in lc[h]]
    ax.plot(n, [r[1] for r in lc[h]], "o-", color=COL["xgb"], label="XGBoost S4")
    ax.plot(n, [r[2] for r in lc[h]], "o-", color=COL["lstm"], label="LSTM" + (" L1" if h <= 48 else " L4"))
    ax.set_title(f"{h} h")
axes[0, 0].legend(frameon=False, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Training samples")
for ax in axes[:, 0]:
    ax.set_ylabel("Validation MAE (MW)")
fig.suptitle("Learning curves: validation MAE vs amount of training data (12 months, 18 months, full)", y=1.0)
fig.tight_layout()
save(fig, "fig10_learning_curves")

# Fig 11: demand growth and the data splits
df = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", usecols=["timestamp", "electricity_demand_mw"], parse_dates=["timestamp"]).set_index("timestamp")
d = df["electricity_demand_mw"].resample("D").mean() / 1000
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(d.index, d, color="#BBBBBB", lw=0.6, label="Daily mean")
ax.plot(d.index, d.rolling(30, center=True, min_periods=15).mean(), color="k", lw=1.4, label="30-day mean")
for a, b, c, lab in (("2021-09-01", "2023-12-31", "#56B4E9", "Training"), ("2024-01-01", "2024-06-30", "#E69F00", "Validation"), ("2024-07-01", "2025-06-24", "#D55E00", "Test")):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.15, label=lab)
ax.set_ylabel("All-India demand (GW)")
ax.set_title("Hourly demand series (daily means) and the chronological splits")
ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower right")
save(fig, "fig11_demand_and_splits")

# Fig 12: study design diagram
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)


def box(x, y, w, h, text, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=c, ec="#444444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


box(0.1, 3.6, 2.0, 1.1, "Data\nall-India hourly demand,\nweather, calendar,\nsolar/wind", "#EEEEEE")
box(2.5, 3.6, 2.2, 1.1, "Cleaning and features\nvalid-sample rule,\nratio-to-anchor target,\nfeature groups g0-g4", "#EEEEEE")
box(5.1, 3.6, 2.2, 1.1, "Models\nnaive, ridge, XGBoost,\nLSTM (3 seeds)", "#EEEEEE")
box(7.7, 3.6, 2.2, 1.1, "Horizons\n1, 6, 24, 48 h\n+ 72, 168 h extension", "#EEEEEE")
box(0.1, 1.9, 3.2, 1.0, "Training period\nto 2023-12-31\n(fit; cross-validated tuning)", "#56B4E9")
box(3.5, 1.9, 3.0, 1.0, "Validation 2024-H1\n(tuning, early stopping,\ninterval calibration)", "#E69F00")
box(6.7, 1.9, 3.2, 1.0, "Test 2024-07-01 to 2025-06-24\n(each variant evaluated once)", "#D55E00")
box(0.1, 0.2, 4.7, 1.0, "Analyses on the test period\npaired block bootstrap, ablation, seasonal breakdowns,\nprediction intervals, cost", "#EEEEEE")
box(5.2, 0.2, 4.7, 1.0, "Robustness\nrolling origin, walk-forward, learning curve,\ndaily multi-year, equal-budget LSTM", "#EEEEEE")
for x in (2.1, 4.7, 7.3):
    ax.annotate("", xy=(x + 0.4, 4.15), xytext=(x, 4.15), arrowprops=dict(arrowstyle="->", color="#444444"))
ax.set_title("Study design", fontsize=11)
save(fig, "fig12_study_design")
