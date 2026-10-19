"""Part P1 of the completion batch (docs/experiment_protocol.md, entry of 2026-10-19): peak-exceedance classification. No model is fitted.

Event = a day whose actual peak (maximum of the 24 hourly values) is at or above the 90th percentile of the daily peaks of the 365 days before the test period (2023-07-01 to 2024-06-30),
a threshold known before the test period. A model predicts the event if its predicted daily peak is at or above the same threshold. Days with all 24 hours valid in both prediction sets.
Rolling 24 h-ahead forecasts (h24.npz, frozen main run) and the day-ahead forecasts (dayahead.npz); precision, recall, F1, base rate; 95% paired moving-block bootstrap over days (28-day blocks).
Writes reports/peak_events_test.md, reports/tables/peak_events_results.json, reports/figures/fig15_peak_events.png.   Usage: py -I scripts/run_peak_events.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "interim" / "predictions"
BLOCK, B, SEED = 28, 1000, 0
rng = np.random.default_rng(SEED)
COL = {"naive": "#7F7F7F", "ridge": "#0072B2", "xgb": "#D55E00", "lstm": "#009E73"}
LAB = {"naive": "Best naive", "ridge": "Ridge", "xgb": "XGBoost S4", "lstm": "LSTM L4"}
MS = ["naive", "ridge", "xgb", "lstm"]


def to_t(ns):
    return pd.to_datetime(np.asarray(ns, dtype="int64").astype("datetime64[ns]"))


def day_matrix(t, y, preds):
    df = pd.DataFrame({"day": t.normalize(), "hour": t.hour, "y": y, **preds})
    df = df[df.groupby("day")["hour"].transform("size") == 24].sort_values(["day", "hour"])
    return df["day"].iloc[::24].values, {k: df[k].to_numpy().reshape(-1, 24) for k in ["y", *preds]}


def prf(pred, act):
    tp, fp, fn = int((pred & act).sum()), int((pred & ~act).sum()), int((~pred & act).sum())
    pr, rc = tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": int((~pred & ~act).sum()), "precision": pr, "recall": rc, "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")}


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "india_hourly_canonical.csv", usecols=["timestamp", "electricity_demand_mw", "demand_imputed"], parse_dates=["timestamp"]).set_index("timestamp")
    ref = df.loc["2023-07-01":"2024-06-30"]
    ref = ref[ref["electricity_demand_mw"].notna() & ~ref["demand_imputed"].astype(bool)]["electricity_demand_mw"]
    g = ref.groupby(ref.index.normalize())
    peaks = g.max()[g.size() == 24]
    thr = float(np.percentile(peaks, 90))
    h24, da = np.load(PRED / "h24.npz", allow_pickle=True), np.load(PRED / "dayahead.npz", allow_pickle=True)
    zv = np.load(PRED / "dayahead_val.npz", allow_pickle=True)
    nk = "naive_a24" if np.mean(np.abs(zv["naive_a24"] - zv["y"])) <= np.mean(np.abs(zv["naive_a168"] - zv["y"])) else "naive_a168"
    t_r, y_r = to_t(h24["target_time_te"]), h24["y_te"].astype(float)
    rolling = {"naive": h24["te__snaive24"], "ridge": h24["te__ridge_best"], "xgb": h24["te__xgb_S4_full_A"], "lstm": h24["te__lstm_L4_full_A"]}
    t_d, y_d = to_t(da["T"]), da["y"].astype(float)
    dayah = {"naive": da[nk], "ridge": da["ridge"], "xgb": da["xgb"], "lstm": da["lstm"]}
    d_r, Wr = day_matrix(t_r, y_r, {m: v.astype(float) for m, v in rolling.items()})
    d_d, Wd = day_matrix(t_d, y_d, {m: v.astype(float) for m, v in dayah.items()})
    common = np.intersect1d(d_r, d_d)
    ir, id_ = np.isin(d_r, common), np.isin(d_d, common)
    assert np.allclose(Wr["y"][ir], Wd["y"][id_])
    act_peak = Wr["y"][ir].max(axis=1)
    act = act_peak >= thr
    n = len(act)
    res = {"threshold_MW": thr, "n_days": int(n), "n_reference_days": int(len(peaks)), "event_days": int(act.sum()), "base_rate_pct": float(100 * act.mean()), "rolling": {}, "dayahead": {}, "diffs": {}}
    P = {"rolling": {m: Wr[m][ir].max(axis=1) >= thr for m in MS}, "dayahead": {m: Wd[m][id_].max(axis=1) >= thr for m in MS}}
    for k in P:
        for m in MS:
            res[k][m] = prf(P[k][m], act)
    nb = int(np.ceil(n / BLOCK))
    idx = (rng.integers(0, n - BLOCK + 1, size=(B, nb))[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]

    def f1_boot(pred):
        tp, fp, fn = ((pred & act)[idx].sum(1)), ((pred & ~act)[idx].sum(1)), ((~pred & act)[idx].sum(1))
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    for k in P:
        for m in MS:
            res[k][m]["F1_ci"] = tuple(np.percentile(f1_boot(P[k][m]), [2.5, 97.5]))
    pairs = [("rolling", "xgb", "rolling", "ridge"), ("rolling", "lstm", "rolling", "xgb"), ("dayahead", "xgb", "dayahead", "ridge"), ("dayahead", "lstm", "dayahead", "xgb"),
             ("dayahead", "xgb", "rolling", "xgb"), ("dayahead", "ridge", "rolling", "ridge"), ("dayahead", "lstm", "rolling", "lstm")]
    for a_k, a_m, b_k, b_m in pairs:
        d = f1_boot(P[a_k][a_m]) - f1_boot(P[b_k][b_m])
        res["diffs"][f"{a_k}:{a_m}-{b_k}:{b_m}"] = (float(res[a_k][a_m]["F1"] - res[b_k][b_m]["F1"]), *map(float, np.percentile(d, [2.5, 97.5])))
    f = lambda v: f"{v:.3f}"
    md = ["# Peak-exceedance classification (generated by `scripts/run_peak_events.py`; part P1 of the completion batch)", "",
          "Declared in `docs/experiment_protocol.md` (entry of 2026-10-19) before it was computed; no model was fitted. "
          f"Event: a day whose actual peak is at or above {thr:,.0f} MW, the 90th percentile of the daily peaks of the {len(peaks)} complete days from 2023-07-01 to 2024-06-30 (known before the test period). "
          f"A model predicts the event if its predicted daily peak is at or above the same threshold. {n} test days with all 24 valid hours in both the rolling 24 h-ahead and the day-ahead prediction sets; "
          f"{res['event_days']} of them are event days ({res['base_rate_pct']:.1f}%; the reference year has 10% by construction, and demand kept growing, which is the likely reason for the difference). "
          "Rolling = the 24 h-ahead forecasts of the frozen main run, day-ahead = forecasts issued once at 10:00 on the previous day (naive: same hour on the latest earlier day). "
          f"Intervals: 95% paired moving-block bootstrap over days (blocks of {BLOCK} days, {B:,} resamples).", "",
          "| forecast | model | TP | FP | FN | TN | precision | recall | F1 [95% CI] |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, kl in (("rolling", "rolling 24 h-ahead"), ("dayahead", "day-ahead")):
        for m in MS:
            r = res[k][m]
            md.append(f"| {kl} | {LAB[m]} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | {f(r['precision'])} | {f(r['recall'])} | {f(r['F1'])} [{f(r['F1_ci'][0])}, {f(r['F1_ci'][1])}] |")
    md += ["", "Paired F1 differences (first minus second; 95% CI):", "", "| first | second | F1 difference (CI) |", "|---|---|---|"]
    for key, d in res["diffs"].items():
        a, b = key.split("-")
        lab = lambda s: f"{LAB[s.split(':')[1]]} ({'rolling' if s.startswith('rolling') else 'day-ahead'})"
        md.append(f"| {lab(a)} | {lab(b)} | {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f}) |")
    md += ["", "Differences whose interval includes zero are not distinguishable.", "", "## Caveats", "",
           f"- With only {n} days and {res['event_days']} event days the F1 values are noisy; the intervals are wide.",
           f"- The threshold was fixed from the year before the test period; the test-year base rate is {res['base_rate_pct']:.1f}% instead of 10%, so the setting is a moderately rare event.",
           "- The classification is derived from the hourly forecasts (the predicted peak is the maximum of the 24 predicted hours); a model trained for the event directly could do better.",
           "- The test year has been used several times for declared analyses; this only re-reads saved predictions.", ""]
    (ROOT / "reports" / "peak_events_test.md").write_text(chr(10).join(md), encoding="utf-8")
    (ROOT / "reports" / "tables" / "peak_events_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(chr(10).join(md))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, key, ttl in zip(axes, ("precision", "recall", "F1"), ("Precision", "Recall", "F1")):
        for i, m in enumerate(MS):
            ax.bar(i - 0.2, res["rolling"][m][key], 0.38, color=COL[m], label="rolling 24 h-ahead" if i == 0 else None)
            ax.bar(i + 0.2, res["dayahead"][m][key], 0.38, color=COL[m], alpha=0.5, hatch="//", label="day-ahead (10:00 on D-1)" if i == 0 else None)
        ax.set_xticks(range(4), [LAB[m].replace("Best naive", "Naive") for m in MS], rotation=15)
        ax.set_title(ttl)
        ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Peak-exceedance days (peak at or above {thr / 1000:,.0f} GW), test period: {n} days, {res['event_days']} events", y=1.03)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "fig15_peak_events.png", dpi=200)
    plt.close(fig)
    print("wrote fig15_peak_events")


if __name__ == "__main__":
    main()
