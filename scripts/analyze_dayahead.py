"""Analysis of the day-ahead model (task 9.7; design in docs/experiment_protocol.md, entry of 2026-10-19). No model is fitted: the script reads the saved predictions.
  py scripts/analyze_dayahead.py --split val    validation predictions (data/interim/predictions/dayahead_val.npz) -> reports/dayahead_analysis_validation.md
  py scripts/analyze_dayahead.py --split test   test predictions (dayahead.npz)                                      -> reports/dayahead_analysis_test.md, reports/figures/fig14_dayahead.png
Contents: hourly MAE/RMSE/MAPE and MAE by target hour; daily energy, daily peak and peak hour (days with all 24 valid hours); paired 95% moving-block bootstrap over whole days
(28-day blocks, 1,000 resamples, fixed seed); and the comparison with the rolling 24 h-ahead forecasts of the frozen main run on the same target hours (h24.npz).
"""
import argparse
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
FAM = {"ridge": ("ridge", "te__ridge_best", "va__ridge_best"), "xgb": ("xgb", "te__xgb_S4_full_A", "va__xgb_S4_full_A"), "lstm": ("lstm", "te__lstm_L4_full_A", "va__lstm_L4_full_A")}
LAB = {"naive": "Best naive", "ridge": "Ridge", "xgb": "XGBoost S4", "lstm": "LSTM L4"}
COL = {"ridge": "#0072B2", "xgb": "#D55E00", "lstm": "#009E73", "naive": "#7F7F7F"}
rng = np.random.default_rng(SEED)
f0 = lambda v: f"{v:,.0f}"
f1 = lambda v: f"{v:,.1f}"
sg = lambda v, f=f1: ("+" if v >= 0 else "-") + f(abs(v))


def boot_idx(n):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(B, nb))
    return (starts[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]


def ci_mean(vals, idx, wts=None):
    if wts is None:
        return tuple(np.percentile(vals[idx].mean(axis=1), [2.5, 97.5]))
    return tuple(np.percentile(vals[idx].sum(axis=1) / wts[idx].sum(axis=1), [2.5, 97.5]))


def to_days(t_ns):
    t = pd.to_datetime(np.asarray(t_ns, dtype="int64").astype("datetime64[ns]"))
    return t, pd.Series(t.normalize()), t.hour.to_numpy()


def day_table(day, hour, **cols):
    """Wide table (days x 24) for the arrays in cols; only days with all 24 hours present."""
    df = pd.DataFrame({"day": day.values, "hour": hour, **cols})
    cnt = df.groupby("day")["hour"].transform("size")
    df = df[cnt == 24].sort_values(["day", "hour"])
    days = df["day"].iloc[::24].values
    return days, {k: df[k].to_numpy().reshape(-1, 24) for k in cols}


def daily(Y, P):
    aE, aP, aH = Y.sum(axis=1) / 1000, Y.max(axis=1), Y.argmax(axis=1)
    pE, pP, pH = P.sum(axis=1) / 1000, P.max(axis=1), P.argmax(axis=1)
    return {"eE": pE - aE, "ePk": pP - aP, "hit0": (pH == aH).astype(float), "hit1": (np.abs(pH - aH) <= 1).astype(float), "aE": aE, "pE": pE}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    sp = ap.parse_args().split
    ev = "te" if sp == "test" else "va"
    z = np.load(PRED / ("dayahead.npz" if sp == "test" else "dayahead_val.npz"), allow_pickle=True)
    zv = np.load(PRED / "dayahead_val.npz", allow_pickle=True)
    h24 = np.load(PRED / "h24.npz", allow_pickle=True)
    y = z["y"].astype(float)
    t, day, hour = to_days(z["T"])
    naive_key = "naive_a24" if np.mean(np.abs(zv["naive_a24"] - zv["y"])) <= np.mean(np.abs(zv["naive_a168"] - zv["y"])) else "naive_a168"
    P = {"naive": z[naive_key].astype(float), "ridge": z["ridge"].astype(float), "xgb": z["xgb"].astype(float), "lstm": z["lstm"].astype(float)}
    MS = list(P)
    out = {"split": sp, "naive_choice": naive_key, "n_hours": int(len(y))}

    # ---- hourly accuracy; bootstrap over whole days (per-day sums and counts)
    dfh = pd.DataFrame({"day": day.values, "hour": hour, "y": y, **{m: P[m] for m in MS}})
    gd = dfh.groupby("day")
    days_all = np.array(sorted(dfh["day"].unique()))
    cnt = gd.size().reindex(days_all).to_numpy().astype(float)
    idx_all = boot_idx(len(days_all))

    def day_sum(vals):
        return pd.Series(vals).groupby(dfh["day"].values).sum().reindex(days_all).to_numpy()

    hourly = {}
    for m in MS:
        e = np.abs(P[m] - y)
        s = day_sum(e)
        hourly[m] = {"MAE": float(e.mean()), "MAE_ci": ci_mean(s, idx_all, cnt), "RMSE": float(np.sqrt(np.mean((P[m] - y) ** 2))), "MAPE": float(100 * np.mean(e / y))}
    pairs = [("xgb", "ridge"), ("lstm", "xgb"), ("lstm", "ridge"), ("ridge", "naive"), ("xgb", "naive"), ("lstm", "naive")]
    hd = {}
    for a, b in pairs:
        d = np.abs(P[a] - y) - np.abs(P[b] - y)
        hd[f"{a}-{b}"] = (float(d.mean()), *map(float, ci_mean(day_sum(d), idx_all, cnt)))
    by_hour = {m: [float(np.abs(P[m] - y)[hour == k].mean()) for k in range(24)] for m in MS}
    out["hourly"], out["hourly_diffs"], out["by_hour"] = hourly, hd, by_hour

    # ---- daily metrics (days with all 24 valid hours)
    days, W = day_table(day, hour, y=y, **{m: P[m] for m in MS})
    DQ = {m: daily(W["y"], W[m]) for m in MS}
    idx_d = boot_idx(len(days))
    dres = {}
    for m in MS:
        q = DQ[m]
        aE, aP = np.abs(q["eE"]), np.abs(q["ePk"])
        dres[m] = {"energy_MAE": float(aE.mean()), "energy_MAE_ci": tuple(map(float, ci_mean(aE, idx_d))), "energy_MAPE": float(100 * np.mean(aE / q["aE"])), "energy_bias": float(q["eE"].mean()),
                   "peak_MAE": float(aP.mean()), "peak_MAE_ci": tuple(map(float, ci_mean(aP, idx_d))), "peak_bias": float(q["ePk"].mean()),
                   "hit0": float(100 * q["hit0"].mean()), "hit1": float(100 * q["hit1"].mean())}
    dd = {}
    for a, b in pairs:
        dd[f"{a}-{b}"] = {k: (float(d.mean()), *map(float, ci_mean(d, idx_d))) for k, d in (("energy", np.abs(DQ[a]["eE"]) - np.abs(DQ[b]["eE"])), ("peak", np.abs(DQ[a]["ePk"]) - np.abs(DQ[b]["ePk"])),
                                                                                         ("hit1_pp", 100 * (DQ[a]["hit1"] - DQ[b]["hit1"])))}
    out["n_days"], out["daily"], out["daily_diffs"] = int(len(days)), dres, dd

    # ---- comparison with the rolling 24 h-ahead forecasts of the frozen run (same target hours)
    th, yh = np.asarray(h24["target_time_" + ("te" if sp == "test" else "va")], dtype="int64"), h24["y_" + ("te" if sp == "test" else "va")].astype(float)
    r24 = {m: h24[FAM[m][1 if sp == "test" else 2]].astype(float) for m in FAM}
    r24["naive"] = h24[("te__" if sp == "test" else "va__") + "snaive24"].astype(float)
    pos24 = pd.Series(np.arange(len(th)), index=th)
    key = np.asarray(z["T"], dtype="int64")
    common = np.isin(key, th)
    sel24 = pos24.reindex(key[common]).to_numpy()
    assert np.allclose(yh[sel24], y[common]), "actuals differ between the day-ahead and the 24 h-ahead sets"
    day_c, hour_c = day[common].reset_index(drop=True), hour[common]
    yc = y[common]
    cmp = {"n_hours": int(common.sum())}
    dfc = pd.DataFrame({"day": day_c.values})
    daysc = np.array(sorted(dfc["day"].unique()))
    cntc = dfc.groupby("day").size().reindex(daysc).to_numpy().astype(float)
    idc = boot_idx(len(daysc))
    sum_c = lambda v: pd.Series(v).groupby(dfc["day"].values).sum().reindex(daysc).to_numpy()
    cmp["hourly"] = {}
    for m in MS:
        ea, er = np.abs(P[m][common] - yc), np.abs(r24[m][sel24] - yc)
        d = ea - er
        cmp["hourly"][m] = {"dayahead": float(ea.mean()), "rolling": float(er.mean()), "diff": (float(d.mean()), *map(float, ci_mean(sum_c(d), idc, cntc)))}
    daysw, WW = day_table(day_c, hour_c, y=yc, **{m + "_da": P[m][common] for m in MS}, **{m + "_ro": r24[m][sel24] for m in MS})
    idw = boot_idx(len(daysw))
    cmp["n_days"] = int(len(daysw))
    cmp["daily"] = {}
    for m in MS:
        qa, qr = daily(WW["y"], WW[m + "_da"]), daily(WW["y"], WW[m + "_ro"])
        dE = np.abs(qa["eE"]) - np.abs(qr["eE"])
        dP = np.abs(qa["ePk"]) - np.abs(qr["ePk"])
        cmp["daily"][m] = {"energy_dayahead": float(np.abs(qa["eE"]).mean()), "energy_rolling": float(np.abs(qr["eE"]).mean()), "energy_diff": (float(dE.mean()), *map(float, ci_mean(dE, idw))),
                           "peak_dayahead": float(np.abs(qa["ePk"]).mean()), "peak_rolling": float(np.abs(qr["ePk"]).mean()), "peak_diff": (float(dP.mean()), *map(float, ci_mean(dP, idw))),
                           "hit1_dayahead": float(100 * qa["hit1"].mean()), "hit1_rolling": float(100 * qr["hit1"].mean())}
    out["comparison"] = cmp

    # ---- report
    tag = "test" if sp == "test" else "validation"
    md = [f"# Day-ahead model (issue 10:00 on D-1): analysis, {tag} period (generated by `scripts/analyze_dayahead.py`; plan task 9.7)", "",
          f"Design: `docs/experiment_protocol.md`, entry of 2026-10-19. Predictions of `scripts/run_dayahead.py` ({'frozen configuration, single test evaluation' if sp == 'test' else 'validation phase; hyper-parameters and early stopping were chosen on this period, so the numbers are optimistic'}). "
          f"{out['n_hours']:,} (day, hour) samples; {out['n_days']} days with all 24 valid hours. Naive = the better of the two naive baselines on validation ({'same hour on the latest earlier day' if naive_key == 'naive_a24' else 'same hour last week'}). "
          f"Intervals: 95% paired moving-block bootstrap over whole days (blocks of {BLOCK} days, {B:,} resamples, fixed seed).", "",
          "## Hourly accuracy over all target hours", "", "| model | MAE (MW) [95% CI] | RMSE (MW) | MAPE % |", "|---|---:|---:|---:|"]
    for m in MS:
        r = hourly[m]
        md.append(f"| {LAB[m]} | {f0(r['MAE'])} [{f0(r['MAE_ci'][0])}, {f0(r['MAE_ci'][1])}] | {f0(r['RMSE'])} | {r['MAPE']:.2f} |")
    md += ["", "Paired differences of the hourly MAE (first minus second; negative = first better):", "", "| first | second | difference (MW) | 95% CI |", "|---|---|---:|---|"]
    for a, b in pairs:
        d = hd[f"{a}-{b}"]
        md.append(f"| {LAB[a]} | {LAB[b]} | {sg(d[0], f0)} | {sg(d[1], f0)} to {sg(d[2], f0)} |")
    md += ["", "## MAE by target hour (MW)", "", "| target hour | " + " | ".join(LAB[m] for m in MS) + " |", "|---:|" + "---:|" * len(MS)]
    for k in range(24):
        md.append(f"| {k:02d}:00 | " + " | ".join(f0(by_hour[m][k]) for m in MS) + " |")
    md += ["", "## Daily energy, daily peak and peak hour", "", "| model | energy MAE (MU) [95% CI] | energy MAPE % | energy bias (MU) | peak MAE (MW) [95% CI] | peak bias (MW) | peak hour exact % | peak hour within 1 h % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MS:
        r = dres[m]
        md.append(f"| {LAB[m]} | {f1(r['energy_MAE'])} [{f1(r['energy_MAE_ci'][0])}, {f1(r['energy_MAE_ci'][1])}] | {r['energy_MAPE']:.2f} | {sg(r['energy_bias'])} | {f0(r['peak_MAE'])} [{f0(r['peak_MAE_ci'][0])}, {f0(r['peak_MAE_ci'][1])}] | {sg(r['peak_bias'], f0)} | {r['hit0']:.1f} | {r['hit1']:.1f} |")
    md += ["", "Paired differences (first minus second):", "", "| first | second | energy MAE (MU) | peak MAE (MW) | peak hour within 1 h (percentage points) |", "|---|---|---|---|---|"]
    for a, b in pairs:
        d = dd[f"{a}-{b}"]
        md.append(f"| {LAB[a]} | {LAB[b]} | {sg(d['energy'][0])} ({sg(d['energy'][1])} to {sg(d['energy'][2])}) | {sg(d['peak'][0], f0)} ({sg(d['peak'][1], f0)} to {sg(d['peak'][2], f0)}) | {sg(d['hit1_pp'][0])} ({sg(d['hit1_pp'][1])} to {sg(d['hit1_pp'][2])}) |")
    md += ["", "## Cost of issuing once per day: day-ahead versus the rolling 24 h-ahead forecasts", "",
           f"Same target hours ({cmp['n_hours']:,} hours; {cmp['n_days']} complete days in both sets). The rolling forecasts are the 24 h-ahead forecasts of the frozen main run (`h24.npz`), each issued 24 h before its target hour with the latest data at that time; "
           "the day-ahead forecasts are issued once at 10:00 on the previous day. The two sets of models use different leads, features and tuning, so this is a descriptive comparison, not a controlled experiment. Difference = day-ahead minus rolling (positive = the fixed issue time costs accuracy).", "",
           "| model | hourly MAE day-ahead (MW) | hourly MAE rolling 24 h (MW) | difference (95% CI) | energy MAE day-ahead (MU) | energy MAE rolling (MU) | difference (95% CI) | peak MAE day-ahead (MW) | peak MAE rolling (MW) | difference (95% CI) |", "|---|---:|---:|---|---:|---:|---|---:|---:|---|"]
    for m in MS:
        h_, d_ = cmp["hourly"][m], cmp["daily"][m]
        md.append(f"| {LAB[m]} | {f0(h_['dayahead'])} | {f0(h_['rolling'])} | {sg(h_['diff'][0], f0)} ({sg(h_['diff'][1], f0)} to {sg(h_['diff'][2], f0)}) | {f1(d_['energy_dayahead'])} | {f1(d_['energy_rolling'])} | "
                  f"{sg(d_['energy_diff'][0])} ({sg(d_['energy_diff'][1])} to {sg(d_['energy_diff'][2])}) | {f0(d_['peak_dayahead'])} | {f0(d_['peak_rolling'])} | {sg(d_['peak_diff'][0], f0)} ({sg(d_['peak_diff'][1], f0)} to {sg(d_['peak_diff'][2], f0)}) |")
    md += ["", "Differences whose interval includes zero are not distinguishable.", "", "## Caveats", "",
           f"- {out['n_days']} complete days from one {'test year' if sp == 'test' else 'validation half-year'}. Intervals reflect day-to-day sampling noise inside that period, not year-to-year variation.",
           "- Scenario A: no forecast weather, so the day-ahead problem is harder than in practice where weather forecasts exist; forecast temperature helped at 24-72 h in the main study.",
           "- The origin is the demand value of the hour starting at 10:00; the issue time and the availability of that value are assumptions of the design.",
           "- XGBoost and the LSTM are pooled over the 24 target hours (one model each), ridge has one model per target hour; the per-hour XGBoost and a 24-output LSTM were not run.",
           "- The test year has been used several times for declared analyses; this is a further use, with the design fixed beforehand.", ""]
    name = "dayahead_analysis_test.md" if sp == "test" else "dayahead_analysis_validation.md"
    (ROOT / "reports" / name).write_text(chr(10).join(md), encoding="utf-8")
    (ROOT / "reports" / "tables" / ("dayahead_results.json" if sp == "test" else "dayahead_results_validation.json")).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(chr(10).join(md))
    if sp != "test":
        return

    # ---- figure (test only)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    for m in MS:
        ax.plot(range(24), np.array(by_hour[m]) / 1000, "o-", ms=3, color=COL[m], label=LAB[m])
    ax.set_xlabel("Target hour of day D")
    ax.set_ylabel("MAE (GW)")
    ax.set_title("Day-ahead MAE by target hour (issued 10:00 on D-1)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    for i, m in enumerate(MS):
        r = hourly[m]
        ax.errorbar(i, r["MAE"], yerr=[[r["MAE"] - r["MAE_ci"][0]], [r["MAE_ci"][1] - r["MAE"]]], fmt="o", color=COL[m], capsize=3)
    ax.set_xticks(range(4), [LAB[m] for m in MS], rotation=15)
    ax.set_ylabel("Hourly MAE (MW)")
    ax.set_title("Hourly accuracy, all target hours (95% CI)")
    for ax, key, ttl, unit in ((axes[1, 0], "energy", "Daily energy error", "MAE (MU)"), (axes[1, 1], "peak", "Daily peak error", "MAE (MW)")):
        for i, m in enumerate(("ridge", "xgb", "lstm")):
            d_ = cmp["daily"][m]
            ax.bar(i - 0.2, d_[key + "_dayahead"], 0.38, color=COL[m], label="day-ahead (10:00 on D-1)" if i == 0 else None)
            ax.bar(i + 0.2, d_[key + "_rolling"], 0.38, color=COL[m], alpha=0.5, hatch="//", label="rolling 24 h-ahead" if i == 0 else None)
        ax.set_xticks(range(3), [LAB[m] for m in ("ridge", "xgb", "lstm")])
        ax.set_ylabel(unit)
        ax.set_title(ttl + ": day-ahead (solid) vs rolling 24 h-ahead (hatched)")
        ax.title.set_fontsize(9)
    fig.suptitle(f"Operational day-ahead model, test period ({out['n_days']} complete days)", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "fig14_dayahead.png", dpi=200)
    plt.close(fig)
    print("wrote fig14_dayahead")


if __name__ == "__main__":
    main()
