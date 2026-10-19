"""Analysis of the completion batch (docs/experiment_protocol.md, entries of 2026-10-19 and 2026-10-20). No model is fitted; the script reads the saved predictions.
  py scripts/analyze_completion.py --split val     validation predictions (code check and validation reports; calibration of the interval methods is then on the same period, so the interval numbers are only a check)
  py scripts/analyze_completion.py --split test    test predictions -> reports/dayahead_models_test.md, reports/other_families_test.md, reports/figures/fig16_dayahead_models.png, fig17_other_families.png
Section A (day-ahead task): hourly accuracy and paired differences of all day-ahead models, daily energy and peak, the feature ablation of the pooled XGBoost, 90% prediction intervals.
Section B (other model families on the main horizons): LightGBM at 1-168 h, GRU/TCN/Transformer at 1 h, 24 h, compared with XGBoost S4, LSTM L4 and ridge.
Bootstrap: 95% paired moving-block bootstrap over whole days (28-day blocks) for the day-ahead task, 168-hour blocks for the main horizons; 1,000 resamples, fixed seed.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
PRED = ROOT / "data" / "interim" / "predictions"
BLOCK, B, SEED = 28, 1000, 0
rng = np.random.default_rng(SEED)
f0 = lambda v: f"{v:,.0f}"
f1 = lambda v: f"{v:,.1f}"
sg = lambda v, f=f0: ("+" if v >= 0 else "-") + f(abs(v))


def ld(name):
    p = PRED / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def boot_idx(n, block=BLOCK):
    nb = int(np.ceil(n / block))
    return (rng.integers(0, n - block + 1, size=(B, nb))[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]


def ci_mean(v, idx):
    return tuple(map(float, np.percentile(v[idx].mean(axis=1), [2.5, 97.5])))


def day_table(day, hour, cols):
    df = pd.DataFrame({"day": day.values, "hour": hour, **cols})
    df = df[df.groupby("day")["hour"].transform("size") == 24].sort_values(["day", "hour"])
    return df["day"].iloc[::24].values, {k: df[k].to_numpy().reshape(-1, 24) for k in cols}


def daily(Y, P):
    aE, aP, aH = Y.sum(axis=1) / 1000, Y.max(axis=1), Y.argmax(axis=1)
    return {"eE": P.sum(axis=1) / 1000 - aE, "ePk": P.max(axis=1) - aP, "hit1": (np.abs(P.argmax(axis=1) - aH) <= 1).astype(float), "aE": aE}


def section_a(sp, md_all):
    sfx = "" if sp == "test" else "_val"
    base = ld("dayahead.npz" if sp == "test" else "dayahead_val.npz")
    T0, y0 = np.asarray(base["T"], dtype="int64"), base["y"].astype(float)
    ix0 = pd.Series(np.arange(len(T0)), index=T0)
    zv = ld("dayahead_val.npz")
    nk = "naive_a24" if np.mean(np.abs(zv["naive_a24"] - zv["y"])) <= np.mean(np.abs(zv["naive_a168"] - zv["y"])) else "naive_a168"
    M = {"naive": ("Best naive", base[nk]), "ridge": ("Ridge", base["ridge"]), "xgb": ("XGBoost S4, pooled", base["xgb"]), "lstm": ("LSTM L4, pooled", base["lstm"])}

    def add(key, label, fname, arr_key=None, aligned=False):
        z = ld(fname)
        if z is None:
            return
        if aligned:                                     # classical: predictions already aligned to the base rows
            M[key] = (label, z["pred"].astype(float))
            return
        arr = z[arr_key].astype(float)
        T = np.asarray(z[arr_key + "_T"] if (arr_key + "_T") in z.files else z["T"], dtype="int64")
        yy = z[arr_key + "_y"].astype(float) if (arr_key + "_y") in z.files else z["y"].astype(float)
        pos = ix0.reindex(T).to_numpy()
        out = np.full(len(T0), np.nan)
        out[pos.astype(int)] = arr
        assert np.allclose(yy, y0[pos.astype(int)])
        M[key] = (label, out)

    add("xgb_perhour", "XGBoost S4, one model per hour", f"dayahead_ext{sfx}.npz", "xgb_perhour")
    add("lstm_l1", "LSTM L1 (history + calendar), pooled", f"dayahead_ext{sfx}.npz", "lstm_l1")
    add("lstm24", "LSTM L4, 24 outputs", f"dayahead_lstm24{sfx}.npz", "lstm24")
    add("lightgbm", "LightGBM S4, pooled", f"lightgbm{sfx}.npz", "dayahead")
    for nm, lab in (("gru", "GRU"), ("tcn", "TCN"), ("transformer", "Transformer")):
        add(nm, lab + " (patched)" if nm == "transformer" else lab, f"neural_{nm}{sfx}.npz", "dayahead")
    add("sarimax", "SARIMAX (references)", f"classical_sarimax{sfx}.npz", aligned=True)
    add("prophet", "Prophet (references)", f"classical_prophet{sfx}.npz", aligned=True)
    keys = list(M)
    ok = np.ones(len(T0), bool)
    for k in keys:
        ok &= ~np.isnan(M[k][1])
    t = pd.to_datetime(T0[ok].astype("datetime64[ns]"))
    y, P = y0[ok], {k: M[k][1][ok].astype(float) for k in keys}
    day = pd.Series(t.normalize())
    hour = t.hour.to_numpy()
    days_all = np.array(sorted(day.unique()))
    cnt = day.groupby(day.values).size().reindex(days_all).to_numpy().astype(float)
    idx_all = boot_idx(len(days_all))
    sums = lambda v: pd.Series(v).groupby(day.values).sum().reindex(days_all).to_numpy()

    def boot_hour(d):
        s = sums(d)
        return tuple(map(float, np.percentile(s[idx_all].sum(axis=1) / cnt[idx_all].sum(axis=1), [2.5, 97.5])))

    ref_key = "xgb"
    res = {"hourly": {}, "vs_xgb": {}, "vs_ridge": {}}
    for k in keys:
        e = np.abs(P[k] - y)
        res["hourly"][k] = (float(e.mean()), *boot_hour(e), float(100 * np.mean(e / y)))
        for rk, store in (("xgb", "vs_xgb"), ("ridge", "vs_ridge")):
            if k != rk:
                d = e - np.abs(P[rk] - y)
                res[store][k] = (float(d.mean()), *boot_hour(d))
    days, W = day_table(day, hour, {"y": y, **P})
    DQ = {k: daily(W["y"], W[k]) for k in keys}
    idx_d = boot_idx(len(days))
    res["daily"] = {k: {"energy": float(np.abs(DQ[k]["eE"]).mean()), "peak": float(np.abs(DQ[k]["ePk"]).mean()), "hit1": float(100 * DQ[k]["hit1"].mean()), "bias_peak": float(DQ[k]["ePk"].mean()), "bias_energy": float(DQ[k]["eE"].mean())} for k in keys}
    res["daily_vs_xgb"] = {k: {"energy": (float((np.abs(DQ[k]["eE"]) - np.abs(DQ["xgb"]["eE"])).mean()), *ci_mean(np.abs(DQ[k]["eE"]) - np.abs(DQ["xgb"]["eE"]), idx_d)),
                              "peak": (float((np.abs(DQ[k]["ePk"]) - np.abs(DQ["xgb"]["ePk"])).mean()), *ci_mean(np.abs(DQ[k]["ePk"]) - np.abs(DQ["xgb"]["ePk"]), idx_d))} for k in keys if k != "xgb"}
    ev = "test" if sp == "test" else "validation"
    md = [f"# Day-ahead models: all variants, {ev} period (generated by `scripts/analyze_completion.py`; completion batch)", "",
          f"Task: forecasts issued at 10:00 on day D-1 for the 24 hours of day D (`docs/experiment_protocol.md`, entries of 2026-10-19 and 2026-10-20). {int(ok.sum()):,} (day, hour) samples, {len(days)} complete days; all models are compared on the same samples. "
          + ("Single test evaluations with frozen configurations; tuning budgets differ between models (XGBoost 30 trials, LSTM 30 trials, LightGBM 30, GRU/TCN/Transformer 12, 24-output LSTM 30; SARIMAX and Prophet are not tuned)." if sp == "test" else "Validation predictions: hyper-parameters and early stopping were chosen on this period, so the numbers are optimistic."),
          f"Intervals: 95% paired moving-block bootstrap over whole days (blocks of {BLOCK} days, {B:,} resamples).", "",
          "## Hourly accuracy", "", "| model | MAE (MW) [95% CI] | MAPE % | vs XGBoost S4, pooled (MW, 95% CI) | vs ridge (MW, 95% CI) |", "|---|---:|---:|---|---|"]
    for k in sorted(keys, key=lambda k: res["hourly"][k][0]):
        h = res["hourly"][k]
        vx = res["vs_xgb"].get(k)
        vr = res["vs_ridge"].get(k)
        md.append(f"| {M[k][0]} | {f0(h[0])} [{f0(h[1])}, {f0(h[2])}] | {h[3]:.2f} | " + (f"{sg(vx[0])} ({sg(vx[1])} to {sg(vx[2])})" if vx else "reference") + " | " + (f"{sg(vr[0])} ({sg(vr[1])} to {sg(vr[2])})" if vr else "reference") + " |")
    md += ["", "Differences are model minus reference (negative = the model is better); an interval that includes zero is not distinguishable.", "", "## Daily energy, daily peak and peak hour", "",
           "| model | energy MAE (MU) | vs XGBoost S4, pooled (MU, 95% CI) | peak MAE (MW) | vs XGBoost S4, pooled (MW, 95% CI) | peak bias (MW) | peak hour within 1 h % |", "|---|---:|---|---:|---|---:|---:|"]
    for k in sorted(keys, key=lambda k: res["daily"][k]["energy"]):
        d, v = res["daily"][k], res["daily_vs_xgb"].get(k)
        md.append(f"| {M[k][0]} | {f1(d['energy'])} | " + (f"{sg(v['energy'][0], f1)} ({sg(v['energy'][1], f1)} to {sg(v['energy'][2], f1)})" if v else "reference") + f" | {f0(d['peak'])} | "
                  + (f"{sg(v['peak'][0])} ({sg(v['peak'][1])} to {sg(v['peak'][2])})" if v else "reference") + f" | {sg(d['bias_peak'])} | {d['hit1']:.1f} |")
    out = {"models": {k: M[k][0] for k in keys}, "res": res, "n_days": int(len(days))}

    # ---- feature ablation of the pooled XGBoost
    ext = ld(f"dayahead_ext{sfx}.npz")
    abl_rows = []
    if ext is not None:
        names = ["S0_history", "S1_calendar", "S2_holidays", "S3_weatherA", "S6_forecastC", "S5_oracle_B"]
        pos = ix0.reindex(np.asarray(ext["T"], dtype="int64")).to_numpy().astype(int)
        A = {"S4_full_A": base["xgb"].astype(float)}
        for n in names:
            v = np.full(len(T0), np.nan)
            v[pos] = ext["xgb_" + n]
            A[n] = v
        okA = ~np.isnan(np.vstack(list(A.values()))).any(axis=0)
        tA = pd.to_datetime(T0[okA].astype("datetime64[ns]"))
        dayA = pd.Series(tA.normalize())
        dA = np.array(sorted(dayA.unique()))
        cA = dayA.groupby(dayA.values).size().reindex(dA).to_numpy().astype(float)
        idA = boot_idx(len(dA))
        sA = lambda v: pd.Series(v).groupby(dayA.values).sum().reindex(dA).to_numpy()
        yA = y0[okA]
        order = ["S0_history", "S1_calendar", "S2_holidays", "S3_weatherA", "S4_full_A", "S6_forecastC", "S5_oracle_B"]
        md += ["", "## Feature ablation of the pooled XGBoost (hyper-parameters of the day-ahead S4 model, trees by early stopping on validation)", "",
               "| feature set | MAE (MW) | change vs the previous cumulative set (MW, 95% CI) | change vs S4 (MW, 95% CI) |", "|---|---:|---|---|"]
        prev = None
        for n in order:
            e = np.abs(A[n][okA] - yA)
            cur = float(e.mean())
            dv = e - np.abs(A["S4_full_A"][okA] - yA)
            civ = np.percentile(sA(dv)[idA].sum(axis=1) / cA[idA].sum(axis=1), [2.5, 97.5])
            if prev is not None and n not in ("S6_forecastC", "S5_oracle_B"):
                dp = e - np.abs(A[prev][okA] - yA)
                cip = np.percentile(sA(dp)[idA].sum(axis=1) / cA[idA].sum(axis=1), [2.5, 97.5])
                pstr = f"{sg(dp.mean())} ({sg(cip[0])} to {sg(cip[1])})"
            else:
                pstr = "" if n in ("S0_history",) else "(not a cumulative step)"
            lab = n + (" (oracle weather, upper bound)" if n == "S5_oracle_B" else (" (archived temperature forecasts; issue times unverified)" if n == "S6_forecastC" else ""))
            md.append(f"| {lab} | {f0(cur)} | {pstr} | " + ("reference" if n == "S4_full_A" else f"{sg(dv.mean())} ({sg(civ[0])} to {sg(civ[1])})") + " |")
            if n not in ("S6_forecastC", "S5_oracle_B"):
                prev = n
            abl_rows.append((n, cur))
        out["ablation"] = abl_rows

    # ---- 90% prediction intervals for the pooled XGBoost, ridge and LSTM L4
    iv = []
    if ext is not None:
        pos_e = ix0.reindex(np.asarray(ext["T"], dtype="int64")).to_numpy().astype(int)
        q05, q95 = np.full(len(T0), np.nan), np.full(len(T0), np.nan)
        q05[pos_e], q95[pos_e] = ext["q05"], ext["q95"]
        ev_ext = ld("dayahead_ext_val.npz")
        vbase = ld("dayahead_val.npz")
        Tv = np.asarray(vbase["T"], dtype="int64")
        pos_v = pd.Series(np.arange(len(Tv)), index=Tv).reindex(np.asarray(ev_ext["T"], dtype="int64")).to_numpy().astype(int)
        qv05, qv95 = np.full(len(Tv), np.nan), np.full(len(Tv), np.nan)
        qv05[pos_v], qv95[pos_v] = ev_ext["q05"], ev_ext["q95"]
        alpha = 0.1
        yv = vbase["y"].astype(float)
        order_t = np.argsort(T0)
        tcal = np.asarray(Tv, dtype="int64")

        def score(l, u, yy):
            return float(np.mean((u - l) + (2 / alpha) * np.maximum(l - yy, 0) + (2 / alpha) * np.maximum(yy - u, 0)))

        def cover(l, u, yy):
            return float(100 * np.mean((yy >= l) & (yy <= u))), float(np.mean(u - l)), score(l, u, yy)

        for k, lab in (("xgb", "XGBoost S4, pooled"), ("ridge", "Ridge"), ("lstm", "LSTM L4, pooled")):
            pt = base[k].astype(float)
            ev_abs = np.abs(vbase[k].astype(float) - yv)
            n = len(ev_abs)
            q = np.quantile(ev_abs, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
            iv.append((lab, "split conformal (absolute)") + cover(pt - q, pt + q, y0))
            # rolling conformal: the errors known at the origin (target time <= 10:00 on the day before the target day), last 720
            allT = np.concatenate([tcal, T0])
            allE = np.concatenate([ev_abs, np.abs(pt - y0)])
            srt = np.argsort(allT)
            Ts, Es = allT[srt], allE[srt]
            dts = pd.to_datetime(T0.astype("datetime64[ns]")).normalize()
            lo, hi = np.zeros(len(T0)), np.zeros(len(T0))
            qd = {}
            for D in pd.unique(dts):
                t0 = (D - pd.Timedelta(hours=14)).to_datetime64().astype("datetime64[ns]").astype("int64")
                j = np.searchsorted(Ts, t0, side="right")
                seg = Es[max(0, j - 720):j]
                qd[D] = np.quantile(seg, 0.9) if len(seg) >= 24 else np.quantile(ev_abs, 0.9)
            qr = np.array([qd[D] for D in dts])
            iv.append((lab, "rolling conformal (last 720 errors known at the origin)") + cover(pt - qr, pt + qr, y0))
        # quantile regression, raw and with conformalised correction (CQR)
        iv.append(("XGBoost S4, pooled", "quantile regression, raw 5%-95%") + cover(q05, q95, y0))
        s = np.maximum(qv05 - yv, yv - qv95)
        c = np.quantile(s, min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s)))
        iv.append(("XGBoost S4, pooled", "quantile regression + CQR") + cover(q05 - c, q95 + c, y0))
        md += ["", "## 90% prediction intervals for the day-ahead forecasts", "",
               "Calibrated on the validation period only" + (" (in this validation-mode run the calibration and the evaluation period are the same, so the numbers are only a code check)" if sp != "test" else "") + "; rolling conformal uses only errors of forecasts whose target time is not later than the origin (10:00 on the previous day). "
               "Coverage target 90%; interval score = width + 20 x the miss distance (lower is better).", "",
               "| point forecast | method | coverage % | mean width (MW) | interval score (MW) |", "|---|---|---:|---:|---:|"]
        for lab, meth, cv, w, sc in iv:
            md.append(f"| {lab} | {meth} | {cv:.1f} | {f0(w)} | {f0(sc)} |")
        out["intervals"] = iv
    md += ["", "## Caveats", "",
           f"- {len(days)} complete days from one {ev} period; intervals reflect day-to-day noise inside the period, not year-to-year variation. The test year has been used several times for declared analyses.",
           "- Scenario A (no forecast weather) except where labelled; the issue time and the availability of the 10:00 demand value are design assumptions.",
           "- XGBoost, LightGBM, the LSTMs, GRU, TCN and Transformer are pooled over the 24 target hours except where labelled (one model per hour; 24 outputs); tuning budgets differ (see above).",
           "- SARIMAX and Prophet are references with fixed specifications (refitted weekly, no tuning, no weather); Prophet has no yearly term and no conditioning on recent data, so a large error is expected.",
           "- S6 uses archived temperature forecasts whose issue times were not verified; S5 uses oracle weather and is an upper bound, not an achievable result.", ""]
    suffix = "test" if sp == "test" else "validation"
    (ROOT / "reports" / f"dayahead_models_{suffix}.md").write_text(chr(10).join(md), encoding="utf-8")
    (ROOT / "reports" / "tables" / f"dayahead_models_{suffix}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    md_all.append(chr(10).join(md[:3]))
    return M, res, abl_rows, iv, keys


def figure_a(M, res, abl, iv, keys):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ks = sorted(keys, key=lambda k: res["hourly"][k][0])
    for i, k in enumerate(ks):
        h = res["hourly"][k]
        ax.errorbar(h[0], i, xerr=[[h[0] - h[1]], [h[2] - h[0]]], fmt="o", color="#D55E00" if k == "xgb" else "#0072B2", capsize=2)
    ax.set_yticks(range(len(ks)), [M[k][0] for k in ks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Hourly MAE (MW), 95% block-bootstrap interval")
    ax.set_title("Day-ahead models: hourly accuracy")
    ax = axes[0, 1]
    ks2 = sorted(keys, key=lambda k: res["daily"][k]["energy"])
    ax.barh(range(len(ks2)), [res["daily"][k]["energy"] for k in ks2], color=["#D55E00" if k == "xgb" else "#0072B2" for k in ks2])
    ax.set_yticks(range(len(ks2)), [M[k][0] for k in ks2], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Daily energy MAE (MU)")
    ax.set_title("Day-ahead models: daily energy")
    ax = axes[1, 0]
    if abl:
        ax.plot(range(len(abl)), [v for _, v in abl], "o-", color="#D55E00")
        ax.set_xticks(range(len(abl)), [n.replace("_", "\n", 1) for n, _ in abl], fontsize=7)
        ax.set_ylabel("MAE (MW)")
        ax.set_title("Feature ablation, pooled XGBoost (S6 = temperature forecasts, S5 = oracle weather)")
        ax.title.set_fontsize(8)
    ax = axes[1, 1]
    if iv:
        labs = [f"{a.split(',')[0]}: {b.split('(')[0].strip()}" for a, b, *_ in iv]
        cov = [r[2] for r in iv]
        ax.barh(range(len(iv)), cov, color="#009E73")
        ax.axvline(90, color="k", ls="--", lw=0.8)
        ax.set_yticks(range(len(iv)), labs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(70, 100)
        ax.set_xlabel("Coverage of the 90% interval (%)")
        ax.set_title("Day-ahead prediction intervals")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "fig16_dayahead_models.png", dpi=200)
    plt.close(fig)


def section_b(sp):
    sfx = "" if sp == "test" else "_val"
    ev = "te" if sp == "test" else "va"
    lg = ld(f"lightgbm{sfx}.npz")
    neu = {nm: ld(f"neural_{nm}{sfx}.npz") for nm in ("gru", "tcn", "transformer")}
    rows, md = {}, []
    HZ = [1, 6, 24, 48, 72, 168]
    for h in HZ:
        if sp == "test":
            if h <= 48:
                z = np.load(PRED / f"h{h}.npz", allow_pickle=True)
                y, ref = z["y_te"].astype(float), {"ridge": z["te__ridge_best"], "XGBoost S4": z["te__xgb_S4_full_A"], "LSTM L4": z["te__lstm_L4_full_A"]}
            else:
                z = np.load(PRED / f"long_h{h}.npz", allow_pickle=True)
                y, ref = z["y_te"].astype(float), {"ridge": z["ridge"], "XGBoost S4": z["XGBoost_S4"], "LSTM L4": z["LSTM_L4"]}
        else:
            if h <= 48:
                z = np.load(PRED / f"h{h}.npz", allow_pickle=True)
                y_va = None
                ref = {"ridge": z["va__ridge_best"], "XGBoost S4": z["va__xgb_S4_full_A"], "LSTM L4": z["va__lstm_L4_full_A"]}
                y = None
            else:
                continue
        P = {}
        if lg is not None and str(h) in lg.files:
            P["LightGBM S4"] = lg[str(h)]
            yl = lg[str(h) + "_y"].astype(float)
            if y is not None:
                assert np.allclose(yl, y), f"LightGBM samples differ at {h} h"
            y = yl if y is None else y
        for nm, lab in (("gru", "GRU"), ("tcn", "TCN"), ("transformer", "Transformer (patched)")):
            zz = neu[nm]
            if zz is not None and str(h) in zz.files:
                P[lab] = zz[str(h)]
                yn = zz[str(h) + "_y"].astype(float)
                if y is not None:
                    assert np.allclose(yn, y), f"{nm} samples differ at {h} h"
                y = yn if y is None else y
        if not P or y is None:
            continue
        rows[h] = {"y": y, "ref": {k: np.asarray(v, dtype=float) for k, v in ref.items()}, "P": {k: np.asarray(v, dtype=float) for k, v in P.items()}}
    ev_name = "test" if sp == "test" else "validation"
    md = [f"# Other model families on the main horizons: {ev_name} period (generated by `scripts/analyze_completion.py`; completion batch, parts P3)", "",
          "LightGBM (S4, scenario A) at 1, 6, 24, 48, 72 and 168 h; GRU, TCN and a patched Transformer (LSTM L4 inputs and training procedure) at 1 h and 24 h; same samples as the XGBoost and LSTM models of each horizon. "
          "Tuning budgets differ: XGBoost 90 trials per horizon (30 at 72 and 168 h), LSTM 12 (+78 in the equal-budget LSTM) per horizon, LightGBM 30, GRU/TCN/Transformer 12. Differences are model minus reference in MAE (MW), 95% moving-block bootstrap (168-hour blocks); negative = the new model is better; an interval that includes zero is not distinguishable. "
          + ("Single test evaluations with frozen configurations." if sp == "test" else "Validation predictions (optimistic)."), "",
          "| horizon | model | MAE (MW) | vs XGBoost S4 (MW, 95% CI) | vs LSTM L4 (MW, 95% CI) | vs ridge (MW, 95% CI) |", "|---:|---|---:|---|---|---|"]
    res = {}
    for h, r in rows.items():
        for nm in ("ridge", "XGBoost S4", "LSTM L4"):
            md.append(f"| {h} h | {nm} (existing) | {f0(np.mean(np.abs(r['ref'][nm] - r['y'])))} | | | |")
        for nm, p in r["P"].items():
            e = np.abs(p - r["y"])
            cells = []
            for rn in ("XGBoost S4", "LSTM L4", "ridge"):
                d = e - np.abs(r["ref"][rn] - r["y"])
                lo, hi = rb.block_ci(d, rng)
                cells.append(f"{sg(d.mean())} ({sg(lo)} to {sg(hi)})")
                res[f"{h}:{nm}:{rn}"] = (float(d.mean()), float(lo), float(hi))
            md.append(f"| {h} h | **{nm}** | {f0(e.mean())} | " + " | ".join(cells) + " |")
        res[f"{h}:mae"] = {k: float(np.mean(np.abs(v - r["y"]))) for k, v in {**r["ref"], **r["P"]}.items()}
    md += ["", "## Caveats", "",
           "- One test year per horizon; tuning budgets differ (stated above); GRU, TCN and the Transformer were evaluated at 1 h, 24 h and the day-ahead task only, after a cost amendment of the protocol (2026-10-20).",
           "- The TCN pools the hourly input to 3-hour steps and the Transformer uses 6-hour patches (cost reasons); they are not the architectures of the original papers.",
           "- LightGBM uses the same feature set S4 as XGBoost, so differences mostly reflect the learning algorithm and the tuning budget.", ""]
    suffix = "test" if sp == "test" else "validation"
    (ROOT / "reports" / f"other_families_{suffix}.md").write_text(chr(10).join(md), encoding="utf-8")
    (ROOT / "reports" / "tables" / f"other_families_{suffix}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return rows


def figure_b(rows):
    if not rows:
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})
    hs = sorted(rows)
    names = ["ridge", "XGBoost S4", "LSTM L4", "LightGBM S4", "GRU", "TCN", "Transformer (patched)"]
    col = {"ridge": "#0072B2", "XGBoost S4": "#D55E00", "LSTM L4": "#009E73", "LightGBM S4": "#CC79A7", "GRU": "#56B4E9", "TCN": "#E69F00", "Transformer (patched)": "#7F7F7F"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, nm in enumerate(names):
        xs, ys = [], []
        for j, h in enumerate(hs):
            r = rows[h]
            allp = {**r["ref"], **r["P"]}
            if nm in allp:
                xs.append(j)
                ys.append(np.mean(np.abs(allp[nm] - r["y"])))
        if xs:
            ax.plot(xs, ys, "o-", color=col[nm], label=nm)
    ax.set_xticks(range(len(hs)), [f"{h} h" for h in hs])
    ax.set_yscale("log")
    ax.set_ylabel("MAE (MW, log scale)")
    ax.set_title("Other model families vs the existing models")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "fig17_other_families.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    sp = ap.parse_args().split
    md_all = []
    M, res, abl, iv, keys = section_a(sp, md_all)
    rows = section_b(sp)
    if sp == "test":
        figure_a(M, res, abl, iv, keys)
        figure_b(rows)
        print("wrote fig16 and fig17")
    print(md_all[0])
    print("models:", [M[k][0] for k in keys])
    print("done")


if __name__ == "__main__":
    main()
