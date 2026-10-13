"""Phase 5: baseline forecasts evaluated on the VALIDATION period.

Follows docs/experiment_protocol.md. The script loads rows only up to the end of
the validation period (2024-06-30 23:00); the test period is never read.

Models (per horizon h in 1, 6, 24, 48):
  persistence      y(t)
  snaive24         y(t+h-24m), m = ceil(h/24)      (same hour on the latest earlier day that is <= t)
  snaive168        y(t+h-168)                      (same hour last week)
  ridge_g01_abs    ridge on G0 history + G1 calendar, predicts MW
  ridge_g01_rel    same, but target and demand features are divided by the 24-h anchor (D-07 variant)
  ridge_full_abs   + holiday flags, past weather (scenario A), past solar/wind
  ridge_full_rel   relative version of the above
Ridge is implemented with numpy (no extra dependency); features are standardised
with TRAINING statistics; alpha is chosen on validation MAE from ALPHAS.

All models are scored on the SAME valid samples (protocol section 3): target observed
(not missing, not imputed) and every input of every model present.

Outputs: reports/tables/baselines_validation.md and .jsonl (one JSON line per run
with git commit, dataset hash, timings).
Usage:   py scripts/run_baselines.py
"""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "india_hourly_canonical.csv"
OUT_MD = ROOT / "reports" / "tables" / "baselines_validation.md"
OUT_JSONL = ROOT / "reports" / "tables" / "baselines_validation.jsonl"

TRAIN_END = pd.Timestamp("2023-12-31 23:00")
VAL_START, VAL_END = pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-06-30 23:00")
HORIZONS = [1, 6, 24, 48]
LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]
ALPHAS = [0.1, 1, 10, 100, 1000, 1e4, 1e5]
BOOT_B, BOOT_BLOCK, SEED = 1000, 168, 0


def git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return h, dirty
    except Exception:
        return None, None


def build(df: pd.DataFrame, h: int):
    """Feature frames for horizon h, indexed by ORIGIN time t. Target time is t+h."""
    y = df["electricity_demand_mw"]
    obs = y.notna() & ~df["demand_imputed"].astype(bool)
    m24 = math.ceil(h / 24)
    shift24 = 24 * m24 - h                  # >= 0 so the anchor time is <= t
    anchors = {"a24": y.shift(shift24)}
    if h <= 168:
        anchors["a168"] = y.shift(168 - h)
    level = {f"lag{k}": y.shift(k) for k in LAGS}
    r24, r168 = y.rolling(24, min_periods=24), y.rolling(168, min_periods=168)
    level.update({"roll24_mean": r24.mean(), "roll24_std": r24.std(),
                  "roll168_mean": r168.mean(), "roll168_std": r168.std()})
    tgt = df.shift(-h)                      # row t holds the value at target time t+h
    T = df.index + pd.Timedelta(hours=h)
    cal = pd.DataFrame(index=df.index)
    for k in range(24):
        cal[f"hr{k}"] = (T.hour == k).astype(float)
    for k in range(7):
        cal[f"dow{k}"] = (T.dayofweek == k).astype(float)
    cal["month_sin"] = np.sin(2 * np.pi * (T.month - 1) / 12)
    cal["month_cos"] = np.cos(2 * np.pi * (T.month - 1) / 12)
    other = pd.DataFrame(index=df.index)    # G2-G4 (scenario A: weather/generation at <= t only)
    other["is_holiday_T"] = tgt["is_holiday"]
    other["is_major_T"] = tgt["is_major_festival"]
    other["temp_t"] = df["temp_c"]
    other["temp_t_sq"] = df["temp_c"] ** 2
    other["temp_lag24"] = df["temp_c"].shift(24)
    other["temp_roll24"] = df["temp_c"].rolling(24, min_periods=24).mean()
    other["rh_t"] = df["rel_humidity_pct"]
    other["ghi_t"] = df["ghi_wm2"]
    other["solar_t"] = df["solar_generation_mw"]
    other["wind_t"] = df["wind_generation_mw"]
    return {"y": y, "obs": obs, "target": y.shift(-h), "target_obs": obs.shift(-h, fill_value=False),
            "T": pd.Series(T, index=df.index), "anchors": anchors, "level": level, "cal": cal, "other": other,
            "season_T": tgt["season"]}


def design(b, rel: bool, full: bool):
    a24 = b["anchors"]["a24"]
    lev = pd.DataFrame(b["level"])
    anc = pd.DataFrame({"a24": b["anchors"]["a24"], "a168": b["anchors"].get("a168")})
    if rel:
        std_cols = [c for c in lev.columns if c.endswith("std")]
        lev = lev.div(a24, axis=0)
        anc = anc.div(a24, axis=0).drop(columns="a24")
    X = pd.concat([lev, anc, b["cal"]], axis=1)
    if full:
        X = pd.concat([X, b["other"]], axis=1)
    return X.dropna(axis=1, how="all")


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(axis=0), X.std(axis=0).replace(0, 1)
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A.to_numpy(), (Z.T @ (y - ym)).to_numpy())
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def ridge_pred(m, X):
    return ((X - m["mu"]) / m["sd"]).to_numpy() @ m["w"] + m["b"]


def metrics(y, p):
    e = p - y
    return {"MAE_MW": float(np.mean(np.abs(e))), "RMSE_MW": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_pct": float(100 * np.mean(np.abs(e) / y)), "WAPE_pct": float(100 * np.sum(np.abs(e)) / np.sum(y)),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


def block_ci(d, rng):
    """95% moving-block-bootstrap CI of mean(d) for a time-ordered difference series."""
    n = len(d)
    cs = np.concatenate([[0], np.cumsum(d)])
    nb = math.ceil(n / BOOT_BLOCK)
    starts = rng.integers(0, n - BOOT_BLOCK + 1, size=(BOOT_B, nb))
    means = (cs[starts + BOOT_BLOCK] - cs[starts]).sum(1) / (nb * BOOT_BLOCK)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df = pd.read_csv(SRC, parse_dates=["timestamp"]).set_index("timestamp")
    df = df.loc[:VAL_END]                  # the test period is never loaded
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    commit, dirty = git_state()
    rng = np.random.default_rng(SEED)
    train_peak_thr = df.loc[:TRAIN_END, "electricity_demand_mw"].quantile(0.95)
    rows, lines, jl = [], [], []
    for h in HORIZONS:
        b = build(df, h)
        X = {n: design(b, rel, full) for n, (rel, full) in
             {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
        allfeat = pd.concat(X.values(), axis=1)
        need = allfeat.notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna()
        if h <= 168:
            need &= b["anchors"]["a168"].notna()
        valid = need & b["target_obs"]
        T = b["T"]
        in_tr = valid & (T <= TRAIN_END)
        in_va = valid & (T >= VAL_START) & (T <= VAL_END)
        n_tr, n_va = int(in_tr.sum()), int(in_va.sum())
        n_va_all = int(((T >= VAL_START) & (T <= VAL_END)).sum())
        yv = b["target"][in_va]
        preds = {"persistence": b["y"][in_va], "snaive24": b["anchors"]["a24"][in_va],
                 "snaive168": b["anchors"]["a168"][in_va]}
        info = {}
        for name, Xd in X.items():
            rel = name.endswith("rel")
            tgt = (b["target"] / b["anchors"]["a24"]) if rel else b["target"]
            best = None
            t0 = time.perf_counter()
            for a in ALPHAS:
                m = ridge_fit(Xd[in_tr], tgt[in_tr], a)
                p = ridge_pred(m, Xd[in_va])
                p = p * b["anchors"]["a24"][in_va].to_numpy() if rel else p
                mae = float(np.mean(np.abs(p - yv.to_numpy())))
                if best is None or mae < best[0]:
                    best = (mae, a, m, p)
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            _ = ridge_pred(best[2], Xd[in_va])
            inf_s = time.perf_counter() - t1
            preds["ridge_" + name] = pd.Series(best[3], index=yv.index)
            info["ridge_" + name] = {"alpha": best[1], "n_features": Xd.shape[1], "train_and_alpha_search_s": round(fit_s, 3),
                                     "inference_s_for_val_set": round(inf_s, 5)}
        lines.append(f"\n### Horizon {h} h  (train samples {n_tr:,}; validation targets {n_va_all:,}, valid {n_va:,} = "
                     f"{100 * n_va / n_va_all:.1f}%)\n")
        lines.append("| model | MAE (MW) | RMSE (MW) | MAPE % | WAPE % | R2 | skill vs best baseline |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        res = {k: metrics(yv.to_numpy(), v.to_numpy()) for k, v in preds.items()}
        best_base = min(res[k]["MAE_MW"] for k in ("persistence", "snaive24", "snaive168"))
        for k, r in res.items():
            sk = "" if k in ("persistence", "snaive24", "snaive168") else f"{100 * (1 - r['MAE_MW'] / best_base):+.1f}%"
            lines.append(f"| {k} | {r['MAE_MW']:,.0f} | {r['RMSE_MW']:,.0f} | {r['MAPE_pct']:.2f} | {r['WAPE_pct']:.2f} | {r['R2']:.3f} | {sk} |")
            rows.append({"horizon": h, "model": k, **r})
        # D-07 evidence: relative vs absolute, paired block bootstrap on |error| differences
        for tag in ("g01", "full"):
            ea = (preds[f"ridge_{tag}_abs"] - yv).abs().to_numpy()
            er = (preds[f"ridge_{tag}_rel"] - yv).abs().to_numpy()
            d = er - ea
            lo, hi = block_ci(d, rng)
            lines.append(f"\nridge_{tag}: mean(|err| rel - |err| abs) = {d.mean():+.0f} MW (95% block-bootstrap CI {lo:+.0f} to {hi:+.0f}); negative = relative target better.")
            rows.append({"horizon": h, "model": f"diff_rel_minus_abs_{tag}", "mean_diff_MW": float(d.mean()), "ci_lo": lo, "ci_hi": hi})
        # seasons and peak hours for the best ridge variant
        bestk = min((k for k in preds if k.startswith("ridge")), key=lambda k: res[k]["MAE_MW"])
        s = b["season_T"][in_va]
        lines.append(f"\nBest ridge on validation: **{bestk}** (alpha {info[bestk]['alpha']}). MAE by season of the target hour: " +
                     "; ".join(f"{sn}: {np.mean(np.abs(preds[bestk][s == sn] - yv[s == sn])):,.0f} MW (n={int((s == sn).sum()):,})"
                     for sn in ["winter", "pre_monsoon_summer", "monsoon", "post_monsoon"] if (s == sn).any()))
        pk = yv >= train_peak_thr
        lines.append(f"Peak hours (target >= 95th percentile of TRAINING demand, {train_peak_thr / 1000:.0f} GW; n={int(pk.sum())}): "
                     f"MAE {np.mean(np.abs(preds[bestk][pk] - yv[pk])):,.0f} MW vs {np.mean(np.abs(preds['snaive168'][pk] - yv[pk])):,.0f} MW for snaive168.")
        jl.append({"run_date": pd.Timestamp.now().isoformat(timespec="seconds"), "stage": "baselines_validation", "horizon": h,
                   "git_commit": commit, "git_dirty": dirty, "dataset_sha256": sha, "dataset_rows_loaded": len(df),
                   "split": {"train_target_end": str(TRAIN_END), "val": [str(VAL_START), str(VAL_END)]},
                   "n_train": n_tr, "n_val_valid": n_va, "seed": SEED, "ridge": info, "metrics": res})
    # D-07 summary by the pre-specified rule: mean validation MAE over horizons, ridge_g01 abs vs rel
    mm = pd.DataFrame([r for r in rows if "MAE_MW" in r]).pivot(index="horizon", columns="model", values="MAE_MW")
    summ = ["\n## D-07 decision (pre-specified rule: lower mean validation MAE across the four horizons, ridge on the same samples)\n",
            "| formulation | mean MAE over horizons (MW) |", "|---|---:|"]
    for tag in ("g01", "full"):
        for r in ("abs", "rel"):
            summ.append(f"| ridge_{tag}_{r} | {mm[f'ridge_{tag}_{r}'].mean():,.0f} |")
    header = ["# Baselines on the validation period (generated by `scripts/run_baselines.py`)", "",
              f"Dataset sha256 `{sha[:16]}...`, git commit `{(commit or 'n/a')[:10]}` (working tree dirty: {dirty}). "
              f"Validation targets {VAL_START:%Y-%m-%d} to {VAL_END:%Y-%m-%d}. The test period was not loaded. "
              "All models use the same valid samples per horizon. Scenario A (no future weather).", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(header + lines + summ) + "\n", encoding="utf-8")
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for r in jl:
            f.write(json.dumps(r) + "\n")
    print("\n".join(lines + summ))


if __name__ == "__main__":
    main()
