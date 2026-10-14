"""Phase 7: LSTM on the VALIDATION period (test period is never loaded).

Follows docs/experiment_protocol.md (and its deviation log for the reduced tuning budget).
  * one direct model per horizon h in {1, 6, 24, 48}, ratio-to-24h-anchor target (D-07),
    trained as z = 100 * (ratio - 1) / sd_train(z)
  * input: the last 168 hourly steps ending at the origin t. Per step: demand / anchor, the national weather
    (7 variables) and 5 regional temperatures, solar and wind generation (all at times <= t), and sin/cos of the
    step's hour and weekday. Level channels are standardised with TRAINING statistics. Solar/wind gaps inside a window
    are forward-filled from the past (3 short dropouts only).
  * after the recurrent layer, deterministic information about the TARGET time (hour, weekday, month, holiday and
    festival-window features, last week's anchor relative to yesterday's) is concatenated and passed to a small MLP
  * same valid samples as the baselines and XGBoost (counts are asserted equal to the XGBoost log)
  * step 1: random search (N_TRIALS per horizon, <= 10 epochs, patience 3) on the full set L4; step 2: 3 seeds with the
    best configuration for the feature sets L1 (history + calendar), L2 (+ holidays), L4 (full, scenario A)
Outputs: reports/tables/lstm_validation.md and .jsonl.   Usage: py scripts/run_lstm.py [--smoke]
"""
import argparse
import copy
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")
rx = importlib.import_module("run_xgboost")

OUT_MD = ROOT / "reports" / "tables" / "lstm_validation.md"
OUT_JSONL = ROOT / "reports" / "tables" / "lstm_validation.jsonl"
XGB_JSONL = ROOT / "reports" / "tables" / "xgboost_validation.jsonl"
L = 168
N_TRIALS, TUNE_EPOCHS, TUNE_PATIENCE = 12, 10, 3
FINAL_EPOCHS, FINAL_PATIENCE, SEEDS, BATCH = 30, 5, [0, 1, 2], 256
torch.set_num_threads(4)
WEATHER = ft.WEATHER + ft.REGIONAL_T


class Net(nn.Module):
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, s):
        o, _ = self.lstm(x)
        return self.head(torch.cat([o[:, -1], s], dim=1)).squeeze(1)


def step_channels(df, use_weather, use_gen, train_end):
    """Per-hour channels that do not depend on the sample (everything except demand/anchor)."""
    cols = []
    hr, dw = df.index.hour.to_numpy(), df.index.dayofweek.to_numpy()
    cols += [np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)]
    tr = df.index <= train_end
    def z(series):
        v = series.ffill().to_numpy(dtype=float)       # past-only fill
        return (v - np.nanmean(v[tr])) / (np.nanstd(v[tr]) + 1e-9)
    if use_weather:
        cols += [z(df[c]) for c in WEATHER]
    if use_gen:
        cols += [z(df["solar_generation_mw"]), z(df["wind_generation_mw"])]
    return np.stack(cols, axis=1).astype(np.float32)


def static_features(df, h, b, use_holidays):
    X = ft.build_features(df, h, ["g1"] + (["g2"] if use_holidays else []), True)
    T = df.index + pd.Timedelta(hours=h)
    cols = [np.sin(2 * np.pi * T.hour / 24), np.cos(2 * np.pi * T.hour / 24), np.sin(2 * np.pi * T.dayofweek / 7),
            np.cos(2 * np.pi * T.dayofweek / 7), np.sin(2 * np.pi * (T.month - 1) / 12), np.cos(2 * np.pi * (T.month - 1) / 12),
            (T.dayofweek >= 5).astype(float)]
    if use_holidays:
        cols += [X["is_holiday_T"].to_numpy(), X["is_major_T"].to_numpy()]
        cols += [X[c].fillna(30).to_numpy() / 30.0 for c in ("d_next_major", "d_since_major", "d_next_hol", "d_since_hol")]
    a168 = (b["anchors"]["a168"] / b["anchors"]["a24"] - 1).fillna(0).to_numpy() / 0.2
    cols.append(a168)
    return np.stack(cols, axis=1).astype(np.float32)


class Data:
    """Gathers windows on the fly: windows are never copied for the whole dataset."""
    def __init__(self, y, a24, C, S, zt, idx):
        self.y, self.a24, self.C, self.S, self.zt, self.idx = y, a24, C, S, zt, idx
        self.off = np.arange(-(L - 1), 1)

    def batch(self, rows):
        p = self.idx[rows]
        w = p[:, None] + self.off[None, :]
        dem = ((self.y[w] / self.a24[p][:, None]) - 1.0) / 0.2
        x = np.concatenate([dem[:, :, None].astype(np.float32), self.C[w]], axis=2)
        return torch.from_numpy(x), torch.from_numpy(self.S[p]), torch.from_numpy(self.zt[p].astype(np.float32))


def predict(net, data, n, bs=1024):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            x, s, _ = data.batch(np.arange(i, min(i + bs, n)))
            out.append(net(x, s).numpy())
    return np.concatenate(out)


def train(tr, va, n_in, n_static, hp, seed, epochs, patience, zsd, a_va, y_va):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(n_in, n_static, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, best_state, bad, hist = np.inf, None, 0, []
    ntr, nva = len(tr.idx), len(va.idx)
    proc = psutil.Process()
    t0, c0 = time.perf_counter(), time.process_time()
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(ntr)
        for i in range(0, ntr, BATCH):
            x, s, z = tr.batch(perm[i:i + BATCH])
            loss = ((net(x, s) - z) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        zhat = predict(net, va, nva)
        pred = (1 + zhat * zsd / 100.0) * a_va
        mae = float(np.mean(np.abs(pred - y_va)))
        hist.append(mae)
        if mae < best - 1e-6:
            best, best_state, bad = mae, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    net.load_state_dict(best_state)
    peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
    return net, best, hist, wall, cpu, peak


def main():
    global N_TRIALS, TUNE_EPOCHS, FINAL_EPOCHS, SEEDS, OUT_MD, OUT_JSONL, XGB_JSONL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage2", action="store_true",
                    help="pre-test stage 2: tune the history+calendar variant (L1) on its own, final fits for L1 only, "
                         "compare with the stage-2 XGBoost log (xgboost_validation_wide.jsonl)")
    args = ap.parse_args()
    horizons = rb.HORIZONS
    tune_set, tune_seed = "L4_full_A", 2000
    if args.stage2:
        tune_set, tune_seed = "L1_history_calendar", 4000
        XGB_JSONL = XGB_JSONL.with_name("xgboost_validation_wide.jsonl")
        OUT_MD, OUT_JSONL = OUT_MD.with_name("lstm_validation_stage2_L1.md"), OUT_JSONL.with_name("lstm_validation_stage2_L1.jsonl")
        rx.MAX_TREES = 3000            # the stage-2 XGBoost configurations may need more trees
    if args.smoke:
        N_TRIALS, TUNE_EPOCHS, FINAL_EPOCHS, SEEDS, horizons = 2, 2, 2, [0], [24]
        OUT_MD, OUT_JSONL = OUT_MD.with_name("lstm_validation_smoke.md"), OUT_JSONL.with_name("lstm_validation_smoke.jsonl")
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:rb.VAL_END]
    sha = hashlib.sha256(rb.SRC.read_bytes()).hexdigest()
    commit, dirty = rb.git_state()
    meta = {"git_commit": commit, "git_dirty": dirty, "dataset_sha256": sha, "torch": torch.__version__, "python": platform.python_version(),
            "os": platform.platform(), "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), "threads": torch.get_num_threads(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "gpu": "none used (CPU only)"}
    xg_counts = {}
    for line in open(XGB_JSONL, encoding="utf-8"):
        r = json.loads(line)
        if r["stage"] == "ablation_fit":
            xg_counts[r["horizon"]] = (r["n_train"], r["n_val"])
    xg_params = {}
    for line in open(XGB_JSONL, encoding="utf-8"):
        r = json.loads(line)
        if r["stage"] == "ablation_fit" and r["feature_set"] == rx.HEADLINE and r["relative_target"] and r["seed"] == 0:
            xg_params[r["horizon"]] = r["params"]
    log = open(OUT_JSONL, "a", encoding="utf-8")
    y = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)       # only used inside valid windows
    md = ["# LSTM on the validation period (generated by `scripts/run_lstm.py`)", "",
          f"Dataset sha256 `{sha[:16]}...`, git commit `{(commit or 'n/a')[:10]}` (working tree dirty: {dirty}); torch {torch.__version__}; "
          f"{meta['threads']} threads, CPU only. Validation targets 2024-01-01 to 2024-06-30; the test period was not loaded. Scenario A. "
          f"Tuning: {N_TRIALS} random trials per horizon (XGBoost used 30; see protocol deviation log) on feature set " + ("L1 (stage 2: own tuning of the history+calendar variant; compared with the stage-2 XGBoost log)" if args.stage2 else "L4") + "; early stopping on validation, "
          "so the numbers are mildly optimistic. Headline = mean prediction of 3 seeds (same convention as XGBoost).", ""]
    for h in horizons:
        b = rb.build(df, h)
        Xs = {n: rb.design(b, rel, full) for n, (rel, full) in
              {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}.items()}
        need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna()
        if h <= 168:
            need &= b["anchors"]["a168"].notna()
        valid = need & b["target_obs"]
        in_tr = valid & (b["T"] <= rb.TRAIN_END)
        in_va = valid & (b["T"] >= rb.VAL_START) & (b["T"] <= rb.VAL_END)
        a24 = b["anchors"]["a24"]
        ntr, nva = int(in_tr.sum()), int(in_va.sum())
        assert (ntr, nva) == xg_counts[h], f"sample sets differ from the XGBoost run: {(ntr, nva)} vs {xg_counts[h]}"
        pos = np.arange(len(df))
        idx_tr, idx_va = pos[in_tr.to_numpy()], pos[in_va.to_numpy()]
        assert idx_tr.min() >= L - 1 and idx_va.min() >= L - 1, "window would start before the data"
        z_all = ((b["target"] / a24 - 1) * 100).to_numpy()
        zsd = float(np.nanstd(z_all[idx_tr]))
        zt = np.nan_to_num(z_all / zsd)
        a24v = a24.fillna(1.0).to_numpy()
        y_va = b["target"].to_numpy()[idx_va]
        a_va = a24v[idx_va]
        # feature-set definitions
        FS = {"L1_history_calendar": dict(w=False, g=False, hol=False), "L2_holidays": dict(w=False, g=False, hol=True),
              "L4_full_A": dict(w=True, g=True, hol=True)}
        built = {}
        for name, c in FS.items():
            if args.stage2 and name != tune_set:
                continue
            C = step_channels(df, c["w"], c["g"], rb.TRAIN_END)
            S = static_features(df, h, b, c["hol"])
            built[name] = (Data(y, a24v, C, S, zt, idx_tr), Data(y, a24v, C, S, zt, idx_va), C.shape[1] + 1, S.shape[1])
        # ---- step 1: tuning on the declared set (L4 in stage 1, L1 in stage 2)
        tr, va, n_in, n_st = built[tune_set]
        rng = np.random.default_rng(tune_seed + h)
        best, t_search = None, time.perf_counter()
        for trial in range(N_TRIALS):
            hp = {"hidden": int(rng.choice([16, 32, 48, 64])), "lr": float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
                  "dropout": float(rng.uniform(0, 0.3)), "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3))))}
            net, mae, hist, wall, cpu, peak = train(tr, va, n_in, n_st, hp, 0, TUNE_EPOCHS, TUNE_PATIENCE, zsd, a_va, y_va)
            log.write(json.dumps({"stage": "tuning_trial", "horizon": h, "trial": trial, "params": hp, "val_MAE_MW": mae,
                                  "epochs_run": len(hist), "fit_s": round(wall, 1)}) + "\n"); log.flush()
            print(f"  h={h} trial {trial}: hidden={hp['hidden']} lr={hp['lr']:.4f} -> MAE {mae:.0f} ({len(hist)} ep, {wall:.0f}s)", flush=True)
            if best is None or mae < best[0]:
                best = (mae, hp, trial)
        search_s = time.perf_counter() - t_search
        hp = best[1]
        # ---- step 2: 3 seeds per feature set
        rows = {}
        for name, (tr, va, n_in, n_st) in built.items():
            maes, preds, extra = [], [], None
            for seed in SEEDS:
                net, mae, hist, wall, cpu, peak = train(tr, va, n_in, n_st, hp, seed, FINAL_EPOCHS, FINAL_PATIENCE, zsd, a_va, y_va)
                zhat = predict(net, va, nva)
                preds.append((1 + zhat * zsd / 100.0) * a_va)
                maes.append(float(np.mean(np.abs(preds[-1] - y_va))))
                lat = None
                if seed == SEEDS[0]:
                    net.eval()
                    x, s, _ = va.batch(np.arange(min(1024, nva)))
                    with torch.no_grad():
                        net(x[:8], s[:8])
                        bt, sg = [], []
                        for _ in range(10):
                            t = time.perf_counter(); net(x, s); bt.append(time.perf_counter() - t)
                            t = time.perf_counter(); net(x[:1], s[:1]); sg.append(time.perf_counter() - t)
                    lat = {"batch_rows": int(x.shape[0]), "batch_ms_median": 1000 * float(np.median(bt)),
                           "single_window_ms_median": 1000 * float(np.median(sg))}
                nparams = sum(p.numel() for p in net.parameters())
                torch.save(net.state_dict(), ROOT / "data" / "interim" / "_tmp_lstm.pt")
                size = (ROOT / "data" / "interim" / "_tmp_lstm.pt").stat().st_size
                log.write(json.dumps({"stage": "final_fit", "horizon": h, "feature_set": name, "seed": seed, "n_train": ntr, "n_val": nva,
                                      "val_MAE_MW": maes[-1], "epochs_run": len(hist), "best_epoch": int(np.argmin(hist)) + 1,
                                      "train_wall_s": round(wall, 1), "train_cpu_s": round(cpu, 1), "peak_working_set_mb": round(peak / 1e6, 1),
                                      "n_parameters": nparams, "model_bytes": size, "inference": lat, "params": hp, **meta}) + "\n"); log.flush()
            rows[name] = {"mae_mean": float(np.mean(maes)), "mae_sd": float(np.std(maes)), "pred": np.mean(preds, axis=0)}
            print(f"  h={h} {name}: MAE {np.mean(maes):.0f} +- {np.std(maes):.0f}; mean-of-seeds MAE {np.mean(np.abs(rows[name]['pred'] - y_va)):.0f}", flush=True)
        # ---- comparisons on the same samples: best ridge, XGBoost headline refit
        a24_va = a24v[idx_va]
        ridge_best = None
        for rname, Xd in {"ridge_full_rel": Xs["full_rel"], "ridge_g01_rel": Xs["g01_rel"], "ridge_full_abs": Xs["full_abs"], "ridge_g01_abs": Xs["g01_abs"]}.items():
            rel = rname.endswith("rel")
            tgt = (b["target"] / a24) if rel else b["target"]
            for al in rb.ALPHAS:
                m = rb.ridge_fit(Xd[in_tr], tgt[in_tr], al)
                p = rb.ridge_pred(m, Xd[in_va]); p = p * a24_va if rel else p
                mae = float(np.mean(np.abs(p - y_va)))
                if ridge_best is None or mae < ridge_best[0]:
                    ridge_best = (mae, rname, p)
        Xh = ft.build_features(df, h, ft.FEATURE_SETS[rx.HEADLINE], True)
        zx = (b["target"] / a24)
        xp = np.mean([rx.fit_predict(Xh[in_tr], zx[in_tr], Xh[in_va], zx[in_va], xg_params[h], s, a24_va, True)[0] for s in (0, 1, 2)], axis=0)
        cmp = {}
        for name in rows:
            d_r = np.abs(rows[name]["pred"] - y_va) - np.abs(ridge_best[2] - y_va)
            d_x = np.abs(rows[name]["pred"] - y_va) - np.abs(xp - y_va)
            cmp[name] = {"mae_ens": float(np.mean(np.abs(rows[name]["pred"] - y_va))), "vs_ridge": (float(d_r.mean()), *rb.block_ci(d_r, np.random.default_rng(0))),
                         "vs_xgb": (float(d_x.mean()), *rb.block_ci(d_x, np.random.default_rng(0)))}
        md += [f"## Horizon {h} h (train samples {ntr:,}; validation samples {nva:,})", "",
               f"Tuning: {N_TRIALS} trials in {search_s / 60:.1f} min; best trial #{best[2]}: hidden={hp['hidden']}, lr={hp['lr']:.4g}, dropout={hp['dropout']:.2f}, weight_decay={hp['weight_decay']:.2g}.", "",
               "| feature set | MAE per seed, mean +- sd (MW) | MAE of 3-seed mean prediction | vs best ridge (CI) | vs XGBoost headline (CI) |", "|---|---:|---:|---|---|"]
        for name, r in rows.items():
            c = cmp[name]
            md.append(f"| {name} | {r['mae_mean']:,.0f} +- {r['mae_sd']:,.0f} | {c['mae_ens']:,.0f} | {c['vs_ridge'][0]:+,.0f} ({c['vs_ridge'][1]:+,.0f} to {c['vs_ridge'][2]:+,.0f}) | "
                      f"{c['vs_xgb'][0]:+,.0f} ({c['vs_xgb'][1]:+,.0f} to {c['vs_xgb'][2]:+,.0f}) |")
        md += ["", f"Reference on the same samples: best ridge ({ridge_best[1]}) {ridge_best[0]:,.0f} MW; XGBoost headline (3-seed mean) {np.mean(np.abs(xp - y_va)):,.0f} MW. "
               "Negative differences mean the LSTM is better (paired MAE, 95% moving-block bootstrap).", ""]
        print(f"h={h} done: {tune_set} mean-of-seeds MAE {cmp[tune_set]['mae_ens']:.0f}; XGBoost {np.mean(np.abs(xp - y_va)):.0f}; ridge {ridge_best[0]:.0f}", flush=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.close()
    tmp = ROOT / "data" / "interim" / "_tmp_lstm.pt"
    if tmp.exists():
        tmp.unlink()
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
