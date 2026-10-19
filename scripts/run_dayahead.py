"""Task 9.7: operational day-ahead model (design in docs/experiment_protocol.md, deviation log 2026-10-19).

Issue time 10:00 on day D-1 (origin t0 = D-1 10:00); the 24 hourly values of day D are forecast (leads h = 14..37 h). Definitions, features, anchors and the valid-sample rule of the main
study are reused for each lead and restricted to origins at 10:00. Models: naive (same hour on the latest earlier day not after the origin / same hour last week), ridge (one model per
target hour), XGBoost S4 and LSTM L4 (each one model pooled over the 24 target hours).
  py scripts/run_dayahead.py               PHASE 1, validation only (test not loaded): ridge variant/penalty, XGBoost (30 random trials), LSTM (30 random trials + the stage-3 24 h
                                           configuration), 3 seeds; writes reports/tables/dayahead_config.json, reports/dayahead_validation.md, data/interim/predictions/dayahead_val.npz
  py scripts/run_dayahead.py --with-test   PHASE 2, single test evaluation with the frozen configuration (training period only, trees/epochs fixed from validation, 3 seeds);
                                           lock reports/tables/DAYAHEAD_TEST_DONE.lock; writes data/interim/predictions/dayahead.npz (analysis: scripts/analyze_dayahead.py)
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
rb = importlib.import_module("run_baselines")
ft = importlib.import_module("features_tabular")
rx = importlib.import_module("run_xgboost")
rl = importlib.import_module("run_lstm")
rt = importlib.import_module("run_test_evaluation")
r3 = importlib.import_module("run_lstm_stage3")
T = ROOT / "reports" / "tables"
CONFIG, LOCK = T / "dayahead_config.json", T / "DAYAHEAD_TEST_DONE.lock"
PRED = ROOT / "data" / "interim" / "predictions"
ORIGIN_HOUR, LEADS, SEEDS, N_XGB, N_LSTM = 10, list(range(14, 38)), [0, 1, 2], 30, 30
VARIANTS = {"g01_abs": (False, False), "g01_rel": (True, False), "full_abs": (False, True), "full_rel": (True, True)}


class Pooled:
    """Windows gathered on the fly for rows that pool several leads: per-row anchor, static features and target."""
    def __init__(self, y, C, p, a24r, S, zt):
        self.y, self.C, self.p, self.a24r, self.S, self.zt = y, C, p, a24r, S, zt
        self.idx = np.arange(len(p))
        self.off = np.arange(-(rl.L - 1), 1)

    def batch(self, rows):
        p = self.p[rows]
        w = p[:, None] + self.off[None, :]
        dem = ((self.y[w] / self.a24r[rows][:, None]) - 1.0) / 0.2
        x = np.concatenate([dem[:, :, None].astype(np.float32), self.C[w]], axis=2)
        return torch.from_numpy(x), torch.from_numpy(self.S[rows]), torch.from_numpy(self.zt[rows].astype(np.float32))


def collect(df, leads, with_test):
    """Per lead: rows (origin positions at 10:00) of the training, validation and (optionally) test samples with all inputs."""
    out = {}
    pos_all = np.arange(len(df))
    at10 = (df.index.hour == ORIGIN_HOUR)
    for h in leads:
        b = rb.build(df, h)
        Xs = {n: rb.design(b, rel, full) for n, (rel, full) in VARIANTS.items()}
        need = pd.concat(Xs.values(), axis=1).notna().all(axis=1) & b["y"].notna() & b["anchors"]["a24"].notna() & b["anchors"]["a168"].notna()
        valid = need & b["target_obs"] & at10
        Tt = b["T"]
        masks = {"tr": valid & (Tt <= rb.TRAIN_END), "va": valid & (Tt >= rb.VAL_START) & (Tt <= rb.VAL_END)}
        if with_test:
            masks["te"] = valid & (Tt >= rt.TEST_START) & (Tt <= rt.TEST_END)
        Xf = ft.build_features(df, h, ft.FEATURE_SETS["S4_full_A"], True)
        Xf["lead"] = float(h)
        a24 = b["anchors"]["a24"]
        a24v = a24.fillna(1.0).to_numpy()
        S = rl.static_features(df, h, b, True)
        out[h] = {"b": b, "Xs": Xs, "Xf": Xf, "a24": a24, "a24v": a24v, "S": S, "pos": {k: pos_all[m.to_numpy()] for k, m in masks.items()},
                  "zx": (b["target"] / a24), "target": b["target"].to_numpy(), "T": Tt}
    return out


def main():
    global LEADS_USED, N_XGB, N_LSTM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rx.MAX_TREES = 3000
    commit, dirty = rb.git_state()
    LEADS_USED = LEADS
    if args.smoke:
        LEADS_USED, N_XGB, N_LSTM = [14, 26, 37], 1, 1
    if args.with_test:
        if LOCK.exists():
            sys.exit("The day-ahead test evaluation was already run.")
        if dirty:
            sys.exit("Working tree is not clean: commit first.")
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = pd.read_csv(rb.SRC, parse_dates=["timestamp"]).set_index("timestamp").loc[:(rt.TEST_END if args.with_test else rb.VAL_END)]
    y_all = df["electricity_demand_mw"].ffill().to_numpy(dtype=float)
    D = collect(df, LEADS_USED, args.with_test)
    splits = ["tr", "va"] + (["te"] if args.with_test else [])
    n = {s: sum(len(D[h]["pos"][s]) for h in LEADS_USED) for s in splits}
    print("samples:", n, flush=True)

    def rows(split):
        """Stacked arrays in lead-major order: lead, origin position, anchors, actual target, target time, XGBoost features, static LSTM features."""
        lead = np.concatenate([np.full(len(D[h]["pos"][split]), h) for h in LEADS_USED])
        pos = np.concatenate([D[h]["pos"][split] for h in LEADS_USED])
        g = lambda f: np.concatenate([f(D[h], D[h]["pos"][split]) for h in LEADS_USED])
        a168 = g(lambda d, p: d["b"]["anchors"]["a168"].to_numpy()[p])
        return {"lead": lead, "pos": pos, "a24": g(lambda d, p: d["a24v"][p]), "a168": a168, "y": g(lambda d, p: d["target"][p]), "T": g(lambda d, p: d["T"].to_numpy()[p].astype("datetime64[ns]").astype("int64")),
                "zx": g(lambda d, p: d["zx"].to_numpy()[p]), "S": np.concatenate([D[h]["S"][D[h]["pos"][split]] for h in LEADS_USED]),
                "Xf": pd.concat([D[h]["Xf"].iloc[D[h]["pos"][split]] for h in LEADS_USED]), "season": g(lambda d, p: d["b"]["season_T"].to_numpy()[p]).astype(str)}

    R = {s: rows(s) for s in splits}
    cols = list(R["tr"]["Xf"].columns)
    for s in splits:
        assert list(R[s]["Xf"].columns) == cols
    z_tr = (R["tr"]["zx"] - 1.0) * 100
    zsd = float(np.nanstd(z_tr))
    C = rl.step_channels(df, True, True, rb.TRAIN_END)
    n_in = C.shape[1] + 1
    n_st = R["tr"]["S"].shape[1]
    mkP = lambda s: Pooled(y_all, C, R[s]["pos"], R[s]["a24"], R[s]["S"], np.nan_to_num(((R[s]["zx"] - 1.0) * 100) / zsd))
    md = ["# " + ("Day-ahead model (issue 10:00 on D-1): single test evaluation" if args.with_test else "Day-ahead model (issue 10:00 on D-1): validation") + " (generated by `scripts/run_dayahead.py`)", "",
          f"Git commit `{(commit or 'n/a')[:10]}`. Leads 14-37 h from the 10:00 origin; scenario A, same (day, hour) samples for all models. Samples: training {n['tr']:,}, validation {n['va']:,}" + (f", test {n['te']:,}" if args.with_test else "")
          + ". MAE in MW. " + ("Models fitted on the training period only with the validation-chosen configuration." if args.with_test else "Hyper-parameters chosen on validation; early stopping on validation (mildly optimistic)."), ""]
    ev = "te" if args.with_test else "va"
    E = R[ev]
    t0 = time.perf_counter()

    # ---------------- ridge: one model per target hour (= per lead)
    def ridge_run(vname, al, fit_split, pred_split):
        rel = vname.endswith("rel")
        out = []
        for h in LEADS_USED:
            d = D[h]
            Xd = d["Xs"][vname]
            ptr, pp = d["pos"][fit_split], d["pos"][pred_split]
            tgt = (d["zx"] if rel else pd.Series(d["target"], index=d["zx"].index))
            m = rb.ridge_fit(Xd.iloc[ptr], tgt.iloc[ptr], al)
            p = rb.ridge_pred(m, Xd.iloc[pp])
            out.append(p * d["a24v"][pp] if rel else p)
        return np.concatenate(out)

    if not args.with_test:
        best = None
        for vname in ("full_rel", "g01_rel", "full_abs", "g01_abs"):
            for al in rb.ALPHAS:
                mae = float(np.mean(np.abs(ridge_run(vname, al, "tr", "va") - E["y"])))
                if best is None or mae < best[0]:
                    best = (mae, vname, al)
        ridge_cfg = [best[1], best[2]]
        print("ridge", best, flush=True)
    else:
        ridge_cfg = cfg["ridge"]
    p_ridge = ridge_run(ridge_cfg[0], ridge_cfg[1], "tr", ev)

    # ---------------- XGBoost S4, pooled over the 24 target hours
    Xtr, Xe = R["tr"]["Xf"], R["va"]["Xf"] if not args.with_test else R["te"]["Xf"]
    ztr = R["tr"]["zx"]
    if not args.with_test:
        r_ = np.random.default_rng(8500)
        xb = None
        for _ in range(N_XGB):
            prm = rx.sample_params(r_)
            p, *_ = rx.fit_predict(Xtr, ztr, R["va"]["Xf"], R["va"]["zx"], prm, 0, R["va"]["a24"], True)
            mae = float(np.mean(np.abs(p - R["va"]["y"])))
            if xb is None or mae < xb[0]:
                xb = (mae, prm)
        trees, xp = {}, []
        for sd in SEEDS:
            p, m, *_ = rx.fit_predict(Xtr, ztr, R["va"]["Xf"], R["va"]["zx"], xb[1], sd, R["va"]["a24"], True)
            trees[sd] = int(m.best_iteration) + 1
            xp.append(p)
        xgb_cfg = {"params": xb[1], "trees": {str(k): v for k, v in trees.items()}}
        print("xgb", xb[0], trees, flush=True)
    else:
        xgb_cfg, xp = cfg["xgb"], []
        for sd in SEEDS:
            mdl = xgb.XGBRegressor(n_estimators=xgb_cfg["trees"][str(sd)], objective="reg:squarederror", tree_method="hist", device="cpu", n_jobs=rx.NJOBS, random_state=sd, eval_metric="mae", **xgb_cfg["params"])
            mdl.fit(Xtr, (ztr - 1.0) * 100.0, verbose=False)
            xp.append((1.0 + mdl.predict(Xe) / 100.0) * E["a24"])
    p_xgb = np.mean(xp, axis=0)

    # ---------------- LSTM L4, pooled over the 24 target hours
    trD, eD = mkP("tr"), mkP(ev)
    if not args.with_test:
        vaD = eD
        cands, r2 = [], np.random.default_rng(9500)
        orig = json.loads((T / "lstm_stage3_config.json").read_text(encoding="utf-8"))["24"]["params"]
        mae0, *_ = r3.trial_fit(trD, vaD, n_in, n_st, orig, zsd, E["a24"], E["y"], None)
        cands.append((mae0, orig, "stage-3 24 h configuration"))
        for k in range(N_LSTM):
            hp = r3.sample_hp(r2)
            mae, *_ = r3.trial_fit(trD, vaD, n_in, n_st, hp, zsd, E["a24"], E["y"], None)
            cands.append((mae, hp, f"new#{k}"))
        sel = min(cands, key=lambda x: x[0])
        epochs, lp = {}, []
        for sd in SEEDS:
            net, mae, hist, *_ = rl.train(trD, vaD, n_in, n_st, sel[1], sd, 30, 5, zsd, E["a24"], E["y"])
            epochs[sd] = int(np.argmin(hist)) + 1
            lp.append((1 + rl.predict(net, vaD, len(E["y"])) * zsd / 100.0) * E["a24"])
        lstm_cfg = {"params": sel[1], "epochs": {str(k): v for k, v in epochs.items()}, "selected_from": sel[2], "zsd": zsd}
        print("lstm", sel[0], sel[2], epochs, flush=True)
    else:
        lstm_cfg, lp = cfg["lstm"], []
        for sd in SEEDS:
            net = rt.lstm_fit_fixed(trD, n_in, n_st, lstm_cfg["params"], sd, lstm_cfg["epochs"][str(sd)])
            lp.append((1 + rl.predict(net, eD, len(E["y"])) * zsd / 100.0) * E["a24"])
    p_lstm = np.mean(lp, axis=0)

    P = {"naive_a24": E["a24"], "naive_a168": E["a168"], "ridge": p_ridge, "xgb": p_xgb, "lstm": p_lstm}
    mae = {k: float(np.mean(np.abs(v - E["y"]))) for k, v in P.items()}
    names = {"naive_a24": "naive: same hour on the latest earlier day", "naive_a168": "naive: same hour last week", "ridge": f"ridge ({ridge_cfg[0]}, alpha {ridge_cfg[1]})", "xgb": "XGBoost S4 (pooled)", "lstm": "LSTM L4 (pooled)"}
    md += ["| model | " + ("test" if args.with_test else "validation") + " MAE (MW) | MAPE % |", "|---|---:|---:|"] + [f"| {names[k]} | {mae[k]:,.0f} | {100 * np.mean(np.abs(v - E['y']) / E['y']):.2f} |" for k, v in P.items()]
    if not args.with_test:
        md += ["", f"Selected: ridge {ridge_cfg[0]} alpha {ridge_cfg[1]}; XGBoost best of {N_XGB} trials (trees {list(trees.values())}); LSTM {sel[2]} (hidden {sel[1]['hidden']}, lr {sel[1]['lr']:.4g}, dropout {sel[1]['dropout']:.2f}, "
               f"weight decay {sel[1]['weight_decay']:.2g}), best epochs {list(epochs.values())}; the stage-3 24 h configuration scored {mae0:,.0f} MW under the same trial procedure.", ""]
    md += ["", f"Run time {time.perf_counter() - t0:,.0f} s.", ""]
    np.savez_compressed(PRED / ("dayahead.npz" if args.with_test else "dayahead_val.npz"), y=E["y"], lead=E["lead"], T=E["T"], pos=E["pos"], season=E["season"], **P,
                        xgb_perseed=np.array(xp), lstm_perseed=np.array(lp)) if not args.smoke else None
    print({k: round(v) for k, v in mae.items()}, flush=True)
    if args.smoke:
        print("smoke finished (nothing written)")
        return
    if args.with_test:
        (ROOT / "reports" / "dayahead_test.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
        LOCK.write_text(f"day-ahead test run completed at commit {commit}" + chr(10), encoding="utf-8")
    else:
        CONFIG.write_text(json.dumps({"ridge": ridge_cfg, "xgb": xgb_cfg, "lstm": lstm_cfg}, indent=1), encoding="utf-8")
        (ROOT / "reports" / "dayahead_validation.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
