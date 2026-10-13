# Experiment protocol (frozen 2026-10-13)

This document fixes *how* every forecasting experiment is run, before any model is trained. Changing anything below after results exist is allowed only as a logged **post-hoc deviation** (section 13) and must be reported as such. Source of truth for decisions: `docs/project_plan.md` (decision log). Data: `data/processed/india_hourly_canonical.csv` (columns in `reports/data_dictionary.md`).

## 1. Task definition (decision D-06)
- One forecast origin per clock hour `t` (IST). At origin `t` the model may use only information with timestamp ≤ `t` (plus deterministic calendar information about the future).
- Target: `electricity_demand_mw` at `t + h` (single value, a **direct** forecast; one model per horizon).
- Horizons `h` ∈ {1, 6, 24, 48} hours. 72 h only as an optional extension if everything else is finished.
- Multi-output/sequence forecasts are not part of the core protocol.

## 2. Splits (decision D-04; frozen)
Chronological, no shuffling. A sample belongs to the split of its **target time** `t + h`; its input window may reach into earlier splits (inputs are past information), but a training sample's target must lie inside the training period.

| Split | Target times | Purpose |
|---|---|---|
| Train | 2021-09-01 00:00 → 2023-12-31 23:00 | fit parameters |
| Validation | 2024-01-01 00:00 → 2024-06-30 23:00 | early stopping, hyper-parameters, feature-group decisions |
| **Test** | 2024-07-01 00:00 → 2025-06-24 23:00 | final reporting, **touched once** |

Notes: the test period spans a full year (monsoon, post-monsoon, winter, summer; Diwali 2024, Holi 2025). Demand grows about 7–9 % per year, so test demand is higher than anything in training; this is expected and reported (see section 6, trend).

## 3. Valid samples and missing data (decision D-14)
- A sample is **valid** only if its target is an observed value: not missing and not imputed (`demand_imputed == False` and not NaN).
- Inputs: imputed demand values may be used as inputs (flagged); a sample whose required input window contains an **unimputed missing** value is excluded for models that cannot handle NaN (LSTM, ridge). XGBoost may receive NaN natively; to keep comparisons fair, **all models are evaluated on the same set of valid samples** (the intersection).
- Counts of excluded samples per split and horizon are reported in every results table.
- Nothing is filled beyond the 3-hour interpolation already in the dataset; nothing is invented.

## 4. Feature groups and what is allowed at origin t
Ablation groups (each added to the previous one; every added group must be justified on **validation**):

| Group | Content | Rule |
|---|---|---|
| G0 history | demand lags y(t−k) for k ∈ {0,1,2,3,6,12,24,48,168}; rolling mean/std of the last 24 h and 168 h ending at t; seasonal anchors y(t+h−24·m) with the smallest m such that the anchor time ≤ t, and the same for 168 h | only values with time ≤ t. Anchors are therefore the "same hour yesterday/last week" whenever they are already known |
| G1 calendar | hour, weekday, month, day of year, weekend, season, sin/cos encodings **of the target time** t+h (and of t) | deterministic |
| G2 holidays | `is_holiday`, `is_major_festival` for the target day, signed days to the nearest major festival and holiday | published a year ahead; known |
| G3 weather | scenario A (headline): ERA5 national weather and regional temperatures at times ≤ t (lags 0, 1, 3, 6, 24 h; rolling 24 h mean/max of temperature). Scenario B (upper bound): additionally the ERA5 values at t+h. Degree-hours (CDD/HDD) with a base temperature chosen on the training period only | scenario B is an oracle and must always be labelled as such |
| G4 generation | past solar, wind (and `is_national_night` of the target time) at times ≤ t | **never** actual solar/wind at t+1…t+h |

Decision D-11: **scenario A is the headline result**; B is reported as an upper bound; C (archived forecast weather, GFS previous-day runs) is optional and limited by the caveats in decision D-13.
Features that describe cleaning (`demand_raw_mw`, `demand_quality_flag`, `demand_imputed`, `demand_stuck_flag`, `solar_raw_mw`) are never model inputs.

## 5. Models
1. **Baselines (mandatory, computed first):** persistence y(t); seasonal-naive-24 and seasonal-naive-168 using the anchor rule above; ridge regression on G0+G1 (and a variant with G2–G4 as the strongest linear reference).
2. **XGBoost:** one model per horizon, squared-error objective, early stopping on validation, tabular features.
3. **LSTM:** one model per horizon, input sequence of the last 168 h of the same variables, plus the deterministic features of the target time concatenated at the output stage; modest architecture (1–2 layers). Scalers fitted on the training period only.
4. **Equal tuning budget:** 30 random-search trials per model per horizon (adjustable only before the first test run, documented), selection by validation MAE; ≥ 3 random seeds for the LSTM, 3 for XGBoost; report mean and spread across seeds.
5. Any model-design change after seeing test results is a post-hoc deviation.

## 6. Non-stationarity / trend (decision D-07 — still open)
Candidate formulations: (a) predict absolute MW; (b) predict the ratio to the seasonal anchor (the anchor rule of section 4, G0). The training-period EDA (`reports/eda_training_period.md`) found a full-year mean growth of 7.8% (2022 to 2023), month-to-month year-over-year growth between −1% and +22%, and very strong 24 h / 168 h autocorrelation (0.96 / 0.90), which favours an anchor-relative target as the primary candidate. **Frozen 2026-10-14:** the rule fixed in advance (lower mean validation MAE over the four horizons, ridge on identical samples) selected the anchor-relative target (3,132 vs 3,226 MW; full model 3,122 vs 3,192 MW). Primary formulation = relative to the anchor; absolute MW is still run as a sensitivity variant.

## 7. Metrics
Per horizon, on the same valid samples: MAE (MW), RMSE (MW), MAPE (%), WAPE = Σ|e| / Σ|y| (%), and R² (supplementary only). Breakdowns: season (IMD-style), hour of day, weekday vs weekend vs holiday, ±1 day around major festivals, and **peak hours** (hours whose demand is at or above the 95th percentile of the *training* demand). Skill relative to the best baseline: 1 − MAE_model / MAE_baseline.

## 8. Comparing models fairly
- Same samples, same period, same metrics for every model.
- Differences are tested on the paired absolute-error differences with the Diebold–Mariano test (HAC / Newey–West variance with lag h−1) and a moving-block bootstrap (block length 168 h, 1,000 resamples) for confidence intervals.
- No statement such as "model X is better" without the baseline comparison and an interval on the difference.
- Feature-group conclusions come from ablations at fixed model and hyper-parameter budget.

## 9. Rolling-origin stability check (does not touch the test period)
Expanding window on train+validation only: (i) train to 2022-12-31 → evaluate 2023-01-01…2023-06-30; (ii) train to 2023-06-30 → evaluate 2023-07-01…2023-12-31; (iii) the main split (train to 2023-12-31 → validation 2024-H1). Report the variation of MAE across folds.

## 10. Uncertainty (optional extension; only after core results)
Quantile levels 5 %, 50 %, 95 % (90 % prediction interval) with XGBoost quantile objective and an LSTM multi-quantile head, plus split/adaptive conformal intervals calibrated on the validation period. Metrics: PICP (target 90 %), mean interval width, pinball loss; coverage reported by season and hour. Limitations to state: non-exchangeability under drift, quantile crossing, MC-dropout miscalibration.

## 11. Computational-efficiency logging (every run)
Written to one JSON line per run: run id, date, dataset SHA-256 and row count, feature groups, horizon, scenario, split boundaries, hyper-parameters, seeds, library versions, OS, CPU model and thread count, RAM, GPU (if any); training wall-clock and CPU time (data loading reported separately), epochs/rounds actually used, peak memory, model size and parameter count; inference latency (single sample and batch of 1,024; after warm-up; median and IQR over ≥ 30 repeats); number of tuning trials and total tuning time. Same thread count and hardware for both models. No efficiency claim without these measurements.

## 12. Reproducibility
Fixed seeds, scripts only (no hand-edited results), outputs in `reports/tables/` and `reports/figures/`, the dataset hash in every log. The project is not under version control yet; **initialise git before the first model run** so each run can record a commit.

## 13. Deviations log
Anything changed after results were seen is appended here with date, reason, and its effect:

| Date | Deviation | Reason | Effect |
|---|---|---|---|
| 2026-10-14 | Degree-hours (CDD/HDD) are **not** used for XGBoost | Trees learn temperature thresholds themselves; avoids choosing a base temperature | Pre-results clarification; may be added for the LSTM/ridge |
| 2026-10-14 | Hyper-parameter search (30 random trials per horizon) is run once on the headline set S4_full_A with the ratio target; the ablation over S0-S5 and the absolute-target variant reuse those hyper-parameters (3 seeds) | Keeps the ablation at a fixed model budget (section 8) and the search affordable | Pre-results clarification; the absolute-target variant is therefore not separately tuned |
| 2026-10-14 | The ratio target is trained as 100 x (ratio - 1) | Same target on a scale where gamma/lambda ranges are meaningful; the first smoke run (2 trials, one horizon) showed identical results for all feature sets because the penalties blocked every split | Smoke-test only, no real results had been produced |
| 2026-10-14 | `night_T` removed from the feature set | It was derived from future ERA5 radiation, which a strict leakage test must flag; hour and day of year carry the same information | Pre-results |
| 2026-10-14 | XGBoost training rows = the same valid samples as the baselines; early stopping rounds 40, at most 1,500 trees | Same samples for all models (section 3) | Pre-results |

## 14. Gate 3 checklist (must all be true before any model training)
- [x] Canonical dataset built and documented
- [x] Splits frozen (section 2)
- [x] Task and horizons defined (section 1)
- [x] Feature groups and allowed information defined (section 4)
- [x] Metrics, comparison procedure and logging defined
- [x] D-07 trend formulation frozen (relative to anchor)
- [x] Baselines implemented and logged (Phase 5): `reports/tables/baselines_validation.md`
- [x] `git init` done (2026-10-14; first commit f2a56f3; record the commit hash in every run log)
