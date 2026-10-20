# Project plan — India-wide electricity demand forecasting (XGBoost vs LSTM)

**This is the living plan we work from.** Update the status boxes and the decision log whenever something is done or decided. Last updated: 2026-10-13.

Related documents: `docs/literature_review_and_research_gap.md` (background and gap), `reports/` (audit, dictionary, acquisition plan), `references.md` (sources).

Legend: `[x]` done · `[ ]` to do · `[~]` partly done (the row says what remains) · **GATE** = must pass before the next phase starts · size S ≈ an hour or two, M ≈ a day, L ≈ several days (rough, for an undergraduate working alone).

---

## 0. How we work (rules)

1. **One phase at a time, in order.** Do not start a phase until the previous GATE is passed.
2. **No model training before Gate 3** (clean dataset, frozen splits, baselines defined).
3. **Never fabricate or fill data to look complete.** Every imputation or exclusion is counted and logged.
4. **Raw files are immutable.** All processing reads raw and writes to `data/interim` or `data/processed`.
5. **Chronological order only** — no random splits, no shuffling, no future information (see leakage rules in `reports/data_acquisition_plan.md` section 6).
6. **Test period is touched once**, after all choices (features, hyper-parameters, thresholds) are frozen on train/validation.
7. **Every claim is traceable**: a source in `references.md`, or a result file produced by a script that is committed.
8. **No claims of model superiority** without measured results, baselines, and uncertainty on the difference.
9. **Decisions go in the decision log (section 13)** with the reason and date, so they can be defended later.
10. Scripts must be reusable, documented, runnable with one command from the project root, and use fixed random seeds.

---

## 1. Where we are (status snapshot)

| Area | Status |
|---|---|
| Workspace and structure | `[x]` created, `.gitignore`, `requirements.txt`, README |
| Electricity data (Mendeley) | `[x]` downloaded, extracted, fully audited — all-India series, 2021-09-01 → 2025-06-24, 33,432 hours, needs cleaning |
| Weather (Open-Meteo ERA5) | `[x]` 12 cities downloaded, 33,432 rows each, 0 nulls; aggregated to national (equal-weight) in the canonical dataset |
| Calendar/holidays | `[x]` time features built; 85 central gazetted holidays sourced from and verified against the DoPT memoranda (0 differences); regional festivals added as an exploratory extension |
| Zenodo daily state data | `[x]` 42 files, MD5-verified; used to validate Mendeley (r = 0.997); separate daily study optional |
| Literature | `[x]` 8 papers recorded and DOIs checked; novelty search still pending |
| Cleaning / canonical dataset | `[x]` cleaned hourly demand/solar/wind and the merged canonical dataset (`india_hourly_canonical.csv`) |
| Baselines / models | `[x]` baselines (Phase 5), XGBoost (Phase 6) and LSTM (Phase 7) done on validation; test set untouched |

---

## 2. Final research design in one page (current working version)

- **Target:** all-India demand (MW), HH:00 snapshot, hourly, IST, 2021-09-01 00:00 → 2025-06-24 23:00. Sensitivity check with hourly means for the 2021–2023 period.
- **Inputs (feature groups for ablation):** (G0) demand history (lags, rolling stats up to the forecast origin); (G1) calendar (hour, weekday, month, season, cyclical); (G2) holidays/festivals; (G3) weather (national aggregate of 12 ERA5 cities); (G4) solar/wind generation **up to the origin only**.
- **Models:** baselines (persistence, seasonal-naive 24 h and 168 h, ridge) → XGBoost → LSTM. Same splits, features and metrics for all.
- **Horizons:** 1, 6, 24, 48 hours (72 h only if justified). Each defined precisely as a single-target "direct" forecast of the value at t+h unless we decide otherwise (decision D-06).
- **Splits (proposed, to be frozen):** train 2021-09-01 → 2023-12-31; validation 2024-01-01 → 2024-06-30; test 2024-07-01 → 2025-06-24. Plus rolling-origin check.
- **Weather scenarios:** (A) no future weather, (B) oracle weather at t+h as an upper bound, (C) archived forecasts if available.
- **Metrics:** MAE, RMSE, MAPE, WAPE (R² supplementary), per horizon and season/hour/holiday.
- **Uncertainty (optional extension):** quantile regression and/or conformal intervals; PICP, width, pinball loss.
- **Efficiency:** training/inference time, hardware, memory, parameters — logged for every run.
- **Core vs optional** follows section 6 of the literature document. EV variables are out unless reliable data appears.

---

## 3. Phase 1 — Finish data acquisition and validation  *(mostly done)*

| # | Task | Size | Status |
|---|---|---|---|
| 1.1 | Audit Mendeley workbooks (schema, period, gaps, quality) | L | `[x]` `reports/dataset_audit.md` |
| 1.2 | Download 12-city ERA5 weather | S | `[x]` `data/raw/weather/` |
| 1.3 | Validate weather files: time grid, ranges, units | S | `[x]` grid OK, ranges plausible |
| 1.4 | Record weather units in the dictionary (wind in **km/h**, GHI W/m² preceding-hour mean, precipitation mm preceding hour, T in °C) and fix the wind-speed column name | S | `[x]` done 2026-10-13, units verified from the API |
| 1.5 | Verify holiday CSV against the official DoPT memoranda | S | `[x]` done 2026-10-13: all 5 years (85 rows) compared with Annexure-I of the DoPT OMs; 0 differences. 2023 from the official doptcirculars.nic.in PDF; 2021/2022/2024/2025 from scanned OM PDFs hosted on staffnews.in |
| 1.6 | Decide on regional/state festivals (see D-08); if yes, collect with citations | M | `[x]` deferred (D-08), then done as an exploratory extension 2026-10-16: state-level holidays from the `holidays` package (`reports/regional_holidays.md`); source incomplete and not checked against state gazettes; no benefit found |
| 1.7 | Spot-check Zenodo per-state weather against Open-Meteo for one state and a few days | S | `[x]` 2026-10-14: Delhi daily mean temperature, 969 overlapping days: correlation 0.999, mean difference -0.19 C, MAE 0.31 C (0.22 C on a UTC day) — consistent with ERA5-derived data as assumed |
| 1.8 | Investigate availability of **archived weather forecasts** for scenario C | M | `[x]` done 2026-10-13, see decision D-13 below |
| 1.9 | Investigate any **solar/wind forecast** availability (otherwise G4 uses past values only) | S | `[x]` investigated 2026-10-16: no public archive of historical forecasts found; see `docs/data_availability_notes.md`; G4 uses past actuals only |

**GATE 1:** all raw sources documented in `reports/dataset_inventory.md` and `references.md`; the three open questions (1.4, 1.5, D-08) are resolved or consciously deferred.

---

## 4. Phase 2 — Cleaning and the canonical hourly dataset

Goal: `data/processed/india_hourly_canonical.csv` (columns as in `reports/data_dictionary.md`) built by one reproducible script with a cleaning log.

| # | Task | Size | Status |
|---|---|---|---|
| 2.1 | Freeze cleaning rules (zeros → missing; plausibility bounds; dip-and-recovery; isolated-point; stuck values; solar clipping) | M | `[x]` frozen 2026-10-13, see data dictionary. Night-baseline correction deferred to Phase 3 |
| 2.2 | Write `scripts/build_hourly_dataset.py`: HH:00 snapshots from all 29 workbooks, drop footer rows, apply rules, add `demand_quality_flag` and `source_regime` | L | `[x]` output `data/processed/india_hourly_demand_clean.csv` |
| 2.3 | Gap handling: interpolate ≤ 3 h only; longer gaps kept missing; decide how windows touching gaps are excluded from training/evaluation | M | `[x]` interpolation (49 h), 12 h left missing; samples are valid only if the target is observed and all inputs present (protocol section 3) |
| 2.4 | Produce a cleaning log: hours affected per rule, per year | S | `[x]` `reports/cleaning_log.md` |
| 2.5 | Plots before/after cleaning (demand, outages) and the solar night baseline | S | `[x]` `reports/figures/cleaning_before_after.png`, `reports/figures/solar_night_baseline.png` |
| 2.6 | Sensitivity file: hourly-mean target for 2021–2023 | M | `[x]` `data/interim/demand_hourmean_2021_2023.csv`, `reports/hourmean_sensitivity.md`; ridge result in `reports/sensitivity_summary.md` |
| 2.7 | Unit tests / assertions: 33,432 rows, strictly hourly, no duplicate timestamps, no values outside bounds after cleaning | S | `[x]` assertions run inside the script |

**GATE 2:** canonical demand/solar/wind file exists, passes assertions, and the cleaning log is reviewed by you.

---

## 5. Phase 3 — Weather aggregation and feature tables

| # | Task | Size | Status |
|---|---|---|---|
| 3.1 | Choose aggregation weights and **document the source** (Census 2011 population per grid region, or regional electricity consumption from CEA/Grid-India); decision D-03 | M | `[x]` equal weights as baseline; population and consumption weights computed and compared (`reports/weather_weights.md`, `reports/sensitivity_summary.md`) |
| 3.2 | Build national weather: region mean → weighted national mean; also keep 5 regional temperature columns | M | `[x]` equal-weight baseline built (D-03) |
| 3.3 | Derived weather features: CDD/HDD (base temperature chosen on **training data only**), heat index/apparent temperature, temperature lags | M | `[x]` done 2026-10-16 for ridge (cooling/heating degree-hours with training-chosen base 26 C, 24 h means, apparent temperature): no practical gain on validation or test (differences of -9 to +5 MW, `reports/ridge_cddh_test.md`); trees were not given them by design; temperature lags and rolling means are in the feature pipeline |
| 3.4 | Align weather timestamps with demand snapshots; handle the preceding-hour (precipitation, radiation) vs instant offset deliberately (D-09) | S | `[x]` no shift needed (D-09) |
| 3.5 | Rebuild calendar features for the final range; add festival-window features (days to/from major festivals) | S | `[x]` festival/holiday windows (days to next / since last) added in `features_tabular.py` group g2 |
| 3.6 | Merge into the canonical dataset; update the data dictionary (every column, unit, source, known-at-forecast-time flag) | M | `[x]` `data/processed/india_hourly_canonical.csv` (33,432 rows), `reports/canonical_dataset_log.md`, data dictionary updated |
| 3.7 | Sensitivity: equal vs population vs consumption weights; national mean vs regional columns | M | `[x]` weights in `reports/weather_weights.md`; ridge validation MAE changes by at most 41 MW (`reports/sensitivity_summary.md`) — the choice of weights does not matter at this level. National-mean vs regional temperature columns remains to be tested with XGBoost |

**GATE 3:** canonical dataset with all feature groups and dictionary complete; **splits frozen** (D-04); evaluation protocol written (Phase 4).

---

## 6. Phase 4 — Evaluation protocol and exploratory analysis (training period only)

| # | Task | Size | Status |
|---|---|---|---|
| 4.1 | Write `docs/experiment_protocol.md`: exact horizons definition, splits, metrics formulas, scenarios A/B/C, baselines, ablation list, random seeds, logging schema | M | `[x]` written and frozen 2026-10-13 |
| 4.2 | Exploratory analysis **on the training period only**: daily/weekly/annual profiles, trend/growth, temperature–demand curve, festival and holiday effects, autocorrelation, solar/wind relations | L | `[x]` `reports/eda_training_period.md` + `reports/figures/eda_training_period.png` (2026-10-13) |
| 4.3 | Decide the treatment of trend/non-stationarity (levels vs ratios to lag; tree models cannot extrapolate beyond the training range) — D-07 | M | `[x]` frozen 2026-10-14: ratio to the 24 h seasonal anchor, chosen by a rule fixed in advance on validation |
| 4.4 | Build the experiment-logging utility (JSON/CSV: config, git state, seeds, hardware, timings) | M | `[x]` implemented inside each run script: one JSON line per fit with git commit, dataset hash, seeds, hardware, timings (`reports/tables/*.jsonl`); not a separate utility |

---

## 7. Phase 5 — Baselines

| # | Task | Size | Status |
|---|---|---|---|
| 5.1 | Persistence, seasonal-naive 24 h and 168 h | S | `[x]` |
| 5.2 | Ridge / linear model with calendar + lags (+ full variant with holidays, past weather, past solar/wind) | S | `[x]` 4 variants (abs/rel x history/full) |
| 5.3 | Evaluate all baselines on validation (test is never loaded) per horizon | S | `[x]` `reports/tables/baselines_validation.md` + `.jsonl` (commit, dataset hash, timings) |
| 5.4 | Establish the "must beat" table and error by season/peak | S | `[x]` see below; the bar for XGBoost/LSTM is the best ridge, not the naive baselines |

**GATE 4:** baselines reproducible and logged. Only now can model training start.

---

## 8. Phase 6 — XGBoost

| # | Task | Size | Status |
|---|---|---|---|
| 6.1 | Feature pipeline with leakage tests (unit test: perturbing every measurement after the origin changes no earlier feature row) | L | `[x]` `scripts/features_tabular.py`, `tests/test_leakage.py` (0 changed rows over 3 origins x 4 horizons x 2 formulations; leaky canary detected) |
| 6.2 | Direct models per horizon (1, 6, 24, 48 h), chronological early stopping on validation | M | `[x]` ratio target (primary) and absolute (sensitivity), 3 seeds |
| 6.3 | Limited, documented hyper-parameter search on validation (equal budget with LSTM) | M | `[x]` 30 random trials per horizon (best trial indices and timings in `reports/tables/xgboost_validation.md`). Some tuned values sit at the edge of the search ranges (max_depth 3 at 1 h and 48 h; gamma near 5 at 24 h), so the search is coarse |
| 6.4 | Ablations: G0 → +G1 → +G2 → +G3 (scenarios A and B) → +G4 | L | `[x]` S0..S5, both formulations; see `reports/tables/xgboost_validation.md` |
| 6.5 | Feature importance / SHAP for interpretation (validation data) | M | `[x]` `reports/xgboost_feature_importance.md` (built-in TreeSHAP) |
| 6.6 | Log training/inference time, memory, hardware | S | `[x]` in `reports/tables/xgboost_validation.jsonl` (28.5 min total run; per-fit training 4-15 s, 0.5-1.6 MB models, 400 MB peak working set, 13 ms single-row latency dominated by Python overhead) |

## 9. Phase 7 — LSTM

| # | Task | Size | Status |
|---|---|---|---|
| 7.1 | Sequence windowing with no cross-split leakage; scalers fit on train only | L | `[x]` `scripts/run_lstm.py` (168-step windows gathered on the fly; level channels standardised with training statistics; same valid samples as XGBoost, asserted) |
| 7.2 | Architecture kept modest (one LSTM layer + small MLP head; single-output direct per horizon) | M | `[x]` hidden 32-64, 10-27 thousand parameters |
| 7.3 | Limited hyper-parameter search with the same budget as XGBoost; multiple seeds (>= 3) | L | `[x]` first tuning had 12 trials per horizon (documented); equal-budget re-tuning with 90 trials per horizon done 2026-10-16 (`reports/lstm_stage3.md`, test in `reports/lstm_stage3_test.md`); it did not change the conclusions; 70% of the 1 h trials were pruned early |
| 7.4 | Same ablations as 6.4 | L | `[~]` L1 (history+calendar), L2 (+holidays), L4 (full, scenario A) done; oracle-weather scenario and group-wise ablation of weather vs solar/wind not run for the LSTM |
| 7.5 | Log training/inference time, memory, parameters, hardware | S | `[x]` `reports/tables/lstm_validation.jsonl`; summary in `reports/validation_summary.md` |

## 10. Phase 8 — Comparison and robustness

| # | Task | Size | Status |
|---|---|---|---|
| 8.1 | Final test evaluation (once), all models, all horizons | M | `[x]` run once on 2026-10-15 at commit 155b206 (67 min); results in `reports/test_results.md`, summary in `reports/test_summary.md`; lock file `reports/tables/TEST_RUN_DONE.lock` prevents a second run |
| 8.2 | Error breakdowns: horizon, hour, season, holiday vs non-holiday, peaks, festival windows | M | `[x]` `reports/test_results.md` (season, day type, festival window, peak), `reports/hour_of_day.md` + `reports/figures/test_hour_of_day.png` |
| 8.3 | Significance of differences (Diebold–Mariano or block bootstrap) | M | `[x]` block-bootstrap CIs and DM statistics for six pairs per horizon in `reports/test_results.md` |
| 8.4 | Rolling-origin stability check | M | `[x]` `reports/rolling_origin.md` (2 folds on 2023 + the main split; 24 h ranking stable; 6 h XGBoost-vs-ridge ranking flips between folds; 1 h ridge at least as good as XGBoost in all folds) ; walk-forward with quarterly retraining over 2024-25 done 2026-10-16 (`reports/walk_forward.md`: XGBoost beats ridge in 6 of 6 windows at 6 h and 24 h; first two windows are validation-period) |
| 8.5 | Robustness: performance on the cleaned vs raw-flagged periods, the sampling-regime change, hourly-mean target sensitivity | M | `[~]` `reports/robustness.md` (excluding samples with imputed/stuck inputs; seed spread) plus earlier hourly-mean and weight sensitivities; the 5-min/10-s/hourly regime change cannot be tested on the test period (all hourly) |
| 8.6 | Answer the sub-questions in the literature document (weather? festivals? horizons? solar/wind?) with evidence | M | `[x]` `reports/research_question_answers.md` |

## 11. Phase 9 — Optional extensions (only if time permits)

| # | Task | Size | Status |
|---|---|---|---|
| 9.1 | Uncertainty: XGBoost quantile regression and split/adaptive conformal intervals; LSTM quantile head or MC dropout; PICP, width, pinball loss | L | `[x]` split, anchor-scaled and rolling conformal (XGBoost, LSTM, ridge), XGBoost quantile regression + CQR, LSTM quantile head + CQR (`reports/uncertainty.md`); interval score used instead of pinball loss; MC dropout not run |
| 9.2 | Daily regional study with Zenodo state data (regional forecasts → India total) as a separate experiment | L | `[x]` single test run 2026-10-16: bottom-up (sum of five regional XGBoost models) did not beat the direct national model (`reports/regional_daily_study_test.md`) ; multi-year daily walk-forward 2017-2024 with an LSTM added done 2026-10-16 (`reports/daily_multiyear.md`) |
| 9.3 | Regional festivals / state holidays features | M | `[x]` population-weighted state-holiday features from the `holidays` package (`reports/regional_holidays.md`); no benefit on validation or test (`reports/extensions_xgb_test.md`); source incomplete for multi-day regional festivals |
| 9.4 | Gridded (population-weighted) weather alternative | M | `[x]` done 2026-10-16 with state-capital points (not true grids) weighted by Census-2011 population: no robust effect (validation null; test 1-2% better for XGBoost at 24-48 h with a sign that flips from validation; `reports/statepop_weather_test.md`) |
| 9.5 | Longer horizons (72 h, 168 h) | M | `[x]` done 2026-10-16 (validation tuning, one test run; `reports/long_horizons.md`, `reports/long_horizons_test.md`): 72 h XGBoost lowest but not distinguishable from ridge or the LSTM; 168 h ridge clearly best, XGBoost and LSTM only 3.6% better than naive ; extended 2026-10-16 to the ablation, walk-forward, rolling origin, learning curve, cross-validated tuning, equal-budget LSTM, forecast temperature and regional holidays, intervals, breakdowns and feature importance at 72 h and 168 h (all in `reports/`; see the README list) |
| 9.6 | EV features — only if reliable historical data is found | – | `[x]` closed as not feasible and not material (EV charging about 0.2% of 2024 consumption; no open hourly data); see `docs/data_availability_notes.md` |
| 9.7 | A separate operational day-ahead model: issue time 10:00 on day D-1, the 24 hourly values of day D (leads 14 to 37 h); naive, ridge (one model per target hour), XGBoost S4 and LSTM L4 (both pooled over the target hours; deviation from the original plan wording explained in the protocol entry); same chronological split, validation first, one test run. **Done 2026-10-19** (approved by the owner; declared in the protocol log first): `scripts/run_dayahead.py`, `scripts/analyze_dayahead.py`, `reports/dayahead_validation.md`, `reports/dayahead_test.md`, `reports/dayahead_analysis_test.md`, `reports/figures/fig14_dayahead.png`; findings in `reports/research_question_answers.md` | L | `[x]` |
| 9.8 | Derived day-wise view from the saved 24 h-ahead hourly predictions (daily energy in MU, daily peak and peak hour; bootstrap by whole days, 28-day blocks; no refitting, no new test evaluation). **Done 2026-10-19** (approved by the owner; declared in the protocol log first): `scripts/run_daywise.py`, `reports/daywise_test.md`, `reports/figures/fig13_daywise.png`; findings in `reports/research_question_answers.md` | S | `[x]` |
| 9.9 | **Completion batch (owner, mentor approved, 2026-10-19; declared in the protocol log first):** P1 peak-exceedance precision/recall/F1; P2 day-ahead completions (per-hour XGBoost, LSTM L1, 24-output LSTM, feature ablation incl. forecast temperature, intervals); P3 other model families (LightGBM at 1-168 h and day-ahead; GRU at 1 h, 24 h and day-ahead; the TCN and the Transformer were dropped by the owner on 2026-10-20 before any run of the amended versions); P4 SARIMAX and Prophet on the day-ahead task (dropped by the owner 2026-10-20 before any run; scripts kept); P5 licence (needs the owner's choice) | L | `[x]` done 2026-10-20: P1, P2, P3 (LightGBM, GRU) complete; P4 and the TCN/Transformer dropped by the owner; results in `reports/research_question_answers.md` |

## 12. Phase 10 — Novelty check and writing

| # | Task | Size | Status |
|---|---|---|---|
| 10.1 | Broader literature search for national-scale Indian forecasting (ML/DL, weather, festivals); update the research gap honestly | L | `[ ]` (before any publication claim) |
| 10.2 | Computational-efficiency table from the logs | S | `[x]` `reports/efficiency_table.md`, generated by `scripts/build_efficiency_table.py` from the logs (final refits, per-fit cost, inference, tuning budget; 1-48 h; the 72/168 h extension has no comparable cost log) |
| 10.3 | Figures and tables generated by scripts, saved to `reports/figures/` and `reports/tables/` | M | `[x]` 12 figures by `scripts/make_figures.py` in `reports/figures/fig01..fig12` (PNG) with captions and caveats in `reports/figures/FIGURE_CAPTIONS.md`; final selection for the paper is the owner's choice |
| 10.4 | Write report/paper: data (with documented limitations), method, results, threats to validity, reproducibility appendix | L | `[~]` first draft written 2026-10-21 (`docs/paper_draft.md`; outline in `docs/paper_outline.md`); open: venue and template, human review of the draft, verification of every [verify] reference, the licence, a clean-checkout reproduction test; the systematic literature search was skipped by decision of the owner (no novelty claim) |
| 10.5 | Clean the repository: README with exact reproduction commands, requirements pinned, licence/attribution notes | S | `[~]` README rewritten and extended 2026-10-16 (exact commands, figure/efficiency scripts); `requirements.txt` pinned to the versions used; still open: licence for the code (the owner's decision), a clean-checkout reproduction test |

---

## 13. Decision log and open decisions

| ID | Decision | Status | Notes |
|---|---|---|---|
| D-01 | Target = all-India HH:00 demand snapshot (MW) | **Decided (working)** 2026-10-13 | Consistent across regimes; hourly-mean sensitivity planned |
| D-02 | Weather locations = 12 cities across 5 grid regions | **Approved by owner** 2026-10-13 | Downloaded; city selection is a stated limitation |
| D-03 | Aggregation weights for national weather | **Decided 2026-10-13; sensitivity done 2026-10-14: weights are immaterial for ridge validation error (<= 41 MW)** | Baseline = equal weights over the 5 grid regions (region mean of its cities, then mean of regions). Population-/consumption-weighted versions are sensitivity analyses (3.7) and need a cited source before use; none is adopted without one |
| D-04 | Train/validation/test boundaries | **Frozen 2026-10-13 (approved by owner)** | Train targets 2021-09-01 → 2023-12-31; validation 2024-01-01 → 2024-06-30; test 2024-07-01 → 2025-06-24 (touched once). See `docs/experiment_protocol.md` |
| D-05 | Cleaning thresholds: 80–260 GW bounds, 15 % jump + return rule, 10 % isolated-point rule, ≤ 3 h interpolation | **Frozen 2026-10-13** | Set from physical plausibility and hour-to-hour deviation statistics of the whole series, which includes the future validation/test years (no forecast performance used). Disclose this in the write-up; a robustness run with the rules applied only on training-period statistics can be added |
| D-06 | Horizon definition | **Decided 2026-10-13 (approved by owner)** | Direct single-target forecast of demand at t+h, one model per horizon, h ∈ {1, 6, 24, 48} |
| D-07 | Trend handling | **Frozen 2026-10-14** | By the rule fixed in advance: mean validation MAE over the four horizons, ridge on identical samples: relative-to-24h-anchor 3,132 MW vs absolute 3,226 MW (history+calendar model) and 3,122 vs 3,192 MW (full model). The relative target is clearly better at 24 h and 48 h (block-bootstrap CI of the paired difference excludes 0) and equal or marginally worse at 1 h (+4 MW) and 6 h. **Primary formulation: target and demand features relative to the seasonal anchor; absolute MW remains a reported sensitivity variant.** Caveat: decided with a linear model; XGBoost/LSTM results will be reported for both |
| D-08 | Include regional/state festivals | **Deferred 2026-10-13** | Core model uses the 85 verified central holidays only; regional festivals become optional extension 9.3 after the main results exist |
| D-09 | Handling of preceding-hour weather vs instant demand | **Decided 2026-10-13** | No shift. Precipitation and radiation are preceding-hour aggregates that end at their label, so they are fully known at that label, the same instant as the demand snapshot |
| D-10 | Whether to use solar/wind past values (G4) | **Open** | Past values only; actual future values never |
| D-11 | Which weather scenario is the headline result | **Decided 2026-10-13 (approved by owner)** | Scenario A (no future weather) is the headline; B (oracle weather at t+h) is an upper bound; C optional |
| D-13 | Weather scenario C (real forecast weather) | **Run 2026-10-16 as an exploratory extension, temperature only** (the archive has no other variable before 2024): forecast temperature helped by -4.5% at 24 h and -7.3% at 48 h on the test period, slightly worse at 1 h; the frozen headline is unchanged. Original assessment: feasible with caveats | Open-Meteo Historical *Forecast* API only stitches the first hours of each run (not a true day-ahead forecast) and must NOT be used for 24/48 h leads. Previous Runs API (`gfs_seamless`, `*_previous_day1/2`) gives lead-time-specific GFS forecasts for 2021-09 to 2025-06, but **missing ≈ 43 h in 2023-12 and ≈ 450 h in 2024-01** (sits at the train/validation boundary), and in a Delhi spot check was biased about +3 to +4 °C vs ERA5 with MAE 4.2 °C (day 1) and 4.6 °C (day 2) in May 2023. Using it requires either bias handling or training on forecast data; ECMWF IFS runs only from mid-2024. Scenario C would therefore be an optional, clearly caveated analysis on the later period |
| D-14 | Valid-sample rule | **Decided (working) 2026-10-13** | A sample is valid only if its target is observed (not missing, not imputed); all models are compared on the same valid samples; counts are reported |
| D-12 | Demand definition = all-India demand met (SCADA, MW) | **Assumed** | Supported by Zenodo cross-check (r = 0.997, MAPE 0.57 %) |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Only ~3.8 years of hourly data → few festival repetitions, limited season cycles | Report uncertainty honestly; use daily Zenodo data (2013–2024) as a supplementary festival analysis |
| Demand growth makes test data higher than training range (trees cannot extrapolate) | Model relative to a lagged level or add a trend term; report separately |
| Sampling-regime change (5 min / 10 s / hourly) may create inconsistent behaviour | Include `source_regime`; test robustness (8.5) |
| Cleaning choices influence results | Freeze before test, report sensitivity |
| ERA5 reanalysis is not a forecast | Scenarios A/B/C, clearly labelled |
| Weather aggregation is a design choice | Sensitivity analysis (3.7) |
| LSTM tuning cost on a student machine | Modest architecture, equal and limited search budgets, log everything |
| Scope creep | Core vs optional split; extensions only after Phase 8 |
| Gap may not be novel | Phase 10.1 before any publication claim |

---

## 15. Next actions (always keep this list short and current)

1. **Owner review** of `reports/test_summary.md`, `reports/research_question_answers.md` (now including the extensions) and `docs/novelty_assessment.md`. No frozen model or setting may be changed in response without logging a post-hoc deviation.
2. **Phase 10:** a proper systematic literature search (Scopus / Web of Science / Google Scholar; queries and open points are listed in `docs/novelty_assessment.md`), then the write-up. Any "novel" claim must wait for that search.
2b. **Figures: done 2026-10-16** (12 PNG figures, `scripts/make_figures.py`, captions in `reports/figures/FIGURE_CAPTIONS.md`); the consolidated efficiency table is `reports/efficiency_table.md`. Open for the owner: choose which figures go into the paper, and decide on a licence for the code.
2c. **Parked pending the mentor:** the choice of a licence for the code (none added yet; the data licences are listed in the README) and whether to add other model families. The day-ahead model and the day-wise view (tasks 9.7 and 9.8) were done on 2026-10-19.
3. Remaining optional items (not done): MC dropout for the LSTM, forecast humidity/wind/radiation (not available in the archive before 2024), capacity-normalised solar/wind features, an LSTM in the daily regional study, a comparison with operational Grid-India forecasts if they can be obtained.
4. Housekeeping: delete the temporary folders `D:\tmp_pip` and `D:\pylibs_verify` (outside the project); decide whether to keep `data/interim/predictions` (saved predictions, git-ignored).

**Test results (MAE, MW; 1/6/24/48 h; `reports/test_summary.md`):** XGBoost headline 1,111 / 2,747 / 3,728 / 5,158; ridge 1,049 / 3,082 / 4,214 / 5,549; LSTM full features 1,643 / 3,170 / 3,743 / 5,325; LSTM history+calendar 1,527 / 2,981 / 3,753 / 5,280; best naive 4,272 / 5,414 / 5,414 / 6,964. XGBoost beats ridge at 6-48 h; at 1 h they are not distinguishable; the LSTM is worse than XGBoost at 1-6 h and not distinguishable from it at 24-48 h. Past weather/solar/wind did not help on test (they did on validation); oracle weather would (-5.9% at 24 h, -11.5% at 48 h); holiday features give 1-2%.

**Validation status (details in `reports/validation_summary.md`):** XGBoost stage 2 (MAE, MW; 1/6/24/48 h): 1,104 / 2,504 / 3,170 / 4,247; best ridge 1,089 / 2,941 / 3,763 / 4,663; LSTM full features 1,584 / 3,048 / 3,273 / 4,778; LSTM history+calendar (own tuning) 1,485 / 2,696 / 3,278 / 4,337. Hypothesis (not tested): LSTM weather/generation inputs hurt because validation solar exceeds the training range (8.9% of validation hours above the training maximum; mean 40% higher).

**Diagnostic learning curve (2026-10-15, `reports/learning_curve.md`):** the LSTM's error is nearly flat in training-set size (12/18/28 months) at 1, 24 and 48 h and improves at 6 h (about 8%, against about 9% for XGBoost), while XGBoost keeps improving at all four horizons (the same holds at 72 h and 168 h: `reports/long_learning_curve.md`); the 'LSTM needs more data' explanation is not supported within this range. Descriptive, three sizes, no significance tests.

Done 2026-10-17: full consistency review after the long-horizon batch (numbers audit 181/181, quoted intervals 94/94, leakage test passed, all 12 test-run locks intact, figures and efficiency table regenerate identically); corrections made to the LSTM inference ratio, the learning-curve wording, a cross-validation sentence and the audit scripts (see the protocol deviation log, entry of 2026-10-17). Done recently: extensions run (LSTM quantile head, forecast-temperature scenario C, regional festivals, regional daily study, novelty search; 2026-10-16); export re-run (reproduced exactly), hour-of-day, robustness, rolling origin, uncertainty extension, answers to the research questions (2026-10-15); learning-curve diagnostic; pre-test stage 2 (widened XGBoost search; own LSTM tuning for L1), 2026-10-15; Phase 7 LSTM; Phase 6 XGBoost; baselines; EDA; sensitivity analyses; leakage test.
