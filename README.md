# India-wide electricity demand forecasting: XGBoost vs LSTM

Research objective: forecast all-India hourly electricity demand 1, 6, 24 and 48 hours ahead with machine learning and deep learning (XGBoost and LSTM, plus ridge and naive baselines), using historical demand, weather and calendar/festival information, and compare accuracy, uncertainty and computational efficiency.

**Current phase: experiments finished (validation and the single test run); novelty check and writing remain.** See `docs/project_plan.md` (living plan, decision log, next actions) and `docs/experiment_protocol.md` (frozen protocol, deviation log).

> **Read first:** `docs/limitations.md` lists the limits that apply to every result (one test year, optimistic validation numbers, tuning budgets, repeated use of the test period, unverified inputs).

## Results in brief (test period 2024-07-01 to 2025-06-24, MAE in MW)
| Model | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| Best naive | 4,272 | 5,414 | 5,414 | 6,964 |
| Ridge | 1,049 | 3,082 | 4,214 | 5,549 |
| XGBoost (declared headline) | 1,111 | 2,747 | 3,728 | 5,158 |
| LSTM full features (declared headline) | 1,643 | 3,170 | 3,743 | 5,325 |
| LSTM full features, equal tuning budget (added later) | 1,549 | 3,151 | 3,776 | 5,418 |
XGBoost is best from 6 h on (the LSTM is not distinguishable from it at 24-48 h); ridge and XGBoost tie at 1 h. Details, intervals and caveats: `reports/test_summary.md`, `reports/research_question_answers.md`.

## Data
- Demand, solar, wind: Mendeley Data "Electricity Demand, Solar and Wind Generation Data (September 2021 - June 2025) of India" (all-India, 2021-09-01 to 2025-06-24; 5-minute / 10-second / hourly sampling reduced to HH:00 snapshots). Audit: `reports/dataset_audit.md`.
- Weather: Open-Meteo ERA5 reanalysis for 12 cities, aggregated to a national equal-weight series. Calendar: 85 central gazetted holidays verified against the DoPT memoranda.
- Cleaning rules and canonical dataset: `reports/cleaning_log.md`, `reports/canonical_dataset_log.md`, `reports/data_dictionary.md`.

## Layout
```
data/raw/            downloads (git-ignored except calendar and population CSVs)
data/interim/        regenerable caches, saved predictions (git-ignored)
data/processed/      cleaned and canonical datasets (git-ignored, regenerable)
scripts/             all code (see below)
tests/               leakage test for the feature pipeline
reports/             audits, logs, result tables (reports/tables), figures (reports/figures)
docs/                project plan, experiment protocol, literature review
references.md        external sources
```

## Setup
```powershell
py -m pip install -r requirements.txt
```
(`py` is the Python launcher on this machine.) `requirements.txt` pins the exact versions used for all results (Python 3.14.3). PyTorch (CPU) is needed for the LSTM and is installed separately.

## Reproduce
```powershell
py scripts\inspect_electricity_data.py         # audit all workbooks (about 35 min first run)
py scripts\inspect_weather_data.py --fetch     # download ERA5 weather
py scripts\build_calendar_features.py
py scripts\build_hourly_dataset.py             # cleaned hourly demand/solar/wind
py scripts\compute_region_weights.py           # sensitivity weights for the weather aggregate
py scripts\build_canonical_dataset.py          # merge demand + weather + calendar
py tests\test_leakage.py                       # leakage test of the feature pipeline
py scripts\eda_training_period.py              # exploratory analysis, training period only
py scripts\run_baselines.py                    # naive and ridge baselines on validation
py scripts\run_xgboost.py ; py scripts\run_xgboost.py --wide   # XGBoost stage 1 and stage 2 (validation)
py scripts\run_lstm.py ; py scripts\run_lstm.py --stage2       # LSTM stage 1 and stage 2 (validation)
py scripts\run_learning_curve.py               # diagnostic learning curve (validation)
py scripts\run_test_evaluation.py --check      # dry run of the test script on VALIDATION data
```
**The test period is single-use.** `py scripts\run_test_evaluation.py` evaluates the frozen configuration once and then refuses to run again (lock file `reports/tables/TEST_RUN_DONE.lock`); `--export` re-runs it unchanged only to save predictions. Afterwards:
```powershell
py scripts\run_rolling_origin.py               # stability on 2023 folds (no validation/test data)
py scripts\run_quantile_xgb.py                 # XGBoost quantile models (intervals)
py scripts\analyze_predictions.py              # hour of day, robustness, 90% intervals
py scripts\analyze_ablation_ci.py              # paired feature-group effects with intervals
py scripts\run_quantile_lstm.py                            # LSTM quantile-head intervals
```
Exploratory extensions (each declared in the protocol first; test runs are single-use with lock files):
```powershell
py scripts\fetch_forecast_weather.py                       # archived GFS forecasts (scenario C)
py scripts\build_regional_holiday_features.py              # state-level holidays
py scripts\run_extensions_xgb.py [--with-test]             # forecast temperature and regional holidays
py scripts\run_daily_regional_study.py [--with-test]       # Zenodo daily study (direct vs bottom-up)
```
The 72 h and 168 h extension is run end to end (with commits between steps and automatic retries) by `py -I -u scripts\pipeline_runner.py scripts\chains\long_extensions.json`; its single scripts are `run_long_horizons.py`, `run_long_ablation.py`, `run_long_learning_curve.py`, `run_long_cv_tuning.py`, `run_long_lstm_tuning.py`, `run_long_intervals.py`, `run_extensions_xgb.py --long` and `analyze_long.py`.

Figures, tables and audits (no model fitting; they read the saved predictions and logs):
```powershell
py -I scripts\make_figures.py                  # figures 1-12 -> reports/figures/fig01..fig12 (PNG; fig13 comes from run_daywise.py)
py -I scripts\build_efficiency_table.py        # reports/efficiency_table.md
py -I scripts\run_daywise.py                   # day-wise view of the 24 h forecasts -> reports/daywise_test.md, fig13
py -I scripts\run_dayahead.py [--with-test]    # day-ahead model: validation phase, then the single test run (lock file)
py -I scripts\analyze_dayahead.py --split test  # day-ahead analysis and fig14 (no refitting)
py -I scripts\audit_numbers.py ; py -I scripts\audit_ci_triples.py   # numbers audits
```
`make_figures.py` needs `data/interim/predictions` (regenerate with `run_test_evaluation.py --export`, which reproduces the first run exactly).

Scripts use paths relative to the project root and have `--help` or a docstring.

## Key documents
- `docs/project_plan.md`, `docs/experiment_protocol.md`: plan, frozen protocol, deviations log.
- `reports/validation_summary.md`, `reports/test_summary.md`, `reports/research_question_answers.md`: results.
- `docs/novelty_assessment.md`: preliminary novelty check (web search only).
- `reports/long_horizons_test.md`, `reports/long_ablation_test.md`, `reports/walk_forward_long.md`, `reports/rolling_origin_long.md`, `reports/long_learning_curve.md`, `reports/long_cv_tuning_test.md`, `reports/long_lstm_tuning_test.md`, `reports/long_extensions_xgb_test.md`, `reports/long_uncertainty.md`, `reports/long_breakdowns.md`, `reports/long_importance.md`: the 72 h and 168 h extension.
- `reports/figures/fig01_...fig12_*.png`: the 12 figures for the write-up (accuracy and skill by horizon, paired differences, example week, feature ablation, walk-forward, daily multi-year, intervals, training cost, learning curves, demand and splits, study design); `reports/figures/FIGURE_CAPTIONS.md`: captions and caveats.
- `reports/efficiency_table.md`: consolidated computational-efficiency table.
- `reports/dayahead_validation.md`, `reports/dayahead_test.md`, `reports/dayahead_analysis_test.md`, `reports/figures/fig14_dayahead.png`: the operational day-ahead model (issue 10:00 on D-1; `scripts/run_dayahead.py`, `scripts/analyze_dayahead.py`).
- `reports/peak_events_test.md`, `reports/dayahead_ext_test.md`, `reports/dayahead_lstm24_test.md`, `reports/lightgbm_test.md`, `reports/dayahead_models_test.md`, `reports/other_families_test.md`, figures `fig15`-`fig17`: the completion batch (peak-exceedance classification, day-ahead variants, ablation and intervals, LightGBM; scripts `run_peak_events.py`, `run_dayahead_ext.py`, `run_dayahead_lstm24.py`, `run_lightgbm.py`, `run_neural_families.py`, `analyze_completion.py`; `run_classical_dayahead.py` and the TCN/Transformer classes are kept but were not run).
- `reports/daywise_test.md`, `reports/figures/fig13_daywise.png`: day-wise view of the 24 h-ahead forecasts (daily energy, peak, peak hour; `scripts/run_daywise.py`, no refitting).
- `reports/daily_multiyear.md`: daily multi-year walk-forward on the full Zenodo series (8 test years).
- `reports/walk_forward.md`, `reports/lstm_stage3.md`, `reports/lstm_stage3_test.md`, `reports/numbers_audit.md`: walk-forward evaluation, equal-budget LSTM, numbers audit.
- `reports/extensions_xgb_test.md`, `reports/regional_daily_study_test.md`, `reports/regional_holidays.md`: extension results.
- `reports/test_results.md`: full test tables (paired comparisons, breakdowns); `reports/uncertainty.md`, `reports/rolling_origin.md`, `reports/hour_of_day.md`, `reports/robustness.md`, `reports/ablation_paired.md`, `reports/learning_curve.md`.

## Data licences
Mendeley dataset: CC BY 4.0 (cite DOI 10.17632/y58jknpgs8.2). Open-Meteo: free for non-commercial use with attribution. See `references.md`.
