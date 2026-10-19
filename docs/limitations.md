# Limitations that apply to every result in this project

This list is meant to be read before any number is quoted. It is repeated, in short form, in the README and at the top of `reports/research_question_answers.md`. Nothing in the results should be quoted without these caveats.

## 1. Only one test year
- The hourly dataset covers 2021-09-01 to 2025-06-24 (about 3.8 years). After about 2.3 years for training and half a year for validation, the test period is **one year** (2024-07-01 to 2025-06-24).
- The confidence intervals in the reports describe hour-to-hour sampling noise inside that year. They do **not** describe year-to-year variation, so they understate the uncertainty about how the models would rank in another year.
- Each season appears once in the test year; festivals appear in only a few days (about 15 days within one day of a major festival).
- Partial remedies, none of which removes the limitation: two extra rolling-origin folds on 2023, a walk-forward evaluation over 2024-25 (whose first two quarters lie in the validation period and are not independent), and a daily multi-year walk-forward on the Zenodo national daily series with eight test years (2017-2024; daily resolution and two horizons only, hyper-parameters tuned on two of the years, one year is the COVID shock). The hourly results still rest on one test year.

## 2. Validation numbers are optimistic
- The validation period (2024-H1) was used for hyper-parameter tuning, early stopping, model selection and conformal calibration. Validation results are therefore better than what an unseen period gives. The test numbers are the honest ones, and they are indeed higher for the models tuned on validation.

## 3. The model comparison depends on tuning effort
- XGBoost's headline set was tuned with about 90 trials per horizon; the LSTM headline had 12 trials in the first tuning (with 10-epoch trials). An equal-budget LSTM (stage 3, 90 trials per horizon) was added afterwards (`reports/lstm_stage3.md`, `reports/lstm_stage3_test.md`): it improved the 1 h result by 6% on test and left the other horizons unchanged within noise, so the comparison did not change. Two caveats: 55 of the 78 new trials at 1 h were stopped early by the pruning rule, so the budget is equal in trial count but not in compute there; and the selected networks peak on validation after only 1 to 5 epochs at 6 h and 48 h. Statements about "LSTM versus XGBoost" apply to this setup, three seeds, one laptop CPU, a 168-hour input window and a small recurrent network.
- Ridge was searched exhaustively over its small space (4 variants x 7 penalties).
- The LSTM varies noticeably from seed to seed (for example 2,904 to 3,335 MW at 6 h), so a single LSTM run can mislead.
- At 72 h and 168 h the validation-selected configurations were very short (XGBoost 2 trees and the LSTM 1 epoch at 168 h). Cross-validated tuning inside the training period did not change the test result, and at 168 h ridge is the best model while XGBoost and the LSTM are only about 3-4% better than the naive baseline; the cause (for example a shift between 2021-2023 and 2024) was not tested.

## 4. The test period was looked at more than once
- The frozen point-forecast evaluation ran once and was reproduced exactly. Afterwards the test period was used again for: intervals, an unchanged re-run to save predictions, hour-of-day and robustness analyses, paired feature-group effects, extensions (forecast temperature, regional holidays, LSTM quantile head), the equal-budget LSTM, the extension to 72 h and 168 h (the horizon comparison, feature ablation, walk-forward, rolling origin, cross-validated tuning, equal-budget LSTM, forecast temperature and regional holidays, and prediction intervals, each with its own single test run) the walk-forward and daily multi-year studies, the day-wise re-aggregation of the 24 h predictions, the operational day-ahead model, and the completion batch (peak-exceedance, day-ahead variants, LightGBM, GRU). Each use was declared in the protocol (`docs/experiment_protocol.md`, deviation log) before it ran, and results are reported whichever way they came out.
- Several analyses were chosen **after** the first test results had been seen. Treat the secondary findings as exploratory. Many intervals are reported without multiple-comparison correction.

## 5. Data and input issues
- The cleaning thresholds were chosen looking at the whole series (including the later years), from physical plausibility and hour-to-hour statistics, not from forecast performance.
- The target is the demand reading at the start of each hour. The 2024-25 hourly values are assumed to be of the same kind, on indirect evidence only (a mean-trained model scored worse on them).
- The three sampling regimes (5-minute, 10-second, hourly) could not be tested against each other; the whole test period is in the hourly regime.
- Weather is ERA5 reanalysis, not what an operator would have had in real time. The archived forecasts used in the forecast-temperature extension exist only for temperature over the full period, their exact issue time was not verified (so the gain may be slightly optimistic), and about 10-14% of the validation hours have no forecast (10-11% at 1-48 h, 11% at 72 h and 14% at 168 h).
- The regional holiday source (`holidays` package) is incomplete for multi-day regional festivals and its state calendars were not checked against state gazettes.
- Solar and wind capacity grew strongly (15% of test hours exceed the training maximum of solar generation), which probably explains why solar/wind inputs did not help. Capacity-normalised features were not tried.

## 6. Statistics
- Block-bootstrap intervals use 168-hour blocks; Diebold-Mariano statistics use a short lag and are overconfident because hourly errors are strongly autocorrelated. The bootstrap interval is the primary evidence.
- "Not distinguishable" means the interval includes zero, which is weaker than "no difference".

## 7. Process
- The analysis was carried out with an AI assistant. Several wrong numbers or wordings were caught and fixed along the way (for example a mis-stated baseline value, overstated ranges, an inference-cost ratio, a learning-curve statement, daily-study percentages and a refit-cost ratio; the later ones are listed in the protocol deviation log, entry of 2026-10-17), and a numbers audit (`reports/numbers_audit.md`) compares the summary documents with the result files. An independent human review of the protocol and its deviation log is recommended before submission.
- Everything ran on one laptop CPU; efficiency numbers are specific to that machine and were measured while other work was sometimes running.

## 8. What was not done
- The day-ahead model (task 9.7) pools XGBoost and the LSTM over the 24 target hours and uses scenario-A inputs only (no forecast weather); a per-hour XGBoost, a 24-output LSTM and forecast-weather inputs were not run, and its issue time (10:00) and the availability of the 10:00 demand value are design assumptions.
- Other model families: the main study evaluated XGBoost, the LSTM, ridge and naive baselines. The completion batch of 2026-10-19/20 added LightGBM (six horizons and the day-ahead task) and a GRU (1 h, 24 h and the day-ahead task, a smaller tuning budget than the LSTM at the main horizons). Convolutional (TCN) and attention-based (Transformer) networks and statistical models (SARIMAX, Prophet) were not evaluated: they were dropped by the owner before any run, and the code of these models is in the repository. The conclusions are about the evaluated families. Horizons beyond 168 h, gridded weather, EV features, MC dropout, an LSTM in the daily study, comparison with operational Grid-India forecasts, a systematic literature search (only a preliminary web search exists, `docs/novelty_assessment.md`), and no claim of novelty is made.
