# Limitations that apply to every result in this project

This list is meant to be read before any number is quoted. It is repeated, in short form, in the README and at the top of `reports/research_question_answers.md`. Nothing in the results should be quoted without these caveats.

## 1. Only one test year
- The hourly dataset covers 2021-09-01 to 2025-06-24 (about 3.8 years). After about 2.3 years for training and half a year for validation, the test period is **one year** (2024-07-01 to 2025-06-24).
- The confidence intervals in the reports describe hour-to-hour sampling noise inside that year. They do **not** describe year-to-year variation, so they understate the uncertainty about how the models would rank in another year.
- Each season appears once in the test year; festivals appear in only a few days (about 15 days within one day of a major festival).
- Partial remedies, none of which removes the limitation: two extra rolling-origin folds on 2023, a walk-forward evaluation over 2024-25 (whose first two quarters lie in the validation period and are not independent), and a separate daily study on a different dataset.

## 2. Validation numbers are optimistic
- The validation period (2024-H1) was used for hyper-parameter tuning, early stopping, model selection and conformal calibration. Validation results are therefore better than what an unseen period gives. The test numbers are the honest ones, and they are indeed higher for the models tuned on validation.

## 3. The model comparison depends on tuning effort
- XGBoost's headline set was tuned with about 90 trials per horizon; the LSTM headline had 12 trials in the first tuning (with 10-epoch trials). An equal-budget LSTM (stage 3, 90 trials) was added afterwards; see `reports/lstm_stage3.md` and the answers document. Statements about "LSTM versus XGBoost" apply to this setup, three seeds, one laptop CPU, a 168-hour input window and a small recurrent network.
- Ridge was searched exhaustively over its small space (4 variants x 7 penalties).
- The LSTM varies noticeably from seed to seed (for example 2,904 to 3,335 MW at 6 h), so a single LSTM run can mislead.

## 4. The test period was looked at more than once
- The frozen point-forecast evaluation ran once and was reproduced exactly. Afterwards the test period was used again for: intervals, an unchanged re-run to save predictions, hour-of-day and robustness analyses, paired feature-group effects, extensions (forecast temperature, regional holidays, LSTM quantile head), and the equal-budget LSTM. Each use was declared in the protocol (`docs/experiment_protocol.md`, deviation log) before it ran, and results are reported whichever way they came out.
- Several analyses were chosen **after** the first test results had been seen. Treat the secondary findings as exploratory. Many intervals are reported without multiple-comparison correction.

## 5. Data and input issues
- The cleaning thresholds were chosen looking at the whole series (including the later years), from physical plausibility and hour-to-hour statistics, not from forecast performance.
- The target is the demand reading at the start of each hour. The 2024-25 hourly values are assumed to be of the same kind, on indirect evidence only (a mean-trained model scored worse on them).
- The three sampling regimes (5-minute, 10-second, hourly) could not be tested against each other; the whole test period is in the hourly regime.
- Weather is ERA5 reanalysis, not what an operator would have had in real time. The archived forecasts used in the forecast-temperature extension exist only for temperature over the full period, their exact issue time was not verified (so the gain may be slightly optimistic), and about 10% of the validation hours have no forecast.
- The regional holiday source (`holidays` package) is incomplete for multi-day regional festivals and its state calendars were not checked against state gazettes.
- Solar and wind capacity grew strongly (15% of test hours exceed the training maximum of solar generation), which probably explains why solar/wind inputs did not help. Capacity-normalised features were not tried.

## 6. Statistics
- Block-bootstrap intervals use 168-hour blocks; Diebold-Mariano statistics use a short lag and are overconfident because hourly errors are strongly autocorrelated. The bootstrap interval is the primary evidence.
- "Not distinguishable" means the interval includes zero, which is weaker than "no difference".

## 7. Process
- The analysis was carried out with an AI assistant. Several wrong numbers or wordings were caught and fixed along the way (for example a mis-stated baseline value and two overstated ranges), and a numbers audit (`reports/numbers_audit.md`) compares the summary documents with the result files. An independent human review of the protocol and its deviation log is recommended before submission.
- Everything ran on one laptop CPU; efficiency numbers are specific to that machine and were measured while other work was sometimes running.

## 8. What was not done
- Longer horizons (72 h, 168 h), gridded weather, EV features, MC dropout, an LSTM in the daily study, comparison with operational Grid-India forecasts, a systematic literature search (only a preliminary web search exists, `docs/novelty_assessment.md`), and no claim of novelty is made.
