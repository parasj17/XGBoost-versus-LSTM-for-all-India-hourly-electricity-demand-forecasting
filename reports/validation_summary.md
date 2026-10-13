# Validation-period summary: baselines, XGBoost, LSTM (2026-10-14)

Validation targets 2024-01-01 to 2024-06-30 (4,367 valid samples per horizon, identical for all models). Scenario A (no future weather). MAE in MW. The test period (2024-07-01 onward) has not been used. All numbers are mildly optimistic because early stopping and hyper-parameter selection used the validation period; XGBoost had 30 tuning trials per horizon, the LSTM 12 (documented in the protocol deviation log), so the comparison is not budget-equal.

| Model | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| Best naive (persistence / same hour yesterday / two days ago) | 4,430 | 4,931 | 4,931 | 6,187 |
| Best ridge | 1,089 | 2,941 | 3,763 | 4,663 |
| **XGBoost, full features (pre-declared headline)** | 1,104 | **2,541** | **3,174** | **4,247** |
| LSTM, full features (pre-declared headline, 3-seed mean) | 1,584 | 3,048 | 3,273 | 4,778 |
| LSTM, history + calendar (3-seed mean) | 1,503 | 2,619 | 3,300 | 4,392 |

Sources: `reports/tables/baselines_validation.md`, `xgboost_validation.md`, `lstm_validation.md` (paired block-bootstrap intervals are in those files).

## What the paired intervals say
- 1 h: ridge and XGBoost tie (+15 MW, CI -14 to +44); the LSTM is clearly worse than both (+399 to +495 MW, intervals exclude 0).
- 6 h: XGBoost beats ridge (-400, CI -507 to -259). LSTM history+calendar beats ridge (-322, CI -458 to -164) and is **not distinguishable** from XGBoost (+78, CI -66 to +216). LSTM full features is worse than XGBoost (+506, CI +321 to +713).
- 24 h: XGBoost beats ridge (-589, CI -772 to -447); every LSTM variant beats ridge (-421 to -490, intervals exclude 0) and none is distinguishable from XGBoost (+99 to +168, intervals include 0).
- 48 h: XGBoost beats ridge (-416, CI -678 to -87). LSTM history+calendar beats ridge (-271, CI -433 to -91) and is not distinguishable from XGBoost (+145, CI -181 to +466); LSTM full features is worse than XGBoost (+531, CI +257 to +814).

## Reading the results honestly
- XGBoost has the lowest point estimate at every horizon from 6 h on, but the LSTM history+calendar variant is statistically indistinguishable from it at 6, 24 and 48 h on 4,367 samples of one half-year; "XGBoost is better" is supported only against ridge and against the LSTM full-feature variant at 6 h and 48 h.
- Adding weather and solar/wind **hurt** the LSTM at 6 h and 48 h but helped XGBoost at 24 h and 48 h. One untested hypothesis is that growing solar/wind capacity pushes 2024 inputs outside the training range, which a small LSTM extrapolates poorly.
- The LSTM hyper-parameters were tuned on the full-feature set, so the history+calendar variants may be slightly pessimistic.
- Holiday features gave no measurable gain for either model on validation, and the validation period contains no Diwali; this stays open for the test period.

## Computational cost (4-thread laptop CPU, no GPU)
| | XGBoost | LSTM |
|---|---:|---:|
| Tuning trials (total, all horizons) | 120 trials in 10.1 min (about 5 s each) | 48 trials in 53.1 min (about 66 s each) |
| Final/ablation fits | 144 fits in 16.6 min (4-15 s per fit for the headline set) | 36 fits in 75.0 min (32-277 s per fit for the headline set) |
| Model size | 0.5-1.6 MB | 43-111 KB (10-27 thousand parameters) |
| Peak working set | about 410 MB | about 730 MB |
| Inference | 22-56 ms for 4,367 rows | 60-164 ms for 1,024 windows (roughly 10-30 times more per sample) |
Single-sample latencies (XGBoost 12.6-13 ms, LSTM 1.2-1.9 ms) were measured through different Python interfaces and are dominated by overhead, so they are not comparable.
