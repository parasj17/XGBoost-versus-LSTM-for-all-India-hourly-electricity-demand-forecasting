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

---

# Pre-test stage 2 (2026-10-15; specified in the protocol deviation log before it was run)

XGBoost: 60 further random trials per horizon with widened ranges, plus the stage-1 best re-fitted as a competitor (`reports/tables/xgboost_validation_wide.md`). LSTM: the history+calendar variant (L1) tuned on its own, 12 trials per horizon, 3 final seeds (`reports/tables/lstm_validation_stage2_L1.md`). All selection used validation only. The test period was not loaded.

| Model (validation MAE, MW) | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| Best ridge | 1,089 | 2,941 | 3,763 | 4,663 |
| XGBoost stage 1 | 1,104 | 2,541 | 3,174 | 4,247 |
| **XGBoost stage 2 (pre-declared final headline)** | 1,104 | **2,504** | **3,170** | **4,247** |
| LSTM full features (stage-1 headline, unchanged) | 1,584 | 3,048 | 3,273 | 4,778 |
| LSTM history+calendar, stage 1 | 1,503 | 2,619 | 3,300 | 4,392 |
| LSTM history+calendar, own tuning (stage 2) | 1,485 | 2,696 | 3,278 | 4,337 |

Stage-2 paired intervals (MAE difference, negative = first model better; 95% block bootstrap):
- XGBoost stage 2 minus best ridge: 1 h +15 (-14 to +44); 6 h -437 (-520 to -317); 24 h -592 (-773 to -452); 48 h -416 (-678 to -87).
- LSTM L1 (own tuning) minus best ridge: 1 h +396 (+340 to +473); 6 h -245 (-398 to -60); 24 h -484 (-698 to -301); 48 h -325 (-509 to -133).
- LSTM L1 (own tuning) minus XGBoost stage 2: 1 h +381 (+331 to +440); 6 h +192 (+19 to +356); 24 h +108 (-113 to +370); 48 h +90 (-237 to +397).

## What stage 2 changed
- The widened XGBoost search changed the chosen configuration at 6 h (-1.5%) and 24 h (-0.1%); at 1 h and 48 h the stage-1 configuration won. At 24 h the new best (depth 10, strong regularisation) is very different from the old (depth 7) yet scores the same, i.e. a flat optimum. XGBoost was not badly under-tuned in stage 1.
- Own tuning of the LSTM L1 variant did not change the picture (differences of -75 to +77 MW against stage 1, inside seed noise).
- Conclusions that survive: XGBoost beats ridge at 6, 24 and 48 h; the two are tied at 1 h; the LSTM is clearly worse than XGBoost at 1 h and 6 h (the 6 h interval now excludes zero for L1) and **not distinguishable from XGBoost at 24 h and 48 h**.

## Cost of stage 2 (4-thread CPU)
XGBoost: 240 + 4 tuning fits in 43.5 min and 144 ablation fits in 27.7 min. LSTM L1: 48 tuning trials in 47 min and 12 final fits (about 26 min of the 73.5 min total run).
Total tuning budget now: XGBoost 90 trials per horizon, LSTM 12 (L4) + 12 (L1) per horizon.
