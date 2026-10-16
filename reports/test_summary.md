# Test-period summary (single run, 2026-10-15)

Source: `reports/test_results.md` (full tables, paired intervals, breakdowns), `reports/tables/test_results.jsonl`, run at commit `155b206`, frozen configuration in `docs/experiment_protocol.md` section 15. Targets 2024-07-01 to 2025-06-24, 8,611 valid hours per horizon. Models fitted on the training period only. The test set was evaluated once; nothing was changed afterwards.

## Headline (MAE, MW; mean prediction of 3 seeds)
| Model | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| Best naive | 4,272 | 5,414 | 5,414 | 6,964 |
| Ridge | **1,049** | 3,082 | 4,214 | 5,549 |
| XGBoost S4_full_A (declared headline) | 1,111 | **2,747** | **3,728** | **5,158** |
| LSTM L4_full_A (declared headline) | 1,643 | 3,170 | 3,743 | 5,325 |
| LSTM L1 history+calendar (own tuning) | 1,527 | 2,981 | 3,753 | 5,280 |

## Paired differences (MAE difference in MW, 95% moving-block bootstrap CI; negative = first better)
| Comparison | 1 h | 6 h | 24 h | 48 h |
|---|---|---|---|---|
| XGBoost - ridge | +62 (-14 to +202) | -335 (-416 to -237) | -486 (-615 to -354) | -391 (-647 to -153) |
| LSTM L4 - XGBoost | +532 (+394 to +633) | +423 (+305 to +537) | +14 (-110 to +105) | +167 (-101 to +429) |
| LSTM L1 - XGBoost | +417 (+285 to +511) | +234 (+127 to +353) | +24 (-81 to +154) | +122 (-118 to +444) |
| LSTM L1 - ridge | +479 (+442 to +530) | -101 (-227 to +58) | -462 (-608 to -297) | -269 (-386 to -121) |

## What the test period supports
- XGBoost beats ridge at 6, 24 and 48 h (intervals exclude zero). At 1 h XGBoost and ridge are not distinguishable by the bootstrap interval; the point estimate favours ridge (the Diebold-Mariano statistic points the same way but understates uncertainty because hourly errors are strongly autocorrelated). At 1 h XGBoost's RMSE (2,140 MW) is much larger than ridge's (1,562 MW) while MAE is similar, so XGBoost has some larger errors; in winter and pre-monsoon its 1 h error is above ridge's (e.g. winter 1,574 vs 1,205 MW). The cause was not investigated.
- The LSTM is worse than XGBoost at 1 h and 6 h (intervals exclude zero) and **not distinguishable from XGBoost at 24 h and 48 h**. Both LSTM variants beat ridge at 24 h and 48 h; at 6 h they are not distinguishable from ridge.
- Skill against the best naive baseline (ridge, XGBoost and the LSTMs): about 74-75% at 1 h (ridge and XGBoost; 62-64% for the LSTMs), 41-49% at 6 h, 22-31% at 24 h and 20-26% at 48 h.

## Validation versus test
| Model (MAE, MW) | 1 h val -> test | 6 h | 24 h | 48 h |
|---|---|---|---|---|
| Ridge | 1,089 -> 1,049 | 2,941 -> 3,082 | 3,763 -> 4,214 | 4,663 -> 5,549 |
| XGBoost headline | 1,104 -> 1,111 | 2,504 -> 2,747 | 3,170 -> 3,728 | 4,247 -> 5,158 |
| LSTM L4 | 1,584 -> 1,643 | 3,048 -> 3,170 | 3,273 -> 3,743 | 4,778 -> 5,325 |
| LSTM L1 (own tuning) | 1,485 -> 1,527 | 2,696 -> 2,981 | 3,278 -> 3,753 | 4,337 -> 5,280 |
Errors rise from validation to test at 6-48 h for every model (XGBoost +10%, +18%, +21%), which is expected for tuned models, but also reflects a harder period (the best naive baseline got worse too: 48 h 6,187 -> 6,964, +13%). The ranking of the models is largely preserved; the gap between XGBoost and ridge at 24 h and 48 h shrank slightly (-592 -> -486 and -416 -> -391 MW).

## Findings that did NOT carry over from validation
- On validation, past weather and solar/wind lowered XGBoost's 24 h and 48 h error by about 4-5%. On test they do not: S2 (holidays) -> S3 (+ past weather) -> S4 (+ solar/wind): 24 h 3,705 -> 3,713 -> 3,728; 48 h 5,157 -> 5,139 -> 5,158 MW. Past weather and generation add nothing measurable on the test period.
- **Oracle** weather (the actual weather at the target hour, not available in reality) does help: 24 h 3,728 -> 3,510 MW (-5.9%), 48 h 5,158 -> 4,564 MW (-11.5%). A perfect weather forecast would therefore be worth something at longer horizons; past weather is not.
- Holiday/festival features (S1 -> S2): 6 h 2,738 -> 2,705, 24 h 3,784 -> 3,705, 48 h 5,213 -> 5,157 MW, i.e. 1-2% gains, consistent in sign across horizons; no gain at 1 h. Error within one day of a major festival is clearly elevated: for XGBoost 1.3 to 1.7 times the overall error at 6-48 h (e.g. 24 h 6,214 MW vs 3,728 MW overall; not elevated at 1 h); n = 360 hours, about 15 days, so this is a small sample. At 48 h the LSTMs and ridge do much worse around festivals (9,000-10,100 MW, 1.7-1.9 times) than XGBoost (6,524 MW).
- The LSTM per-seed spread is wide at 6 h (L1: 2,904-3,335 MW), i.e. the headline mean-of-seeds number hides large run-to-run variation.

## Equal-budget LSTM (added 2026-10-16; declared in the protocol before it was run)
The first LSTM tuning had 12 trials per horizon against about 90 for XGBoost. The LSTM L4 headline was therefore re-tuned with 90 trials per horizon in total (`reports/lstm_stage3.md`; selection on validation only), then evaluated once on the test period with the same fitting rule (`reports/lstm_stage3_test.md`).

| Test MAE (MW) | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| LSTM L4, equal tuning budget | 1,549 | 3,151 | 3,776 | 5,418 |
| LSTM L4, original (12 trials) | 1,643 | 3,170 | 3,743 | 5,325 |
| XGBoost S4 | 1,111 | 2,747 | 3,728 | 5,158 |
| Equal-budget minus original (95% CI) | -94 (-126 to -64) | -19 (-74 to +53) | +33 (-21 to +88) | +94 (-47 to +242) |
| Equal-budget minus XGBoost (95% CI) | +438 (+300 to +532) | +404 (+310 to +506) | +47 (-63 to +128) | +260 (-55 to +563) |

The extra tuning improved the LSTM only at 1 h (-6%); at 6 h, 24 h and 48 h the change is not distinguishable from zero, and the validation gains at 24 h and 48 h (-1% and -3%) did not carry over to the test year. The comparison with XGBoost is unchanged: the LSTM is clearly worse at 1 h and 6 h and not distinguishable from XGBoost at 24 h and 48 h. Caveats: 55 of the 78 new trials at 1 h were stopped early by the pruning rule (almost none at the other horizons), so the budget is equal in trial count but not in compute at 1 h; the selected configurations stop improving on validation after only 1 to 5 epochs at 6 h and 48 h, which suggests fast over-fitting to the 2021-2023 pattern.

## Caveats
- 28.8% of test hours are at or above the training 95th percentile (demand has grown), so the "peak hours" subset is a large share of the test period and is not a rare-event measure.
- Tuning budgets differ (XGBoost 90 trials per horizon; LSTM 12 + 12) and the LSTM was tuned for 10-epoch trials; any statement about "LSTM vs XGBoost" applies to this setup.
- One test year, one hardware setup, one train/validation/test split; rolling-origin folds on 2023 are in `reports/rolling_origin.md` (the XGBoost-vs-ridge ranking at 6 h flips between folds).
- Refit times (3 seeds, laptop CPU) were 5-62 s for XGBoost S4 and 35-522 s for LSTM L4 (L1: 113-953 s); LSTM refits cost roughly 4 to 45 times more depending on horizon and variant.
