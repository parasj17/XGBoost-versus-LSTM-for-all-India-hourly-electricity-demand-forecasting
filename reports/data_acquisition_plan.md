# Data acquisition and methodology plan

Phase scope: acquisition, validation and design only. No models trained. Labels: **[V]** verified, **[A]** assumption, **[R]** recommendation.

## 1. Target definition and assembly strategy

**[V]** The Mendeley dataset has one all-India series; there are no regional columns. Therefore:

| Option | Feasible with the hourly data? | Verdict |
|---|---|---|
| 1. Regional forecasts → aggregate to India | **No at hourly resolution** — no regional hourly demand exists. **Possible at daily resolution** with the Zenodo state-level energy data (separate study, section 9) | Not for the hourly study |
| 2. India-wide target | **Yes** — `NLDC_DEMAND` / `Demand (MW)` is already the national value; **nothing is summed by us** | **Use this** |

**[R] Target:** `electricity_demand_mw` = all-India HH:00 demand in MW, cleaned, 2021-09-01 → 2025-06-24.
**[R] Assembly:** (1) clean each monthly workbook → HH:00 snapshot; (2) concatenate chronologically with a `source_regime` flag; (3) left-join ERA5-derived national weather on `timestamp`; (4) left-join calendar features; (5) save to `data/processed/`; (6) no random shuffling anywhere.
**[R]** Sensitivity analyses: hourly-mean target for 2021–2023 (audit section 6), and equal-vs-weighted weather aggregation.

## 2. Weather: defensible geographic representation

**Constraint:** the target is national, but weather varies enormously across India (Himalaya, Thar, Indo-Gangetic plain, coast, Deccan, North-East). One city is not representative; I do **not** use a single-city proxy.

**Proposal (pilot in `scripts/inspect_weather_data.py`; full download not yet run):**
- 12 cities across the five grid regions, chosen to sit in major load centres: Northern (Delhi, Lucknow, Jaipur), Western (Mumbai, Ahmedabad, Nagpur), Southern (Chennai, Bengaluru, Hyderabad), Eastern (Kolkata, Patna), North-Eastern (Guwahati).
- Source: Open-Meteo archive, `models=era5`, hourly, `timezone=Asia/Kolkata`. Variables: temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, wind_speed_10m, cloud_cover, shortwave_radiation.
- Aggregation: (a) mean within region, (b) national weighted mean across regions. **Weights are not yet decided** and must come from a documented source — candidates: Census 2011 population per region, or each region's share of electricity consumption (CEA/Grid-India). Until chosen, only an equal-weight baseline is defensible. (c) Sensitivity: equal vs population vs consumption weights; national mean vs region-level features (5 temperature columns) for XGBoost.
- Do not weight by the target's own demand in a way that uses test-period data.

**Known limitations [A]:** cities are urban points, not area averages; ERA5 is 0.25° (~25 km); the 12-city set is my selection and is not validated against load-weighted temperature; ERA5 is *reanalysis* (observation-like), not a forecast (section 6). Coordinates are approximate city centres. Open-Meteo non-commercial terms/attribution apply.

**Alternatives to evaluate later:** (i) a gridded national mean weighted by population (removes city-selection bias, more API calls); (ii) re-aggregating the Zenodo per-state daily weather — but that is daily and ends 2024-04, so it cannot supply the hourly weather.

## 3. Calendar and festival plan

- Time features: done in `scripts/build_calendar_features.py` (hour, day, day-of-week, day-of-year, month, weekend, IMD-style season, sin/cos encodings).
- **Holidays:** 85 central-government gazetted holidays for 2021–2025, transcribed from DoPT office memoranda (see `references.md`; each CSV row carries its source document and URL). Fixed national days: Republic Day, Independence Day, Gandhi Jayanti (`national_fixed`); others `central_gazetted_festival`.
- **Limitations:** (a) this is the *Delhi central-government* list, not all regional holidays; (b) lunar dates (Id, Muharram) are planned dates that may shift by a day with moon sighting; (c) all five yearly lists (85 rows) were verified against Annexure-I of the DoPT memoranda (the web page's 'March 29' for Maha Shivratri 2022 was a typo; 1 March is correct); (d) `india.gov.in/calendar` returned HTTP 403 and was not used; (e) restricted/optional and state holidays are not included.
- Do not claim regional festivals (Pongal, Onam, Durga Puja, Chhath, Bihu) are captured; they are not in this list. Adding a small cited set of regional major festivals is a recommended optional step.

## 4. Quality handling

Cleaning rules are proposed in `data_dictionary.md`. Freeze them before touching the test period and report every imputation. The 2021-10-08 → 10-09 eight-hour outage and similar longer gaps should be excluded, not interpolated.

## 5. Evaluation design (to implement later)

- **Horizons:** 1 h, 6 h, 24 h, 48 h (longer only if weather-forecast data justify it).
- **Splits (chronological, no shuffle) [R]:** train 2021-09-01 → 2023-12-31; validation 2024-01-01 → 2024-06-30; test 2024-07-01 → 2025-06-24 (a full 12-month cycle incl. Diwali 2024 and Holi 2025). Also run rolling-origin (expanding-window) evaluation to check stability. A final refit on train+validation with frozen hyper-parameters is optional and must be declared.
- **Baselines (mandatory):** persistence, seasonal-naive 24 h and 168 h, and a linear/ridge model. Never claim a model is "better" without these plus uncertainty on the difference (e.g. Diebold–Mariano or block-bootstrap CIs).
- **Metrics:** MAE, RMSE, MAPE/sMAPE, per horizon, plus error by season, hour, holiday vs non-holiday, and peak hours.
- **Non-stationarity:** mean demand grew about 7–9 % per year (149.9 → 198.3 GW). Tree models cannot extrapolate beyond the training range, and test-period demand is higher than training. Consider modelling relative to a lagged level (ratio/difference to demand at t or the same hour last week) and report it explicitly.

## 6. Leakage and forecast-time information rules

At forecast origin `t`, predicting `y(t+h)`:

| Information | Allowed? |
|---|---|
| Demand/solar/wind **observed at or before t** (lags, rolling stats using data ≤ t) | Yes |
| Demand/solar/wind at t+1…t+h | **Never** (actual future solar/wind are outcomes; only a documented forecast available at time t may be used) |
| Calendar/holiday features for t+h | Yes — deterministic/published in advance |
| ERA5 weather at ≤ t | Yes |
| ERA5 weather at t+h (reanalysis "actuals" in the future) | **Only as a labelled "perfect-weather-forecast" scenario** — not information available at time t |
| Real NWP forecasts issued at t | Needed for a deployable scenario; verified 2026-10-13: the Historical Forecast API only stitches short-lead data (do not use for 24/48 h leads); the Previous Runs API (GFS) gives day-1/day-2 lead forecasts for the whole period except ≈ 500 h in Dec 2023 – Jan 2024 and shows a systematic bias vs ERA5 — see plan decision D-13 |

[R] Report (A) **no future weather** (realistic baseline), (B) **oracle weather at t+h** (upper bound), and where possible (C) archived forecast weather. Rolling features use `shift` so row `t` only uses data ≤ t; scalers/imputers/thresholds fit on the training period only; hyper-parameter search on the validation period only; LSTM windows must not leak across split boundaries (the target window lies entirely within its split). Missing-data handling and assumptions go in an experiment log. For the 48 h horizon, lag-24 is available only if it lies ≥ 24 h before the target.

## 7. Uncertainty estimation (future work; none built)

| Approach | XGBoost | LSTM | Limitations |
|---|---|---|---|
| Quantile regression | `reg:quantileerror`, one model per quantile (e.g. 5/50/95) | Multi-quantile output head with pinball loss | Quantile crossing; coverage not guaranteed |
| Conformal prediction | Split/adaptive conformal on validation residuals | Same | Assumes exchangeability; drift violates it — use rolling/adaptive conformal and check coverage by season |
| Residual / block bootstrap | Resample residual blocks | Same | Assumes stationary errors; underestimates during festivals/heatwaves |
| MC dropout / deep ensembles | – | Dropout at inference, or several seeds | MC dropout is often poorly calibrated; ensembles multiply compute |
| Seed/subsample ensembles | Spread of an ensemble | – | Reflects model variance only, not noise |

Evaluate with PICP (coverage), mean interval width, pinball loss / Winkler score, and calibration by horizon and by hour.

## 8. Computational-efficiency protocol

Record for every run, machine-readably: hardware (CPU model, cores/threads used, RAM, GPU model/driver), OS, Python and library versions (xgboost, torch/tensorflow, CUDA), random seeds, full hyper-parameters, number of features, training rows, early-stopping rounds/epochs actually used, **training wall-clock and CPU time** (data loading reported separately), peak memory, model size / parameter count, **inference latency** (batch and single-sample, after warm-up, median and IQR over ≥ 30 repeats), and the tuning budget (equal number of trials/time for both models). Compare on the same hardware with the same thread count, on CPU and (separately) GPU if used. Also record whether a model is direct (one per horizon), recursive or multi-output, since that changes total cost. No claim about relative efficiency before measured results exist.

## 9. Alternative datasets

- **Zenodo 14983362 (daily; state-level energy 2013-01-01 → 2024-04-28; per-state daily ERA5-type weather 1979 → 2024-04-27):** partly verified (details in `dataset_inventory.md` section 6). It is **daily, in MU (energy)**, with 55 duplicate dates and 236 missing days; its national UTCI file ends 2020-12-31. Uses: (i) **validation** of the Mendeley national series (done: correlation 0.997, MAPE 0.57 % over 955 days); (ii) a **long-history daily study** (11 years, many festival cycles) with the state-level structure Mendeley lacks — a legitimate *separate* daily experiment, e.g. regional daily forecasts aggregated to India (approach 1); (iii) a template for population-weighted weather. **Not mixed into the hourly model.** It ends April 2024, so it cannot extend the hourly test period. Licence is CC BY 4.0 (Hunt & Bloomfield 2025); the `corrected_*` files are the authors' simple transcription-error check. Still unverified: the per-state weather processing. All 42 files downloaded and MD5-verified; the `corrected_*` energy file is the one to use.
- **NITI Aayog ICED:** dashboard exists; some charts show "Download XLS", plus a data-request form. No raw data export verified; do not rely on it until a downloaded file has been inspected.

## 10. Commands for you to run

```powershell
cd D:\LOR_Research_SmartGrid
py -m pip install -r requirements.txt
py scripts\inspect_electricity_data.py            # ~35 min first time (parses 20 large workbooks); cached afterwards
py scripts\build_calendar_features.py
py scripts\inspect_weather_data.py --pilot        # 3-day pilot, saves nothing
py scripts\inspect_weather_data.py --fetch --start 2021-09-01 --end 2025-06-24   # after you approve the city list
```

## 11. Next five steps before any model training

1. Decide/approve the weather city list and aggregation weights (document the weight source); then run the full ERA5 fetch.
2. Freeze and implement the cleaning rules (`scripts/build_hourly_dataset.py`), producing `data/processed/india_hourly_canonical.csv` plus a cleaning log with counts per rule; check before/after plots.
3. Verify the holiday CSV against the official DoPT PDFs (especially the flagged 2022 entry) and decide whether to add a small cited list of regional festivals.
4. Decide whether it is used for daily validation only or also a separate daily regional study.
5. Write the experiment protocol (splits, baselines, scenarios A/B/C, metrics, logging schema) and run exploratory analysis on the **training period only**; then implement baselines before XGBoost/LSTM.
