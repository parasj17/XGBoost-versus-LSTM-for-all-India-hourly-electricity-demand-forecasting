# Data dictionary (proposed canonical dataset)

Proposed file: `data/processed/india_hourly_canonical.csv` — **not built yet** (to be produced after the cleaning rules are agreed; see `data_acquisition_plan.md`). Grain: one row per clock hour, IST, 2021-09-01 00:00 → 2025-06-24 23:00 (33,432 rows). No regional key exists because the source has no regional data.

Status column: **have** = derivable from data already in the workspace; **external** = needs data not yet downloaded; **derived** = computed from other columns.

| Final column | Type / unit | Definition | Source column(s) (original) | Status | Known at forecast time t? |
|---|---|---|---|---|---|
| `timestamp` | datetime, IST (naive, UTC+05:30), hour start | Clock hour | Layout A `Time` (value at HH:00); Layout B `Timestamp` (`dd-mm-yyyy HH:MM:SS`) | have | – (index) |
| `electricity_demand_mw` | float, MW | All-India demand, instantaneous SCADA reading at HH:00, cleaned (see rules below). **MW is a power unit, not MWh**; if MWh/hour is ever needed, 1 h × MW = MWh numerically but state it explicitly | A: `NLDC_DEMAND\|P`; B: `Demand (MW)` | have | value ≤ t: yes; value > t: this is the target |
| `demand_raw_mw` | float, MW | Uncleaned value as in the source (kept for audit) | same | have | as above |
| `demand_quality_flag` | category | `ok`, `zero_outage`, `spike`, `dip_recovery`, `stuck`, `imputed` | derived from rules | derived | – |
| `solar_generation_mw` | float, MW | All-India solar generation, HH:00 reading, negatives clipped at 0 and night offset documented | A: `ALL_IND_SOLAR\|P` (raw; clipped = `Solar`); B: `Solar (MW)` | have | **only values ≤ t**; future solar is an outcome, not a predictor |
| `wind_generation_mw` | float, MW | All-India wind generation, HH:00 reading | A: `ALL_INDIA_WIND\|P`; B: `Wind (MW)` | have | only values ≤ t |
| `total_generation_mw` | float, MW | Total (thermal+hydro+gas+nuclear+solar+wind) in A; `Total Generation (MW)` in B. Optional, low priority | A: `Total`; B: `Total Generation (MW)` | have | only values ≤ t |
| `source_regime` | category | `5min`, `10s`, `hourly` — the raw sampling regime the hour came from | file name / interval | derived | yes |
| `temp_c` | float, °C | National aggregate of 12-city ERA5 `temperature_2m` (instantaneous, 2 m); aggregation weights still to be decided (plan D-03) | Open-Meteo `temperature_2m` | **external** | observed ≤ t only (see leakage rules) |
| `rel_humidity_pct` | float, % | as above, `relative_humidity_2m` | Open-Meteo | **external** | same |
| `dew_point_c` | float, °C | `dew_point_2m` | Open-Meteo | **external** | same |
| `precip_mm` | float, mm; **sum over the preceding hour** | `precipitation` (rain + showers + snow) | Open-Meteo | **external** | same |
| `wind_speed_10m_kmh` | float, **km/h** | `wind_speed_10m` (10 m wind speed). Unit **verified from the API response** (`hourly_units`) and the downloaded data were requested with the default unit, so values are km/h; divide by 3.6 for m/s if needed | Open-Meteo | **external** | same |
| `cloud_cover_pct` | float, % | `cloud_cover` (total cloud cover, area fraction) | Open-Meteo | **external** | same |
| `ghi_wm2` | float, W/m²; **mean over the preceding hour** | `shortwave_radiation` (global horizontal irradiance) | Open-Meteo | **external** | same |
| `temp_c_<region>` (optional) | float, °C | Per-grid-region mean (Northern, Western, Southern, Eastern, North-Eastern) for sensitivity analysis | Open-Meteo | **external** | same |
| `cdd`, `hdd`, `heat_index` | float | Cooling/heating degree-hours vs a base temperature chosen on training data only; humidity-adjusted heat index | derived from weather | derived | same as inputs |
| `hour` | int 0–23 | `timestamp.hour` | – | have | yes (deterministic) |
| `day` | int 1–31 | day of month | – | have | yes |
| `day_of_week` | int 0–6 (Mon = 0) | | – | have | yes |
| `day_of_year` | int 1–366 | | – | have | yes |
| `month` | int 1–12 | | – | have | yes |
| `is_weekend` | 0/1 | Saturday/Sunday | – | have | yes |
| `season` | category | IMD-style: winter (Jan–Feb), pre_monsoon_summer (Mar–May), monsoon (Jun–Sep), post_monsoon (Oct–Dec). Calendar approximation, not the actual monsoon onset | – | have | yes |
| `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | float | Cyclical encodings (periods 24, 7, 12, 365.25) | – | have | yes |
| `is_holiday` | 0/1 | Date is on the DoPT **central gazetted** list (pan-India proxy; **not** state holidays) | `data/raw/calendar/dopt_central_gazetted_holidays.csv` | have (sourced) | yes (dates are published a year ahead) |
| `holiday_name` | text | Name(s) from the list | same | have | yes |
| `festival_name` | text | Holiday name where the type is a festival (not Republic/Independence/Gandhi Jayanti) | same | have | yes |
| `is_major_festival` | 0/1 | Analyst-defined: Holi, Diwali, Dussehra, Id-ul-Fitr, Christmas. **My assumption, not an official classification** | same | have | yes |
| `is_regional_holiday` | 0/1 | State/regional holidays | none yet | **external, not collected** | – |
| `festival_window` (optional) | int | Signed days to the nearest major festival (e.g. −3…+3) | derived | derived | yes |
| `lag_*`, `roll_*` | float | Lags/rolling statistics of demand (and optionally solar/wind) — created only inside the modelling pipeline with `shift(≥ 0)` relative to the forecast origin | demand etc. | derived (later) | yes if built from ≤ t |

### Weather units (verified 2026-10-13 from the Open-Meteo API `hourly_units` field and the 12 downloaded city files)
| API variable | Unit | Time meaning | Raw-file column |
|---|---|---|---|
| `temperature_2m` | °C | instantaneous at the timestamp | `temperature_2m` |
| `relative_humidity_2m` | % | instantaneous | `relative_humidity_2m` |
| `dew_point_2m` | °C | instantaneous | `dew_point_2m` |
| `precipitation` | mm | sum of the **preceding** hour | `precipitation` |
| `wind_speed_10m` | **km/h** | instantaneous | `wind_speed_10m` |
| `cloud_cover` | % | instantaneous | `cloud_cover` |
| `shortwave_radiation` | W/m² | mean of the **preceding** hour | `shortwave_radiation` |

Timestamps are IST (`timezone=Asia/Kolkata`), model ERA5 (0.25° grid; the nearest grid cell is stored in the script output). The preceding-hour variables are labelled at the hour end, whereas demand is an HH:00 snapshot, so there is up to a one-hour offset; how to treat it is open decision D-09 in `docs/project_plan.md`. Units are never mixed: no conversion has been applied to the raw files.

## Columns deliberately not carried forward
`TOTAL_THM_ONLY`, `TOTAL_NLDC_HYD`, `TOTAL_NLDC_GAS`, `TOTAL_NLDC_NPC`, inertia columns, `SYS_IN`, `SPARE 05`: absent from the 2024–25 file, so they cannot form a consistent series. They can be used for exploratory work on 2021–2023 only.

## Cleaning rules (FROZEN 2026-10-13; implemented in `scripts/build_hourly_dataset.py`, results in `reports/cleaning_log.md`)
Output of this stage: `data/processed/india_hourly_demand_clean.csv` (demand, solar, wind, total generation; no weather/calendar yet). Columns: `electricity_demand_mw` (cleaned), `demand_raw_mw`, `demand_quality_flag` (the *reason* the raw value was rejected: `ok`, `zero_outage`, `implausible`, `dip_spike`, `isolated_point`), `demand_imputed` (bool), `demand_stuck_flag` (bool, flag only), `solar_generation_mw` (negatives clipped to 0), `solar_raw_mw`, `wind_generation_mw`, `total_generation_mw`, `source_regime` (`5min`/`10s`/`hourly`).

1. The hourly grid must be complete and unique (assertion: 33,432 rows 2021-09-01 00:00 to 2025-06-24 23:00 IST). Footer rows of the 2024-25 workbook are already excluded upstream.
2. **R1 zero_outage:** demand = 0 → missing. Wind/solar/total are blanked in the same hours (whole channel dropped out).
3. **R2 implausible:** demand outside 80,000–260,000 MW → missing (the lowest valid reading is ≈ 95 GW and the record peak ≈ 250 GW).
4. **R3 dip_spike:** an hour-over-hour jump of more than 15 % that returns to within 10 % of the previous level within 3 h → missing.
5. **R3b isolated_point:** a value differing from the mean of its two neighbours by more than 10 % → missing (normal hours deviate by < 5 % at the 99.9th percentile; this catches glitches inside the morning/evening ramps that R3 cannot).
6. **R4 imputation:** linear interpolation only for missing runs of ≤ 3 h; longer runs stay missing and are never filled. Imputed hours are flagged.
7. **Stuck values:** demand identical to the previous hour is flagged only (29 hours), not removed.
8. **Solar:** negatives clipped to 0. The growing night-time baseline (9 → 886 MW) is **not** corrected here; it is handled in Phase 3 using ERA5 radiation (zero when it is night at all 12 stations).
9. **Not done here:** the hourly-mean sensitivity target (task 2.6) and before/after plots (2.5).

**Result:** 33,432 hours; 61 hours (0.18 %) had a rejected raw demand value: 20 zero_outage, 25 dip_spike, 16 isolated_point (0 implausible); 49 were interpolated and **12 remain missing** (2021-10-08 20:00 → 10-09 03:00, 8 h; 2022-03-04 05:00 → 08:00, 4 h). Windows that touch missing hours must be excluded from training/evaluation, not filled.

**Honesty note:** the thresholds were set from physical plausibility and the distribution of hour-to-hour deviations in the *whole* 2021–2025 series, which includes the later period that will be used for validation/testing. No forecasting performance was used, but this is a small leak of knowledge about the data, and it is recorded in the plan's decision log (D-05) so it can be disclosed.

---

## Actual canonical file: `data/processed/india_hourly_canonical.csv` (built 2026-10-13)
33,432 rows × 44 columns (timestamp + 43). Index `timestamp`: IST, hour start, 2021-09-01 00:00 → 2025-06-24 23:00. Built by `scripts/build_canonical_dataset.py`; counts in `reports/canonical_dataset_log.md`. The proposed table above remains the design reference; differences: degree-hours/lags/rolling features are *not* in this file (they are created inside the modelling pipeline), and regional festivals are not included.

| Group | Columns | Notes |
|---|---|---|
| Time | `timestamp`, `hour`, `day`, `day_of_week` (Mon=0), `day_of_year`, `month`, `is_weekend`, `season`, `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | deterministic, known in advance |
| Holidays | `is_holiday`, `holiday_name`, `festival_name`, `is_major_festival` | 85 central gazetted holidays verified against DoPT memoranda; `is_major_festival` is analyst-defined; known in advance |
| Target | `electricity_demand_mw` | cleaned all-India demand, MW; 12 hours missing by design |
| Target audit | `demand_raw_mw`, `demand_quality_flag`, `demand_imputed`, `demand_stuck_flag` | never use as model inputs (they describe cleaning, partly using future neighbours); use to exclude/weight windows |
| Generation | `solar_generation_mw` (negatives clipped, national-night hours zeroed), `solar_raw_mw`, `wind_generation_mw`, `total_generation_mw` | MW; usable only as **past** values (≤ forecast origin); 20 hours blank where the whole channel dropped out |
| Night flags | `is_national_night`, `solar_night_zeroed` | derived from ERA5 radiation = 0 at all 12 cities; `is_national_night` is a deterministic astronomical fact, known in advance |
| Weather (national, equal weight over 5 regions) | `temp_c`, `rel_humidity_pct`, `dew_point_c`, `precip_mm`, `wind_speed_10m_kmh`, `cloud_cover_pct`, `ghi_wm2` | ERA5 reanalysis, **observation-like, not a forecast**: allowed at time ≤ origin; future values only in the labelled oracle scenario B |
| Weather (regional temperature) | `temp_c_n`, `temp_c_w`, `temp_c_s`, `temp_c_e`, `temp_c_ne` | Northern, Western, Southern, Eastern, North-Eastern region means of 12-city ERA5 temperature |
| Provenance | `source_regime` | `5min` / `10s` / `hourly` sampling of the raw source |
