# Electricity dataset audit

Generated from `scripts/inspect_electricity_data.py` (outputs: `data/interim/electricity_audit.json`, `data/interim/hourly_audit_view.csv.gz`) plus follow-up checks. Raw files were opened read-only and never modified. Labels: **[V]** verified in the files, **[I]** inference from evidence, **[U]** unknown / needs confirmation.

## 1. Headline findings

1. **[V] The data are all-India, not regional.** Every SCADA tag is an "ALL_INDIA"/"NLDC" aggregate (e.g. `NLDC_DEMAND|P`, `ALL_INDIA_WIND|P`, `ALL_IND_SOLAR|P`). There is no region identifier and no per-region column in any workbook. The description's "five regional grids" describes the coverage of the national total, not separate series. Regional forecasting is therefore **impossible with this dataset**.
2. **[V] The dataset is NOT uniformly hourly.** Three sampling regimes exist:

   | Period | Files | Sampling | Raw rows |
   |---|---|---|---|
   | 2021-09-01 → 2022-09-30 | 13 monthly | 5 min (300 s) | 113,760 |
   | 2022-10-01 → 2023-12-31 | 15 monthly | **10 s** | 3,948,480 |
   | 2024-01-01 → 2025-06-24 | 1 combined file | 1 h (3600 s) | 12,984 |

   "1-hour" in the title is true only for the last file. The earlier files must be reduced to hourly by us (see section 6).
3. **[V] Real period is 2021-09-01 00:00 → 2025-06-24 23:00.** The title says "…June 2025" (data stop 6 days short); the description says "…December 2023" (stale). Neither is exact.
4. **[V] Complete hourly grid.** Expected hourly timestamps = 1,393 days × 24 = **33,432**; hourly timestamps present = **33,432** (0 missing, 0 duplicates, 0 unparseable once footer rows are removed). Endpoint convention: first hour (2021-09-01 00:00) and last hour (2025-06-24 23:00) are both inclusive hour-start stamps. **Caveat:** "no missing timestamps" does not mean "no missing data" — see section 5 (sensor drop-outs are encoded as 0).
5. **[V] The data contain errors that must be cleaned:** zero demand during telemetry outages, impossible spikes (402,065 MW and 466,674 MW), partial-dip glitches, a footer block in the last workbook, and a drifting night-time solar offset.
6. **[V] Metadata gaps:** units, definitions, time zone and the meaning of "demand" are not documented anywhere I could access; everything below on those points is inferred.

## 2. Workbook structure

- 28 monthly workbooks (Sep 2021 – Dec 2023) + 1 combined workbook (`January 2024- June 2025.xlsx`). Per-file table in `dataset_inventory.md`.
- **Monthly workbooks ("layout A")** sheets: `_osi_config`, `Sheet1`, `Sheet2`.
  - `_osi_config`: one JSON cell with the SCADA historian configuration. Its refresh rate `r` is 300 (Sep 2021 file) or 10 (Jan 2023 file), matching the observed sampling; it maps each tag key to a signal name (section 3).
  - `Sheet1`: row 1 = SCADA tag ids, row 2 = signal names, data from row 3; 15 columns.
  - `Sheet2`: author scratch sheet (`Time`, `Solar+Wind`, `Every Hourly Timestamps`). Its third column lists the Solar+Wind value at 00:00, 01:00, 02:00… on consecutive rows (verified on Sep 2021: 7839.4, 7322.2, 6423.7 equal the `Sheet1` values at 00:00, 01:00, 02:00). **[I]** This indicates the authors build their "hourly" series by **picking the instantaneous reading at HH:00**, not by averaging. Not used by us.
- **Combined workbook ("layout B")** sheets: `Report` (data), `Sheet2`, `Sheet3` (both empty). Row 1 = header, 5 columns.
  - **12,990 rows = 12,984 data rows + 6 footer rows** (labels `Minimum`, `Timestamp`, `Maximum`, `Timestamp`, `Average`, `Sum`). This Excel summary block yields values up to 2.5×10⁹ if read naively. It must be dropped (the script does).
  - Timestamps are text `dd-mm-yyyy HH:MM:SS` (day-first), unlike the datetime cells in layout A.

## 3. Columns (exact) and meaning

### Layout A (`Sheet1`, 2021-09 → 2023-12)

| Col | Exact header (row 2) | SCADA key (row 1) | Meaning **[I]** | Unit **[I]** |
|---|---|---|---|---|
| A | `Time` | – | timestamp | naive local time |
| B | `TOTAL_THM_ONLY\|P` | `SCADA/ANALOG/044MQ068/0` | all-India thermal generation | MW |
| C | `TOTAL_NLDC_HYD\|P` | `…044MQ062/0` | hydro generation | MW |
| D | `TOTAL_NLDC_GAS\|P` | `…044MQ063/0` | gas generation | MW |
| E | `TOTAL_NLDC_NPC\|P` | `…044MQ064/0` | nuclear generation | MW |
| F | `ALL_INDIA_WIND\|P` | `…044MQ070/0` | all-India wind generation | MW |
| G | `ALL_IND_SOLAR\|P` | `…044MQ206/0` | all-India solar generation (**raw**, can be negative) | MW |
| H | `Solar` | (formula) | `max(0, ALL_IND_SOLAR)` — verified exactly in two files | MW |
| I | `Solar+Wind` (`Wind+Solar` in Feb & May 2023) | (formula) | H + F — verified | MW |
| J | `Total` | (formula) | B+C+D+E+I — verified | MW |
| K | `NLDC_DEMAND\|P` | `…044MQ067/0` | all-India demand (see section 4) | MW |
| L | `INTRA_SYSTEM_INTERTIA\|P` | `…030K90IF/0` | system inertia | undocumented |
| M | `ER_Total_Inertia\|P` | `…031E3193/0` | Eastern Region inertia | undocumented |
| N | `SYS_IN\|P` | `…034MT227/0` | undocumented | undocumented |
| O | `SPARE 05` | `…044MW005/0` | spare channel | – |

The `|P` suffix presumably marks a power channel. Columns L–O are not relevant to this project. Column names are identical across all 28 monthly files except the cosmetic `Wind+Solar` rename in Feb 2023 and May 2023.

### Layout B (`Report`, 2024-01 → 2025-06)

`Timestamp`, `Demand (MW)`, `Wind (MW)`, `Solar (MW)`, `Total Generation (MW)` — units are stated in the headers. **[I]** `Demand` ≈ `NLDC_DEMAND`, `Wind` ≈ `ALL_INDIA_WIND`, `Solar` ≈ **raw** `ALL_IND_SOLAR` (it contains negative values, so it is not the clipped column H), `Total Generation` ≈ `Total`. Thermal/hydro/gas/nuclear/inertia columns do **not** exist in layout B. **[U]** The authors do not confirm these mappings; level continuity across Dec 2023 → Jan 2024 and similar annual means support but cannot prove it.

## 4. What "demand" is

**[U]** The source never says "demand met" vs "requirement". Evidence for an all-India demand-met-like quantity: mean demand 149.9 GW (Sep–Dec 2021), 169.5 (2022), 182.8 (2023), 192.0 (2024), 198.3 (Jan–Jun 2025); the largest valid hourly value in 2024 is 248,688 MW, consistent with the record national peak of about 250 GW reported for May 2024. `Total` exceeds `NLDC_DEMAND` by about 1.4–1.5 GW (median) in the two files checked. I treat it as **all-India instantaneous demand (met) in MW from SCADA** — an assumption to state in the thesis. **[V] Independent support:** Zenodo 14983362's national daily *energy met* (MU) agrees with the Mendeley daily mean MW × 24 / 1000 on 955 overlapping days (correlation 0.997, mean ratio 1.001, MAPE 0.57 %; see `dataset_inventory.md` section 6). The remaining uncertainty is only the exact SCADA definition, not the order of magnitude or the national scope.

## 5. Data-quality issues

Counts are on the hourly HH:00 snapshot series unless stated.

| Issue | Evidence |
|---|---|
| **Telemetry outages recorded as 0** | 20 hourly snapshots have demand = 0 (e.g. 2021-10-08 20:00 → 2021-10-09 03:00 = 8 consecutive hours; 2022-05-23 09–11 h; 2022-07-03 10–12 h). Wind is simultaneously 0 → whole-channel dropout. Across all raw records, 123 hours contain ≥ 1 zero-demand sample and 12 hours are entirely zero. |
| **Impossible spikes** | Raw maxima 402,065 MW (Jan 2023) and 466,674 MW (Sep 2023) vs a real maximum ≈ 250 GW. They fall between HH:00 instants, so the snapshot series misses them, **but an hourly-mean aggregation would not** — it needs robust filtering first. |
| **Dips and recoveries** | 63 hour-over-hour changes > 15 % (e.g. −25 % then +33 % an hour later, many in 2021-09 → 2021-12). Look like partial-telemetry glitches, not demand. Flag with rules; do not delete blindly. |
| **Stuck values** | 41 hours where demand equals the previous hour exactly (suspect, not necessarily wrong). |
| **Negative solar** | Raw solar < 0 often at night (e.g. min −1,887 MW in Oct 2023): sensor offset. Layout A also provides the clipped column H; layout B does not. |
| **Night-time solar drift** | Mean solar at 00–03 h: 9 MW (2021), 105 (2022), 135 (2023), 259 (2024), 886 (Jan–Jun 2025). Real PV output at night is 0, so an instrument/aggregation offset grows over time. Needs a documented correction. |
| **Footer rows** | 6 summary rows in the layout-B sheet. |
| **Metadata contradictions** | Title end = Jun 2025; description end = Dec 2023; actual end = 24 Jun 2025. Title/description say hourly; actual 5 min / 10 s / 1 h. Description credits NERLDC; tags are all-India NLDC. |
| **File naming** | `Novemeber 2023.xlsx` typo. |
| **Overlaps** | None. Monthly files tile exactly (each starts 00:00 on the 1st); Dec 2023 → Jan 2024 hand-over is continuous at hourly level. |
| **Duplicates** | 0 duplicate timestamps in every file; all timestamps strictly increasing. |
| **Empty cells** | 0 null cells in data columns; non-numeric text only in the footer rows. |
| **Gaps** | None at raw level inside each regime (every difference equals the nominal interval); none at hourly level. |

## 6. Aggregation to hourly — the methodology choice that matters most

For the 5-min and 10-s regimes there are two options:
- **Snapshot at HH:00** — matches what the authors do ([I], `Sheet2`) and presumably what the layout-B file contains.
- **Hourly mean** — physically a better energy-type measure but more exposed to outages/spikes.

Measured on 2021-09 → 2023-12 (valid values only), the HH:00 snapshot differs from the hourly mean by **MAE 2,120 MW (1.2 % of mean demand), 99th percentile 8,420 MW**. That is not negligible relative to 1-hour-ahead forecast errors of ~1–3 %. **[U → I]** Whether the layout-B hourly value is a snapshot or a mean cannot be tested directly (no overlapping period). Indirect evidence (2026-10-14): a ridge model trained on 2021-23 hourly means and scored on the 2024 values is about 2.3 times worse at 1 h (MAE 2,570 vs 1,095 MW) than the same model trained on snapshots, so the 2024-25 values behave like HH:00 snapshots.

**Recommendation:** use the HH:00 snapshot as the primary target (consistent across regimes and with the only hourly source), and run a sensitivity check with hourly means for 2021–2023, reporting the difference.

## 7. Time zone and timestamp convention

- Stamps are naive (no UTC offset). **[I] IST (UTC+05:30):** mean solar generation in 2024–25 peaks at 12:00 (≈48.9 GW) and is near its night baseline before 05:00 and after 19:00; UTC-labelled data would peak near 06:30. India has no DST, and there are 0 missing/duplicate hours.
- The interval label convention (start vs end of interval) is **[U]**; irrelevant for snapshots but relevant for hourly means.
- Open-Meteo is requested with `timezone=Asia/Kolkata`, so timestamps align. Open-Meteo's `precipitation` and `shortwave_radiation` are *preceding-hour* aggregates, whereas demand snapshots are instants: a ≤ 1 h offset to be handled deliberately.

## 8. Suitability verdict

**Suitable for all-India hourly demand forecasting** over 2021-09-01 → 2025-06-24 (3.8 years, 33,432 hours), after cleaning, with limitations: one national series only (no regional models, no spatial-aggregation comparison), only ~3.8 years (about 3 occurrences of each movable festival), the sampling-regime change, missing documentation, and an inferred demand definition. Solar/wind series are usable but the solar baseline drift needs correction.
