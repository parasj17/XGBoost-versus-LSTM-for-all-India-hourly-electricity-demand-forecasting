# India-wide electricity demand forecasting (XGBoost / LSTM) — data phase

Research objective: investigate India-wide electricity-demand forecasting with machine learning and deep learning (primarily XGBoost and LSTM), integrating historical demand with weather and calendar/festival effects, and evaluating multi-step accuracy, uncertainty and computational efficiency.

**Current phase: dataset acquisition, audit and design. No models have been trained.**

## Status in one paragraph
The Mendeley dataset is all-India (no regional split), covers 2021-09-01 → 2025-06-24 (not June 2025 or Dec 2023 as the metadata says), and is hourly only for 2024–25; earlier months are 5-minute and 10-second SCADA data that must be reduced to hourly. After cleaning (outage zeros, spikes) it is suitable for India-wide hourly forecasting. Weather (ERA5 via Open-Meteo, 12 cities) is downloaded but not yet aggregated; the holiday file is built from DoPT memoranda. See `reports/dataset_audit.md`.

## Layout
```
data/raw/electricity_mendeley/   extracted Mendeley workbooks (do not edit)
data/raw/Electricity ... .zip    original download (do not edit)
data/raw/weather/                Open-Meteo ERA5 CSVs, 12 cities (downloaded)
data/raw/calendar/               sourced holiday CSV (tracked in git)
data/raw/zenodo_state_daily/     Zenodo 14983362 daily state energy + weather (42 files, MD5-verified)
data/interim/                    audit outputs, parse cache (regenerable)
data/processed/                  calendar features (canonical dataset not built yet)
reports/                         inventory, audit, data dictionary, acquisition plan
scripts/                         inspect_electricity_data.py, build_hourly_dataset.py, build_canonical_dataset.py, plot_cleaning_check.py, inspect_weather_data.py, build_calendar_features.py,
                                 download_zenodo_state_daily.py
references.md                    all external sources and literature
docs/                            literature review, research gap and methodology context
```
`data/raw/*` (except `calendar`), `data/interim`, `data/processed` are git-ignored because they are large/regenerable.

## Setup
```powershell
py -m pip install -r requirements.txt
```
(`py` is the Python launcher on this machine; `python` alone is not on PATH.)

## Reproduce this phase
```powershell
py scripts\inspect_electricity_data.py      # audit all workbooks (~35 min first run, cached afterwards)
py scripts\build_calendar_features.py       # calendar_daily.csv / calendar_hourly.csv
py scripts\inspect_weather_data.py --pilot  # small API check, saves nothing
```
Scripts use paths relative to the project root and have `--help`.

## Plan
`docs/project_plan.md` is the living plan (phases, gates, decision log, next actions). Work follows it.

## Reports
- `reports/dataset_inventory.md` — what is in the workspace, what could and could not be accessed
- `reports/dataset_audit.md` — verified schema, period, resolution, quality issues
- `reports/data_dictionary.md` — proposed canonical columns, units, sources, cleaning rules
- `reports/data_acquisition_plan.md` — target definition, weather/calendar design, leakage rules, uncertainty and efficiency plan, next steps

## Data licences
Mendeley dataset: CC BY 4.0 (cite DOI 10.17632/y58jknpgs8.2). Open-Meteo: free for non-commercial use with attribution. See `references.md`.
