# Novelty check (web search, 2026-10-15)

**Status:** preliminary. It was done with a general web search tool (about ten queries, US-only index, no access to IEEE Xplore, ScienceDirect or Scopus, and most papers were only read as abstracts or search summaries). "I found no study that ..." means exactly that; it does not prove that none exists. A proper systematic search (Scopus / Web of Science / Google Scholar with the queries below, plus citation chaining from the papers listed) must still be done before any novelty claim is written down.

## What was searched
1. XGBoost LSTM India national electricity demand forecasting hourly weather festival features comparison
2. India all-India hourly electricity demand forecasting deep learning Grid-India dataset Mendeley 2021-2025 solar wind
3. electricity load forecasting conformal prediction intervals LSTM XGBoost multiple horizons comparison computational cost
4. Indian festivals Diwali Holi holidays effect on electricity demand forecasting machine learning
5. gradient boosting versus LSTM short-term load forecasting benchmark
6. all-India electricity demand forecast prediction intervals uncertainty quantification India national grid
7. India electricity demand forecasting XGBoost LSTM comparison Regional Load Despatch Centre day-ahead multi-horizon
8. two page checks (arXiv 2603.29179, CIGRE D2-10524-2026)

## What was found (closest work, with what each does and does not show)
| Work | What it is | Relevance / difference |
|---|---|---|
| [Grid-India, "ANN based day-ahead forecasting model for all-India demand" (CIGRE D2_10524_2026)](https://www.e-cigre.org/publications/detail/d2-10524-2026-ann-based-day-ahead-forecasting-model-for-all-india-demand.html) | Feed-forward ANN, 5-minute blocks, SCADA demand + IMD weather from 2024, tested on January 2025; authors from Grid-India | Closest all-India operational model. The page (summary only; the full paper was not accessible) states no comparison with XGBoost/LSTM and no uncertainty estimates; no multi-horizon study mentioned |
| [Bloomfield & Hunt (Newcastle), XGBoost state-level demand from weather](https://eprints.ncl.ac.uk/305255) | Poster: one XGBoost model per Indian state from daily Grid-India reports, weather predictors; also all-India demand and wind/solar supply | Daily resolution, XGBoost only; per the search summary, holidays and day of week matter. Poster only, results not verified by me |
| [Luhach & Jha, Indian peak power demand forecasting with a Temporal Fusion Transformer (arXiv 2603.29179)](https://arxiv.org/abs/2603.29179) | Peak demand, six years, one-year horizon | Long-term peak, not hourly; weather/holiday features and intervals not mentioned on the page |
| [Mendeley dataset used here (Grid-India NERLDC, hourly 2021-09 to 2025-06)](https://data.mendeley.com/datasets/y58jknpgs8) | The data source | Metadata describes the period inconsistently (as found in our own audit); no paper using it for deep learning was found |
| [Politecnico di Milano study on conformal prediction with RF, XGBoost, LSTM, Transformer (solar PV)](https://re.public.polimi.it/handle/11311/1307310) | Conformal intervals for PV output at 5-15 min horizons | Tree models with conformal methods gave well-calibrated intervals while the LSTM produced overly narrow intervals with under-coverage (per the search summary). A different domain, but consistent in spirit with our observation that interval calibration depends on the model and shifts |
| [Greek distribution grid, LightGBM vs deep learning (Energies 2025)](https://www.mdpi.com/1996-1073/18/19/5060) and a [Kazakhstan thesis](https://nur.nu.edu.kz/items/5d7c523a-6dda-4cf4-bdfc-ab9eef3ffd32/full) | Gradient boosting beating LSTM | Supports the view that tree models are hard to beat on engineered tabular features; different systems and data |
| [Informatica 2026, LSTM with temporal feature engineering](https://www.informatica.si/index.php/informatica/article/view/9317) | Reports LSTM significantly better than gradient boosting and others for hourly load | The literature is mixed; results depend on features, data size and tuning |
| [Building thermal load, shallow ML vs deep learning (Applied Energy 2020)](https://eta-publications.lbl.gov/publications/building-thermal-load-prediction) | LSTM good at 1 h but not 24 h; forecast-weather uncertainty hurts XGBoost | A different system. Our all-India results show the opposite at 1 h (ridge/XGBoost ahead of LSTM) |
| [DemandCast, global hourly demand with XGBoost (NeurIPS 2025 poster)](https://neurips.cc/virtual/2025/poster/126932) | Multi-country hourly demand with weather and socio-economic variables | Not India-specific in the abstract; no LSTM comparison stated |
| Other Indian studies that came up: household-level LSTM, Chhattisgarh monthly LSTM (IET 2024), a monthly national comparison of CEA trend model / SARIMA / LSTM / Prophet, peak-demand XGBoost for Delhi NCR and Rajasthan | Different scale (household, state, monthly or peak) | None combines all-India hourly demand, several horizons and uncertainty |

## What I did not find
- A study comparing XGBoost and LSTM on **all-India hourly demand** across several forecast horizons with a strictly chronological, single-use test period.
- An Indian load-forecasting paper with **festival-specific** modelling or a measured effect of festival features beyond calendar features (the searches returned only general statements that holidays matter).
- **Prediction-interval** evaluation (conformal, quantile regression, LSTM quantile head) for all-India demand, or an interval comparison of XGBoost and LSTM for load with computational cost reported.
- A feature-group ablation with an **oracle-weather upper bound** for all-India demand, or an analysis of solar/wind generation as inputs for demand forecasting at all-India scale.

## What this project's contribution can honestly be, given the above and our results
Each ingredient exists somewhere in the literature; the combination for all-India hourly demand was not found. The contribution is therefore best framed as a careful **empirical benchmark and negative/null-result study**, not as a new method:
- a strictly chronological protocol with a frozen configuration and a single test run (reproduced exactly), including paired block-bootstrap comparisons;
- evidence that, on this dataset, XGBoost is as good as or better than an LSTM at 6-48 h and that ridge is hard to beat at 1 h; the LSTM does not close the gap to XGBoost as the training set grows from 12 months to 2.3 years (its error is flat at 1, 24 and 48 h and falls at a similar pace to XGBoost's at 6 h);
- evidence about which inputs matter: calendar features matter most; past weather, past solar/wind and festival indicators add little on the test year; perfect future weather would help at 24-48 h;
- interval results (conformal and quantile methods) showing coverage degrades with horizon under distribution shift, and rolling calibration recovers part of it;
- computational-cost logging.
Claims should avoid "first" or "novel" unless a systematic search confirms it, and should state the limits listed in `reports/research_question_answers.md`.

## To do before writing a novelty claim
- Run the queries above in Scopus / Web of Science / Google Scholar; add queries on "probabilistic load forecasting India", "Grid-India load forecasting machine learning", "festival effect electricity demand", "conformal load forecasting LSTM".
- Read the full texts of the Grid-India ANN paper and the Newcastle work; check their error levels (MAPE) against ours (XGBoost test MAPE 0.57 / 1.43 / 1.95 / 2.70 % at 1 / 6 / 24 / 48 h) before drawing comparisons.
- Confirm the license and citation of the Mendeley dataset (CC BY 4.0, DOI 10.17632/y58jknpgs8.2).
