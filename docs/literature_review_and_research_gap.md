> **Provenance:** This file is the research context and literature review written by the project owner, saved verbatim on 2026-10-13. Paper details below are as provided by the owner. Bibliographic metadata (title/DOI) was cross-checked against Crossref/DataCite and the results are recorded in `references.md`; the descriptions of each paper's methods and results have **not** been independently verified against the full papers. Keep this document as the project's literature record, and update the research gap if the data audit (`reports/dataset_audit.md`) requires it.

---

# RESEARCH PROJECT CONTEXT: INDIA-WIDE ELECTRICITY DEMAND FORECASTING USING MACHINE LEARNING AND DEEP LEARNING

## 1. Project Overview

I am an undergraduate Computer Science Engineering student specializing in Data Science. I am developing a research project on India-wide electricity-demand forecasting using machine learning and deep learning.

I have completed an initial literature review of research papers published between 2020 and 2026. I have identified a preliminary research gap and am now restarting the practical phase of the project, beginning with dataset acquisition and validation.

My primary goal is to study whether electricity-demand forecasting can be improved by combining historical demand with relevant external variables, while comparing machine-learning and deep-learning approaches.

### Proposed research direction

Develop and evaluate an India-wide electricity-demand forecasting framework that integrates historical electricity demand with weather, calendar and festival effects, and potentially renewable-generation variables. Compare XGBoost and LSTM for multi-step forecasting and examine forecast accuracy, uncertainty and computational efficiency.

Electric-vehicle-related features are optional and should only be included if reliable, compatible historical data is available.

### Proposed research question

"How effectively can XGBoost and LSTM forecast India-wide electricity demand when historical demand is combined with weather and calendar/festival-related features, particularly across multiple forecasting horizons, and how do the models compare in terms of forecasting accuracy, predictive uncertainty and computational efficiency?"

This question is preliminary. Refine it based on the actual dataset, the literature evidence and the feasible scope of an undergraduate research project.

---

# 2. LITERATURE REVIEW

## Paper 1: Daily Average Load Demand Forecasting Using LSTM Model Based on Historical Load Trends

**Authors:** Rashmi Bareth, Anamika Yadav, Shubhrata Gupta and Mohammad Pazoki.

**Year:** 2024.

**Journal:** IET Generation, Transmission & Distribution.

**DOI:** https://doi.org/10.1049/gtd2.13132

### Research objective

The paper develops an LSTM-based framework to forecast electricity demand in Chhattisgarh, India. It focuses on monthly electricity-demand forecasting expressed as average daily load demand.

### Dataset

The authors use historical load-demand data from Chhattisgarh's load dispatch system for 2018–2022 to forecast demand for 2023.

The dataset is regional, not an India-wide national electricity-demand dataset.

### Methodology

The principal model is Long Short-Term Memory (LSTM).

The methodology involves:
- Historical load-data collection.
- Data preprocessing, including handling missing values, duplicates and outliers.
- Correlation-based examination of historical load patterns.
- Data normalization.
- LSTM training and testing.
- Evaluation with Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).

The paper discusses seasonal demand differences and the effect of weather on electricity demand, although its forecasting framework primarily uses historical load patterns.

### Key findings

The reported MAPE ranges are:
- Training: 0.010%–0.652%.
- Testing: 0.378%–10.54%.

The authors compare LSTM with an Artificial Neural Network (ANN) and report better forecasting accuracy for LSTM.

The study emphasizes that electricity demand varies with season, and it highlights the importance of forecasting monthly average demand and peak-related patterns for load-dispatch planning.

### Limitations and relevance to our research

The following are limitations relevant to our project, rather than claims that the authors explicitly identified every one of them:

- The study focuses on one Indian state rather than the national electricity system.
- The forecasting target is based on average daily demand rather than a national hourly time series.
- Historical demand is the principal forecasting input.
- The study does not provide the complete multi-variable, national-scale comparison of XGBoost and LSTM proposed in our project.
- Its results should not be directly compared numerically with results from studies using different datasets and forecasting horizons.

### Relevance

This is an important Indian reference because it shows that LSTM-based forecasting has been applied to real regional electricity-demand data in India.

Our project can investigate whether a similar temporal model performs well on a national-scale demand series and whether external features improve performance.

---

## Paper 2: Mid-Term Electricity Load Forecasting by a New Composite Method Based on Optimal Learning MLP Algorithm

**Authors:** Mostafa Askari and Farshid Keynia.

**Year:** 2020.

**Journal:** IET Generation, Transmission & Distribution.

**DOI:** https://doi.org/10.1049/iet-gtd.2019.0797

### Research objective

This paper addresses mid-term electricity-load forecasting, focusing on daily peak demand for a future month. Accurate peak-load estimates are important for operational planning, generation scheduling and resource allocation.

### Dataset

The paper evaluates its approach using two datasets:

- North American electric-utility data covering 1987–1991, with hourly load converted to daily peak-load values.
- Local peak-load data from Kerman, Iran, covering 2016–2017.

These datasets provide an international comparison of daily peak-load forecasting.

### Methodology

The authors combine a Multi-Layer Perceptron (MLP) neural network with two optimization algorithms:

- Particle Swarm Optimization (PSO), using an adaptive inertia-weight approach.
- Ant Lion Optimizer (ALO).

The optimization algorithms train the network by optimizing parameters such as weights and biases and help determine the network structure, including hidden-layer neuron count.

The study also examines input-feature selection and compares one-year training with seasonal training.

### Key findings

The authors report that:
- ALO and PSO both produce promising results.
- ALO generally performs better than PSO in the investigated cases.
- Seasonal training achieves the lowest MSE and MAPE in the two investigated case studies.
- Selecting appropriate input features and network structure can improve forecasting performance and reduce computational time.

### Limitations and relevance to our research

For our project:
- The target is mid-term daily peak load, rather than India-wide hourly demand.
- The data comes from North America and Iran, not the Indian national grid.
- The research emphasizes neural-network optimization and historical load features, rather than our proposed combination of XGBoost, LSTM, weather, calendar effects and renewable-generation information.

### Relevance

This paper demonstrates that model performance depends not only on the algorithm but also on training-window selection, input-feature selection and parameter optimization.

We should therefore conduct a fair comparison using consistent time splits, features and evaluation metrics instead of assuming that a more complicated model will always perform better.

---

## Paper 3: Short-Term Load Forecasting in Smart Grids Using Artificial Intelligence Methods: A Survey

**Authors:** Sirus Salehimehr, Behrooz Taheri and Mostafa Sedighizadeh.

**Year:** 2022.

**Journal:** The Journal of Engineering.

**DOI:** https://doi.org/10.1049/tje2.12183

### Research objective

The paper reviews and evaluates artificial-intelligence methods for short-term electricity-load forecasting in smart-grid environments.

It examines:
- Long Short-Term Memory (LSTM).
- Group Method of Data Handling (GMDH).
- Adaptive Neuro-Fuzzy Inference System (ANFIS).

The smart-grid context includes photovoltaic generation, wind turbines, battery energy storage and electric-vehicle charging.

### Experimental setup

The study uses a simulated smart-grid environment and evaluates the models under normal and noisy conditions.

It also considers computational requirements, including execution time and CPU/RAM usage.

Therefore, its findings should be interpreted as results from the study's evaluation environment, rather than proof of performance on an actual India-wide electricity-demand dataset.

### Key findings

Under normal conditions, the reported MAPE values are approximately:

| Model | MAPE |
|---|---:|
| LSTM | 1.5038% |
| GMDH | 2.3318% |
| ANFIS | 2.3738% |

Under noisy conditions, LSTM also achieves the best reported accuracy among the three models, with MAPE around 3.46%, compared with around 3.99% for GMDH and 3.94% for ANFIS.

LSTM has higher computational and memory requirements. GMDH and ANFIS use fewer resources but achieve somewhat lower accuracy.

### Limitations and relevance to our research

- The experimental environment is simulated rather than a national Indian demand time series.
- The study evaluates three AI approaches, not the XGBoost-versus-LSTM comparison proposed in our research.
- It emphasizes the effect of noise and computational resources, illustrating the need to consider more than prediction error alone.

### Relevance

This paper supports including computational efficiency in our evaluation.

We should record model-training time, inference time and hardware conditions, and assess whether additional accuracy justifies additional computational cost.

It also supports investigating the robustness of forecasting under data-quality issues and changing conditions.

---

## Paper 4: Short-Term Load Demand Forecasting Through Rich Features Based on Recurrent Neural Networks

**Authors:** Dongbo Zhao, Qian Ge, Yuting Tian, Jia Cui, Boqi Xie and Tianqi Hong.

**Year:** Published online in 2020; journal publication in 2021.

**Journal:** IET Generation, Transmission & Distribution.

**DOI:** https://doi.org/10.1049/gtd2.12069

### Research objective

The paper develops a recurrent neural-network forecasting framework that incorporates richer information than historical load alone.

The objective is to improve short-term electricity-load forecasting using sequential modelling, feature selection and attention mechanisms.

### Datasets

The study evaluates three publicly available datasets:

- Polish electricity-system hourly loads, 2002–2004.
- Global Energy Forecasting Competition 2012 (GEFCom2012).
- ISO New England hourly loads, 2011–2015.

These datasets represent international electricity systems, not India-wide national demand.

### Methodology

The authors propose a sequence-to-sequence (Seq2Seq) architecture based on recurrent neural networks, using Gated Recurrent Unit (GRU) units in the encoder and decoder.

A feature-attention mechanism learns useful combinations of input features along the channel and temporal dimensions.

Depending on the dataset, inputs include:
- Historical electricity demand.
- Weekday information.
- Temperature.

The study examines scenarios with historical demand alone and scenarios incorporating external features.

MAPE and RMSE are used for evaluation. The proposed architecture is compared with several established methods, including Random Forest, MLP and other recurrent or neural-network models.

### Key findings

The reported Seq2Seq MAPE values are:
- Polish dataset: 1.41%.
- GEFCom2012: 3.32%.
- ISO New England: 1.39%.

The authors report improved forecasting performance compared with the selected competing methods.

These metrics should not be compared directly with values from other papers without considering dataset, target, horizon and experimental setup.

### Limitations and relevance to our research

- The evaluation uses international datasets.
- The architecture is more complex than a straightforward XGBoost baseline.
- The results do not demonstrate the exact proposed combination of national Indian electricity demand, weather, Indian holidays/festivals, solar/wind generation and XGBoost-versus-LSTM benchmarking.

### Relevance

The paper supports the use of informative external features and temporal learning.

It also suggests a useful research question for our experiments: how much improvement comes from additional variables, compared with improvements attributable to the forecasting architecture itself?

We should investigate this through controlled feature-ablation experiments.

---

## Paper 5: PowerNet: A Smart Energy Forecasting Architecture Based on Neural Networks

**Authors:** Yao Cheng, Chang Xu, Daisuke Mashima, Partha P. Biswas, Geetanjali Chipurupalli, Bin Zhou and Yongdong Wu.

**Year:** 2020.

**Journal:** IET Smart Cities.

**DOI:** https://doi.org/10.1049/iet-smc.2020.0003

### Research objective

The paper develops PowerNet, a neural-network architecture for forecasting electricity consumption using heterogeneous data sources.

The objective is to combine sequential historical consumption with non-sequential external information to improve forecasting in smart-meter environments.

### Data and input features

The study uses real-world smart-meter data from individual apartments.

The inputs cover three dimensions:
- Historical electricity consumption.
- Weather information.
- Calendar information.

### Methodology

PowerNet combines two neural-network components:

1. A Recurrent Neural Network (RNN) represents sequential dependencies in historical electricity consumption.
2. A Multi-Layer Perceptron (MLP) processes non-sequential external features such as weather and calendar information.

The outputs are combined for demand prediction.

The authors compare PowerNet with:
- Gradient Boosting Tree (GBT).
- Support Vector Regression (SVR).
- Random Forest (RF).
- Gated Recurrent Unit (GRU).

### Key findings

PowerNet achieves competitive or better performance in several individual-apartment and aggregated-consumption experiments.

For aggregated consumption:
- With 16 apartments, PowerNet records MAPE of 13.98%, compared with 14.18% for GBT.
- With 114 apartments, PowerNet records MAPE of 10.00%, while GBT performs better at 8.48%.

The result for 114 apartments is especially important because it demonstrates that a proposed neural architecture is not necessarily best in every scenario.

The paper also observes that forecasting errors increase as the prediction horizon extends. Model retraining frequency is an important operational consideration.

### Limitations and relevance to our research

- The study uses apartment-level smart-meter data rather than an India-wide national demand series.
- Its findings may be affected by the particular data used; the authors identify evaluation on other datasets as future work.
- It does not establish that one forecasting architecture is universally best across geographies and aggregation levels.

### Relevance

PowerNet is one of the most important papers for our feature-engineering argument.

It demonstrates an architecture designed to integrate historical load, weather and calendar information.

Our proposed study would adapt the underlying principle—combining demand history and external predictors—to a national Indian electricity-demand series, while comparing XGBoost and LSTM using consistent experiments.

We should not claim that integrating weather and calendar variables is a completely new idea. The possible contribution lies in the Indian national-scale context, variable integration, multi-step evaluation, uncertainty analysis and computational comparison, if these are supported by the final dataset and literature review.

---

## Paper 6: Evaluation of Electrical Load Demand Forecasting Using Various Machine Learning Algorithms

**Authors:** Akanksha Jain and S. C. Gupta.

**Year:** 2024.

**Journal:** Frontiers in Energy Research.

**DOI:** https://doi.org/10.3389/fenrg.2024.1408119

### Research objective

This paper compares several machine-learning and deep-learning approaches for short-term electricity-load forecasting using Indian electricity-utility data.

### Dataset

The authors use approximately five years of electricity-demand data from Chandigarh Union Territory, India.

The data contains demand patterns across weekly, monthly, yearly and seasonal time scales.

This is an Indian regional/UT study, not a national India-wide electricity-demand experiment.

### Methodology

The algorithms evaluated include:
- Support Vector Machine (SVM).
- Recurrent Neural Network (RNN).
- Long Short-Term Memory (LSTM).
- Deep-learning methods.
- Ensemble methods.

The study uses a moving-window LSTM approach and implements its experiments using MATLAB R2018.

The evaluation includes:
- Mean Absolute Error (MAE).
- Root Mean Square Error (RMSE).
- Normalized Mean Squared Error (NMSE).
- Mutual Information (MI).

The preprocessing includes treatment of missing data and data transformation for model training.

### Key findings

LSTM achieves the lowest prediction error among the evaluated alternatives.

For weekly load prediction, the authors report that LSTM's error is lower than that of:
- RNN by 1.35%.
- Deep-learning method by 9.45%.
- Ensemble method by 12.16%.
- SVM by 13.51%.

The reported comparison indicates that model choice materially affects forecasting results even when methods use the same regional electricity-demand data.

### Limitations and relevance to our research

- The dataset covers Chandigarh UT rather than the national electricity system.
- The paper does not evaluate the precise India-wide, multi-source forecasting framework proposed for our research.
- Its results do not establish whether XGBoost would outperform LSTM for our intended dataset and forecast horizons.

### Relevance

This is a key reference for Indian electricity-demand forecasting.

It supports choosing LSTM as a principal deep-learning candidate, but it does not justify assuming that LSTM will outperform XGBoost in our experiments.

Our research should determine which method works better on the target data and whether additional external variables create measurable improvements.

---

## Paper 7: Energy Forecasting in Smart Grid Systems: Recent Advancements in Probabilistic Deep Learning

**Authors:** Devinder Kaur, Shama Naz Islam, Md Apel Mahmud, Md Enamul Haque and Zhao Yang Dong.

**Year:** 2022.

**Journal:** IET Generation, Transmission & Distribution.

**DOI:** https://doi.org/10.1049/gtd2.12603

### Research objective

This paper reviews electricity and energy forecasting methods in smart-grid environments, with particular attention to probabilistic deep learning.

It covers:
- Statistical forecasting.
- Machine learning.
- Deep learning.
- Hybrid forecasting.
- Data preprocessing and feature engineering.
- Probabilistic forecasting and prediction intervals.

The paper also evaluates methods using real energy datasets.

### Datasets

The comparative case study uses:
- Victorian electricity-consumption data from Australia.
- American Electric Power (AEP) data.

It is an international review and comparative analysis, not an India-wide electricity-demand study.

### Methodology

The authors examine deterministic models that produce point forecasts and probabilistic models that produce distributions or prediction intervals.

One important approach investigated is Bayesian bidirectional LSTM (Bayesian BLSTM).

The paper also discusses preprocessing and feature-extraction methods, including dimensionality reduction and autoencoders.

### Key findings

The review reports that:
- Deep-learning models can perform well on larger datasets with complex nonlinear patterns, provided that their hyperparameters are appropriately tuned.
- Bayesian BLSTM performs strongly in the evaluated probabilistic forecasting comparisons.
- Probabilistic deep-learning approaches achieve at least approximately 60% lower prediction errors than the benchmark deep-learning methods reported in the study.
- Probabilistic methods require greater computational effort, creating a trade-off between accuracy, uncertainty modelling and execution time.

The reported improvement is specific to the authors' evaluated datasets and benchmark methods; it should not be treated as a universal performance guarantee.

### Limitations and relevance to our research

- The comparative datasets are international.
- Probabilistic deep learning can introduce considerable computational cost.
- Uncertainty estimates require their own evaluation; a point forecast with low MAE does not automatically provide reliable prediction intervals.

### Relevance

This paper establishes an important motivation for our uncertainty-analysis objective.

XGBoost and LSTM can initially be compared as point-forecasting models. A feasible uncertainty-estimation method can then be added, subject to scope and time.

Possible options include quantile-based forecasting or conformal prediction intervals, with coverage and interval-width metrics.

We should not begin with an expensive Bayesian deep-learning architecture unless the initial results justify the added complexity.

---

## Paper 8: Multi-Step Ahead Short-Term Load Forecasting Based on MCNN-MMoL

**Authors:** Suxun Zhu, Bowen Ren, Lili Zheng, Abdullah M. Alharbi, Changhua Yang, Shidong Wu, Aotian Yuan and Hengrui Ma.

**Year:** 2026.

**Journal:** IET Smart Energy Systems.

**DOI:** https://doi.org/10.1049/ses2.70020

### Research objective

This paper addresses multi-step-ahead short-term electricity-load forecasting.

The authors identify a problem with forecasting multiple future steps: prediction errors can accumulate, and dependencies among future time steps may not be fully captured by conventional approaches.

### Datasets

The experiments use two real-world datasets associated with Alberta and the New York Independent System Operator (NYISO).

The study examines multi-step forecasts at 24-, 48- and 96-step horizons.

These are international datasets, not Indian national electricity-demand data.

### Methodology

The proposed architecture, MCNN-MMoL, combines:

- Multi-path Convolutional Neural Networks (MCNN), which extract load characteristics at different scales.
- A multi-gate mixture of LSTM networks (MMoL), which combines outputs from multiple LSTM experts.
- Task-specific gating, which learns different combinations of experts for different prediction steps.
- Multi-task learning to jointly optimize multiple future-step predictions.

The model is compared with alternative approaches and different multi-step prediction strategies.

### Key findings

For the reported full-input scenario, the study gives the following results:

| Forecast horizon | MAPE | RMSE | R² |
|---|---:|---:|---:|
| 24 steps | 1.34% | 171.58 kW | 93.69% |
| 48 steps | 1.61% | 204.97 kW | 90.96% |
| 96 steps | 2.06% | 257.14 kW | 85.50% |

These figures illustrate that errors and explained variance change as the forecast horizon increases.

The architecture demonstrates the potential benefits of explicitly addressing dependencies between multiple future time steps.

### Limitations and relevance to our research

- The evaluation uses international datasets.
- The architecture is substantially more complex than a basic XGBoost or LSTM implementation.
- The study primarily addresses deterministic multi-step forecasting rather than the complete probabilistic evaluation proposed for our project.
- Its results do not establish that the same architecture would work better on India-wide data.

### Relevance

This is an important paper for our multi-step forecasting design.

It establishes the need to treat forecasting horizon as a separate experimental dimension.

For our initial study, we should evaluate a manageable set of horizons such as 1-hour, 6-hour, 24-hour and 48-hour ahead. The 72-hour horizon can be added if the dataset and implementation support it.

We should initially compare XGBoost and LSTM instead of immediately implementing MCNN-MMoL.

---

# 3. SYNTHESIS OF THE LITERATURE

## 3.1 Evolution of forecasting approaches

The reviewed papers illustrate several directions in electricity-load forecasting:

1. MLP models combined with optimization algorithms such as PSO and ALO.
2. LSTM and recurrent models for temporal dependencies in electricity demand.
3. Recurrent sequence-to-sequence architectures with feature attention.
4. Neural architectures that combine historical demand with external weather and calendar information.
5. Multi-step models combining CNN feature extraction with LSTM experts and multi-task learning.
6. Probabilistic and Bayesian deep-learning approaches for uncertainty estimation.

The literature suggests that there is no universally best forecasting model. Performance depends on the dataset, input features, geographical scope, training strategy, prediction horizon and evaluation design.

## 3.2 Importance of external variables

Historical demand is a useful predictor, but it may not capture all demand variation.

PowerNet demonstrates a framework for combining historical consumption with weather and calendar information. Zhao et al. also investigate richer features, including temperature and weekday information where available.

For India-wide electricity demand, we propose investigating:
- Historical demand.
- Temperature.
- Relative humidity.
- Rainfall/precipitation.
- Solar radiation, if appropriate.
- Hour, weekday and month.
- Weekend and holiday indicators.
- Major Indian festival indicators.
- Solar and wind generation, if the variables are available with compatible definitions and timestamps.

These variables should be chosen according to data availability and research justification, not simply because adding variables sounds beneficial.

Feature-ablation experiments should test whether individual feature groups actually improve forecasting.

## 3.3 Indian data and geographical scope

Two of the reviewed papers directly investigate Indian regional electricity demand:
- Bareth et al. use Chhattisgarh data.
- Jain and Gupta use Chandigarh UT data.

These studies establish that ML/DL forecasting has been applied to Indian electricity systems. However, the reviewed studies do not collectively establish the performance of our particular proposed framework for national-level Indian electricity demand.

This is a potential direction for further study, not proof that no similar national-scale research exists. A more complete novelty assessment would require a broader literature search before submitting a publication.

## 3.4 Multi-step forecasting

The Zhu et al. paper shows why multiple forecasting horizons matter.

A model evaluated only on the next time step may behave differently when forecasting an entire day or several days ahead.

Our model evaluation should therefore compare separate forecasting horizons and report performance at each horizon rather than presenting only one aggregate error metric.

## 3.5 Forecasting uncertainty

The Kaur et al. paper distinguishes deterministic point forecasts from probabilistic forecasts.

A point forecast estimates one demand value. An uncertainty-aware forecast can additionally provide a prediction interval.

For instance, a forecast might estimate demand at 200 GW while providing an interval of 192–208 GW.

The interval must be generated from the model or an explicitly defined statistical method and evaluated on unseen data; it must not be invented or treated as automatically reliable.

## 3.6 Computational efficiency

Salehimehr et al. and Kaur et al. both highlight the relationship between predictive performance and computational requirements.

Our comparison should record:
- Training duration.
- Inference duration.
- Hardware and software environment.
- Relevant model complexity or resource usage.

We should report the observed results and avoid claiming that one model is computationally superior before measurement.

---

# 4. CONSOLIDATED RESEARCH GAP

The following statement is our proposed synthesis of the reviewed literature.

**Preliminary research gap:**

Existing studies demonstrate that electricity-load forecasting can be performed using machine learning, recurrent neural networks, LSTM and hybrid neural architectures. The literature also establishes the relevance of weather/calendar features, multi-step forecasting and probabilistic uncertainty estimation.

Within the collection reviewed so far, Indian studies primarily examine individual regions or union territories, while several advanced feature-rich, multi-step and probabilistic approaches are evaluated using international or other region-specific datasets.

There is therefore a potential opportunity to develop and rigorously evaluate an India-wide electricity-demand forecasting framework that integrates historical demand with relevant weather and calendar/festival information and, where data quality permits, renewable-generation variables. The framework would compare XGBoost and LSTM across multiple forecasting horizons and assess accuracy, uncertainty and computational efficiency under a consistent evaluation methodology.

The literature reviewed does not yet establish that this exact combination is novel. Further literature verification must be performed before presenting it as a confirmed research contribution.

## Proposed central research question

"How effectively can XGBoost and LSTM forecast India-wide electricity demand using historical demand, weather and calendar/festival features across multiple forecasting horizons, and how do the models compare in predictive accuracy, uncertainty and computational efficiency?"

## Potential subquestions

1. Does adding weather information improve forecasting compared with using historical demand and calendar features alone?
2. Do Indian holiday/festival indicators improve forecasting accuracy beyond ordinary calendar features?
3. How does the relative performance of XGBoost and LSTM change across forecasting horizons?
4. Can prediction intervals provide useful uncertainty estimates while maintaining acceptable computational cost?
5. How does the inclusion of solar/wind generation information affect forecast performance, if reliable data is available?

These questions are proposed research questions, not claims about results already obtained.

---

# 5. PROPOSED METHODOLOGY

## 5.1 Stage 1: Dataset acquisition and validation

The immediate priority is to identify, obtain and inspect a reliable India-wide electricity-demand dataset.

The primary candidate is:

**Mendeley Data: Electricity Demand, Solar and Wind Generation Data (September 2021–June 2025) of India at 1-hour interval**

URL: https://data.mendeley.com/datasets/y58jknpgs8/2

The dataset is described as using Grid-India/NERLDC data and containing electricity demand and solar/wind generation information.

However, do not assume its exact coverage, regional structure, units or column meanings until the actual workbook files are inspected. Published descriptions may differ from file contents.

The first stage must establish:
- Geographic scope of each load series.
- Whether the available demand data constitutes a national total or separate regional series.
- Exact time range, resolution, time zone and units.
- Missing values, duplicate timestamps and gaps.
- Whether regional data can legitimately be aggregated.
- Availability and definition of solar/wind-generation fields.

## 5.2 Stage 2: Weather data

Investigate suitable historical weather data, such as ERA5-based data distributed through the Open-Meteo Historical Weather API.

URL: https://open-meteo.com/en/docs/historical-weather-api

Select geographically defensible weather coverage based on the electricity dataset's actual geographic structure.

Do not use one city's weather as a proxy for all India without a clear justification.

Weather records must be aligned with the electricity series by time and location or by a documented aggregation method.

Actual future weather observations must not be used as predictors in a simulated real-time forecast unless the experiment explicitly evaluates a retrospective conditional forecast. For an operational forecast, use weather forecasts available at the prediction origin.

## 5.3 Stage 3: Calendar and festival features

Derive calendar features from timestamps:
- Hour.
- Day of week.
- Month.
- Weekend.
- Season.
- Appropriate cyclical encodings, if useful.

Create a documented historical holiday/festival calendar using reliable sources.

Distinguish national holidays from state-specific holidays and avoid inventing dates or assuming that one regional festival affects national demand uniformly.

## 5.4 Stage 4: Data cleaning and exploratory analysis

After verifying the data:
- Preserve immutable raw files.
- Standardize timestamps and column names.
- Document units.
- Diagnose missing periods and duplicates.
- Investigate outliers.
- Examine daily, weekly and seasonal patterns.
- Assess relationships among demand, weather, calendar and generation variables.
- Create a data dictionary and a reproducible preprocessing pipeline.

Do not create synthetic values to make the dataset appear more complete.

## 5.5 Stage 5: Baseline and model comparison

Start with a simple benchmark such as persistence or seasonal-naive forecasting before training more complex models.

Then implement:
- XGBoost as the principal machine-learning model.
- LSTM as the principal deep-learning model.

A simple statistical baseline may be added to understand whether the ML/DL models provide a genuine benefit.

Use the same underlying training, validation and test periods for all models, subject to their differing input-format requirements.

Avoid random train/test splits for time-series forecasting. Use chronological splits.

## 5.6 Stage 6: Multi-step forecasting

Proposed initial horizons:
- 1-hour ahead.
- 6-hour ahead.
- 24-hour ahead.
- 48-hour ahead.

Extend to 72 hours only if the data and methodology justify it.

The implementation must define precisely whether each horizon refers to a single future target or a full sequence of future hourly values.

Compare models on the same forecasting task and avoid using actual future solar/wind-generation values as inputs unless equivalent forecasts would be available at prediction time.

## 5.7 Stage 7: Evaluation

For deterministic forecasts, consider:
- MAE.
- RMSE.
- MAPE, where appropriate.
- An additional scale-aware metric such as WAPE if it provides useful context.
- R² as a supplementary measure, rather than the sole accuracy criterion.

Report results by forecasting horizon and, where informative, by season.

For uncertainty estimation, choose a feasible method and evaluate:
- Prediction-interval coverage.
- Prediction-interval width.
- Appropriate probabilistic loss or calibration metrics.

For computational efficiency, record:
- Training time.
- Inference time.
- Hardware and software configuration.
- Relevant memory/resource usage where practical.

No forecast results or performance improvements should be claimed until experiments are performed.

---

# 6. SCOPE CONTROL

## Core scope

The minimum defensible version of the project should include:

1. A validated Indian electricity-demand time series with a clearly defined geographic scope.
2. Historical-demand and calendar features.
3. Weather features if compatible data can be obtained and defensibly aligned.
4. A baseline forecast.
5. XGBoost and LSTM comparison.
6. Multiple forecast horizons.
7. Accuracy evaluation and computational-efficiency comparison.

## Optional extensions

Only implement these if the data and timeline permit:
- Solar/wind generation features.
- Advanced Indian festival indicators.
- Prediction intervals and uncertainty evaluation.
- EV-related variables.
- More complex hybrid architectures.

Do not force EV-related variables into the core model if reliable historical data is unavailable.

Do not use complex neural architectures only to make the project appear more advanced.

A carefully evaluated and reproducible XGBoost-versus-LSTM comparison on suitable Indian data is preferable to an overly complicated model built on questionable data.

---

# 7. IMPORTANT INSTRUCTIONS FOR THE RESEARCH ASSISTANT

Use this literature review as the background context for my project.

1. Preserve the distinction between what the papers report and what we infer as a potential research gap.
2. Do not invent missing results, variables, data sources or limitations.
3. Treat the research gap as preliminary until a broader novelty search is complete.
4. Do not assume one model will outperform another.
5. Verify any precise technical claim against the actual paper or authoritative publication page.
6. Keep the project achievable for an undergraduate researcher.
7. Prioritize dataset validation over model training.
8. Maintain traceable sources, a data dictionary and reproducible code.
9. Treat historical weather and actual future generation values carefully to avoid forecasting leakage.
10. Do not begin XGBoost/LSTM training until the dataset's geographic scope, resolution, data quality and final target have been established.

## Immediate next step

Begin with dataset acquisition and validation. Inspect the existing project workspace, determine which original data files are available, verify the primary electricity dataset, and document the exact target series before writing model-training code.

The goal is to build a reliable research pipeline supported by actual data, not to generate numerical results prematurely.
