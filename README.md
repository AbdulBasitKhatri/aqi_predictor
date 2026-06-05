# AQI Predictor for Karachi - 10Pearls Internship Project

## Executive Summary

This repository documents the complete internship project developed for 10Pearls: a production-oriented Air Quality Index (AQI) prediction platform for Karachi. The system combines data ingestion, feature engineering, exploratory analysis, machine learning, explainability, model registry integration, and deployment automation.

The project addresses the problem of forecasting hourly AQI values, enabling operational insight into pollution spikes and air quality trends. It is built as an end-to-end MLOps solution using Hopsworks, OpenWeather APIs, Streamlit, and GitHub Actions.

The application is also deployed on the cloud at: https://aqipredictor-a-basit-khatri-10pearls.streamlit.app/

## Project Context

### Business problem

Karachi experiences frequent air quality degradation due to pollution sources and meteorological conditions. A reliable short-term forecast empowers decision-makers and residents to prepare for poor air quality, expect pollution events, and allocate mitigation efforts.

### Internship objective

As part of the 10Pearls internship, the objective was to:
- build a robust AQI forecast model,
- centralize features in a feature store,
- automate training and ingestion workflows,
- create a live inference dashboard,
- demonstrate explainability with SHAP and LIME,
- document the full MLOps lifecycle.

### Scope

The project focuses on:
- next-hour AQI prediction,
- 72-hour forecast display,
- live feature ingestion from OpenWeather,
- historical backfill to create a training dataset,
- reproducible CI/CD-driven retraining.

## System Overview

### Key components

- `current_feature_pipeline.py`: fetches current AQI and weather observations, computes features, inserts data into Hopsworks.
- `backfill_feature.py`: downloads historical AQI observations, engineers features, and populates the feature store.
- `training_pipeline.py`: prepares data, trains multiple models, evaluates candidates, generates explainability artifacts, and registers the best model.
- `app.py`: Streamlit inference app that downloads the latest model from Hopsworks and serves 72-hour forecasts.
- `eda.py`: exploratory analysis that validates the feature store dataset, reveals correlations, and exposes data quality issues.
- `.github/workflows`: GitHub Actions pipelines for hourly ingestion and daily retraining.

### Architecture diagram (logical)

1. Data sources: OpenWeather air pollution and weather endpoints.
2. Feature engineering layer: local transformation scripts.
3. Hopsworks Feature Store: centralized feature storage and retrieval.
4. Training and registry layer: model training, SHAP/LIME explainability, registry upload.
5. Serving layer: Streamlit dashboard pulling latest model artifacts.
6. Automation layer: GitHub Actions scheduled workflows.

## Data Sources and Feature Store

### OpenWeather APIs

The pipelines use OpenWeather because it provides public air pollution and weather data for Karachi:
- `air_pollution`: current pollutant concentrations and AQI index.
- `weather`: current temperature, humidity, and wind speed.
- `air_pollution/history`: historical AQI values used for backfill.

### Hopsworks Feature Store

This project depends on Hopsworks for feature storage and model registry management.

The feature group is configured as:
- `name = "karachi_aqi_fg"`
- `version = 1`
- `primary_key = ["city", "timestamp"]`
- `event_time = "timestamp"`
- `online_enabled = True`

This ensures:
- deduplicated feature records,
- event-time aware ingestion,
- consistency between batch and online retrieval.

### Feature ingestion strategy

Two ingestion paths were implemented:
- `backfill_feature.py`: historical dataset creation from January 1, 2026 to May 15, 2026.
- `current_feature_pipeline.py`: hourly ingestion of live observations using API keys stored in `.env`.

The two-stage ingestion strategy separates cold-start training data from ongoing operational data.

## Exploratory Data Analysis (EDA)

### Purpose of EDA

EDA was used to:
- verify data completeness,
- quantify missing values,
- inspect the distribution of AQI,
- discover pollutant-AQI relationships,
- identify temporal seasonality and weekend effects.

### Actions performed in `eda.py`

- read the feature group from Hopsworks,
- print data schema and sample rows,
- calculate missing value counts,
- plot AQI distribution with histograms,
- compute correlation matrix for numeric features,
- scatterplot pollutant features against AQI,
- compare AQI on weekends versus weekdays.

### Key EDA findings

- The raw dataset had irregular timestamp spacing and incomplete records.
- AQI distribution was skewed, with frequent moderate pollution and occasional extreme spikes.
- Pollutant features such as `pm25`, `pm10`, `no2`, and `so2` showed strong positive correlation with AQI.
- Temporal signals such as `hour`, `day_of_week`, and weekend indicators were meaningful.

### Interpretation

The EDA confirmed that the model needed to combine:
- pollutant intensity,
- recent AQI history,
- calendar effects,
- short-term trend behavior.

This guided the subsequent feature engineering and time-series modeling decisions.

## Feature Engineering

The project constructs a multi-dimensional feature set to represent air quality dynamics.

### Raw features

- `aqi`: current Air Quality Index.
- `pm25`, `pm10`, `no2`, `so2`, `o3`, `co`: pollutant concentrations.
- `temp`, `humidity`, `wind_speed`: weather context.

### Calendar features

- `hour`: hour of the day.
- `day`: day of the month.
- `month`: month of the year.
- `day_of_week`: weekday index.
- `is_weekend`: weekend indicator.

These features expose daily and weekly cycles that influence pollution dispersion.

### Aggregated features

- `pollutant_sum`: the sum of the six pollutant values, capturing overall pollution load.
- `temp_humidity_index`: the product of temperature and humidity, capturing combined atmospheric conditions.
- `wind_temp_ratio`: a ratio that reflects the effect of wind relative to temperature.
- `aqi_change_rate`: the rate of AQI change between consecutive records.

These engineered features capture domain-specific interactions and help the model generalize.

### Time-series features

- `aqi_lag_1`: AQI one hour prior.
- `aqi_lag_3`: AQI three hours prior.
- `aqi_lag_6`: AQI six hours prior.
- `aqi_lag_24`: AQI 24 hours prior.
- `aqi_rolling_mean_6h`: 6-hour moving average.
- `aqi_rolling_std_6h`: 6-hour moving standard deviation.
- `aqi_rolling_mean_24h`: 24-hour moving average.

These features encode persistence, volatility, and temporal context.

### Target label

- `target_aqi` is defined as `aqi.shift(-1)` in the training pipeline.
- This makes the model a one-hour-ahead forecaster.
- The forecast horizon for display is extended to 72 hours using recursive prediction logic in the dashboard.

## Data Preprocessing

### Timeline cleanup

The training pipeline cleans the dataset by:
- converting timestamps to `datetime`,
- sorting by timestamp,
- resampling to strict hourly frequency,
- averaging duplicate hours,
- linearly interpolating missing hours,
- dropping rows with NaN values from lag or rolling calculations.

This processing pipeline ensures that the model trains on uniformly spaced time-series data.

### Train/test split

- A chronological split is used rather than a randomized split.
- The first 80% of observations become the training set.
- The final 20% becomes the test set.

This preserves temporal causality and simulates a real deployment environment.

## Model Development

### Candidate models

Multiple regression algorithms were evaluated using a standardized pipeline.

- Linear regression family: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`.
- Tree-based estimators: `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`.
- Kernel and instance-based models: `SVR`, `KNeighborsRegressor`.
- Gradient boosting libraries: `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`.

### Pipeline design

Each model is wrapped in a Scikit-learn `Pipeline`:
- `StandardScaler` for feature normalization,
- followed by the regression estimator.

This ensures that scaling is applied consistently during both training and inference.

### Evaluation metrics

The model comparison uses:
- MAE (mean absolute error),
- RMSE (root mean squared error),
- R2 score.

The best model is selected by the highest test-set R2 score.

### Results artifacts

The training script saves:
- `aqi_model/metrics.csv` - performance scores for all candidate models,
- `aqi_model/best_aqi_model.pkl` - the best trained pipeline,
- `aqi_model/predictions.csv` - actual vs predicted values on test data,
- `aqi_model/feature_importance.csv` - if the chosen model exposes feature importances,
- SHAP plots: `aqi_model/shap_summary.png` and `aqi_model/shap_bar.png`,
- LIME instance explanation: `aqi_model/lime_explanation.html`.

### Model registry upload

The selected model is uploaded to Hopsworks Model Registry as a Python model with metadata and metrics.

The registry entry includes:
- model name: `karachi_aqi_predictor`,
- metrics: `r2_score`, `mae`, `rmse`,
- description that documents feature usage and explainability.

This enables versioned model management and future governance.

## Explainability and Interpretability

Explainability is a core part of the project, not an afterthought.

### SHAP explanation

The pipeline generates SHAP summary plots that show:
- the most important features globally,
- the direction of influence for each feature,
- whether high or low feature values tend to increase AQI predictions.

Interpretation of SHAP results answers questions such as:
- are recent AQI values more influential than raw pollutant concentrations?
- does the model trust 24-hour trends or immediate short-term volatility?
- how much do temporal features like `hour` and `day_of_week` contribute?

### LIME explanation

A single instance explanation is generated using LIME.

The `lime_explanation.html` artifact explains:
- which features drove a specific forecast,
- how each feature contributed positively or negatively,
- the local decision boundary around the chosen instance.

This is especially useful for stakeholder review and debugging unusual predictions.

### Feature importance

If the final model exposes `feature_importances_`, the script writes them to `feature_importance.csv`.

These scores help quantify the relative predictive power of features and validate domain assumptions.

## Deployment and Inference

### Streamlit application

`app.py` is the serving application for inference and analysis.

It:
- authenticates to Hopsworks using `HOPSWORKS_API_KEY`,
- retrieves the latest registered `karachi_aqi_predictor` model version,
- downloads model assets and explainability files,
- reads the latest feature data from the Hopsworks feature group,
- computes a 72-hour recursive forecast using the latest observed state,
- displays model metrics, forecast summaries, and explainability visualizations.

### Recursive forecasting logic

The dashboard uses a recursive prediction loop for the 72-hour horizon:
- it starts from the latest available observation,
- uses predicted AQI values as inputs for subsequent future steps,
- updates lag and rolling features dynamically,
- applies a simple decay on pollutant concentrations for future hours,
- uses calendar features derived from future timestamps.

This approximates a short-term forecast horizon while allowing only one model to be served.

### Inference presentation

The app delivers the following to end users:
- top-level model metrics from the latest model package,
- 3-day AQI summary cards,
- SHAP global summary and bar plots,
- local LIME explanation embedded in the UI.

This makes the system useful for both predictions and diagnostics.

## CI/CD and Automation

### GitHub Actions workflows

Two workflows automate the MLOps pipeline:
- `.github/workflows/feature_insert.yml`: hourly feature ingestion using `current_feature_pipeline.py`.
- `.github/workflows/model_training.yml`: daily training using `training_pipeline.py`.

### CI process

Each workflow:
- checks out the repository,
- sets up Python 3.11,
- installs dependencies from `requirements.txt`,
- reads required secrets from GitHub repository settings,
- executes the relevant pipeline script.

### Operational benefits

Automation ensures:
- timely ingestion of new AQI and weather data,
- scheduled refresh of the model using updated historical data,
- environment-level reproducibility,
- reduced manual intervention and fewer operational errors.

## Implementation Details

### Environment

Dependencies are declared in `requirements.txt`, which includes:
- `hopsworks` for feature store and model registry,
- `streamlit` for the dashboard,
- `scikit-learn` for modeling,
- `xgboost`, `lightgbm`, `catboost` for boosted learners,
- `shap` and `lime` for explainability,
- `pandas`, `numpy`, `matplotlib`, `seaborn` for data handling and visualization.

### Authentication and secrets

The project relies on environment variables stored in `.env`:
- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY`

In GitHub Actions, these are passed as repository secrets.

### File-level implementation notes

- `current_feature_pipeline.py` builds current live features and inserts them with `fs.get_or_create_feature_group(...)`.
- `backfill_feature.py` backfills historical AQI values and uses placeholder weather feature values when historical weather is unavailable.
- `training_pipeline.py` trains multiple models, selects the best by R2, generates SHAP/LIME artifacts, and uploads the model to the registry.
- `app.py` downloads the latest model package, reads `metrics.csv`, `shap_summary.png`, `shap_bar.png`, `feature_importance.csv`, and `lime_explanation.html`.
- `eda.py` validates the feature store dataset and supports manual data quality review.

## Challenges and Resolutions

### 1. Data irregularity and missing timestamps

- Challenge: OpenWeather historical data did not arrive uniformly at exact hourly intervals, which threatened the validity of lag and rolling features.
- Resolution: resampled the timeline to hourly frequency, aggregated duplicates by mean, interpolated missing data, and dropped rows with incomplete lag windows.
- Outcome: the training dataset became consistent and suitable for time-series modeling.

### 2. Feature store structuring

- Challenge: designing a feature group schema that supports both history and live ingestion.
- Resolution: used a composite key of `city` + `timestamp` with `event_time="timestamp"` in Hopsworks.
- Outcome: a stable ingestion pipeline that can be rerun hourly without duplicate rows.

### 3. Model explainability

- Challenge: selecting a model cannot rely solely on accuracy metrics in a production setting.
- Resolution: generate SHAP and LIME artifacts to make the model interpretable.
- Outcome: stakeholders can validate that the model’s decisions align with domain expectations.

### 4. CI/CD integration

- Challenge: without automation, manual retraining and ingestion were error-prone.
- Resolution: configured GitHub Actions for scheduled hourly ingestion and daily training.
- Outcome: the MLOps workflow became reproducible and operationally sustainable.

## Lessons Learned

### Feature store adoption

I learned to treat the feature store as the canonical data layer. This project demonstrated how consistent feature schema and event-time semantics are critical for production-quality ML workflows.

### The importance of EDA

The project reinforced that EDA is not optional. Visualizing data distributions and correlations prevented misleading feature assumptions and improved model robustness.

### Explainable AI

By incorporating SHAP and LIME, I learned how to move beyond black-box models. These tools made the model behavior transparent and provided valuable trust signals.

### Production readiness

This internship project taught me the gap between a working model and a deployable system. Building CI/CD, registry integration, and a dashboard created a full lifecycle pipeline.

## Future Work

Potential improvements for the next phase:
- add explicit hyperparameter tuning and cross-validation,
- support online feature retrieval for real-time serving,
- implement drift detection and alerting,
- add test coverage for pipelines,
- package the app in Docker and deploy to cloud infrastructure,
- enhance the backfill pipeline with real historical weather values.

## Getting Started

### Prerequisites

- Python 3.11
- GitHub repository configured with secrets:
  - `OPENWEATHER_API_KEY`
  - `HOPSWORKS_API_KEY`
- Local `.env` file containing the same keys for local execution.

### Setup

```bash
pip install -r requirements.txt
```

### Run pipelines

```bash
python backfill_feature.py
python current_feature_pipeline.py
python training_pipeline.py
```

### Run the dashboard

```bash
streamlit run app.py
```

## Repository Contents

- `app.py`
- `training_pipeline.py`
- `current_feature_pipeline.py`
- `backfill_feature.py`
- `eda.py`
- `requirements.txt`
- `.github/workflows/feature_insert.yml`
- `.github/workflows/model_training.yml`

## Notes

- The backfill path currently sets historical weather values to zeros for features such as `temp`, `humidity`, and `wind_speed` due to OpenWeather free-tier limitations.
- The Streamlit app uses a recursive forecasting strategy for 72-hour output and displays daily maxima as summary values.
- This project is organized as an internship report and a working MLOps deliverable.

---

## Contact

For questions or extensions, use the configured API keys and run the pipelines in the documented order. This repository is intended to be a reproducible learning artifact from the 10Pearls internship.
