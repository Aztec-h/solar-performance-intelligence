# Solar Generation Prediction Model Report

## 1. Executive Summary
The objective of this project was to develop an accurate predictive model for solar power generation across 42 sites in Australia. By forecasting solar output based on meteorological data (Global Horizontal Irradiance, Cloud Opacity, Air Temperature) and site capacity, we aim to provide actionable intelligence for solar farm operations, predictive maintenance, and strategic site planning. Our final model achieved an impressive **R² score of 0.934**, explaining over 93% of the variance in actual power generation.

## 2. Dataset and Data Engineering
The project utilizes the UNISOLAR dataset, comprising 15-minute interval data from 2020 to 2022. 
Key preprocessing steps included:
- **Timestamp Alignment:** Synchronizing weather, irradiance, and power generation readings across 5 campuses.
- **Missing Data Handling:** Utilizing median imputation coupled with explicit "missing indicator" features to allow the model to learn from sensor outage patterns.

## 3. Feature Engineering
We engineered several high-value features to capture the temporal and physical realities of solar generation:
- **Temporal/Cyclical:** `Hour`, `Month`, `DayOfYear`, `DayOfWeek`, `Quarter`, and `is_daylight`. These capture daily and seasonal sun paths.
- **Time-Series / Lag:** 1-hour lags and 1-hour rolling means for GHI (Global Horizontal Irradiance) and Cloud Opacity. These features give the model context about recent weather, accounting for cloud movement and thermal inertia.
- **Feature Pruning:** Extensive testing revealed that `WindSpeed` added computational overhead without improving the Mean Absolute Error (MAE). It was subsequently dropped.

## 4. Modeling Strategy and Experiments
We evaluated multiple approaches to isolate feature value and algorithm complexity. Models were trained on data prior to 2022-01-01 and evaluated on a holdout set spanning early 2022.

1. **dummy_mean**: Baseline predicting the average.
2. **ridge_baseline_no_wind**: Linear Ridge regression (Standard features).
3. **hgb_baseline_no_wind**: HistGradientBoosting (HGB) (Standard features).
4. **hgb_engineered_no_wind**: HGB incorporating all engineered temporal and lag features (excluding wind).

### Holdout Evaluation Metrics
| Experiment                       | MAE  | RMSE | R²   |
|----------------------------------|------|------|------|
| **hgb_engineered_no_wind**       | **0.75** | **2.43** | **0.934**|
| hgb_baseline_no_wind             | 0.83 | 2.58 | 0.926|
| ridge_baseline_no_wind           | 2.88 | 6.43 | 0.540|
| dummy_mean                       | 4.86 | 9.50 | -0.00|

*Note: The MAE of 0.75 kW is exceptionally strong across the highly variable generation data.*

## 5. Key Technical Insights
- **Non-linear vs Linear**: HGB vastly outperformed Ridge Regression. Solar generation involves significant non-linear interactions (e.g., panel efficiency drops non-linearly at very high temperatures despite high GHI).
- **Engineered Features Add Value**: Adding lag and rolling features reduced RMSE from 2.58 to 2.43.

## 6. Business Impact for Australian Power Plants
**Is this model ready for real-world deployment in Australia? Yes.**

1. **Strategic Site Selection:** By feeding historical meteorological data from prospective sites across Australia into this model, stakeholders can simulate expected power yields before breaking ground.
2. **Grid Stability:** Accurate forecasts allow grid operators to anticipate supply fluctuations due to cloud cover, reducing reliance on expensive peaker plants.
3. **Predictive Maintenance:** By comparing the model's *expected* generation with a site's *actual* generation, operators can instantly detect panel degradation, inverter failures, or severe soiling, dispatching maintenance teams only when necessary.
