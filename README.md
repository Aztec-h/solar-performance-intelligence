# Solar Performance Intelligence

## Project Overview
This project aims to develop a highly accurate, predictive machine learning pipeline for forecasting solar power generation across multiple sites in Australia. By leveraging high-granularity weather, irradiance, and historical solar generation data, this pipeline provides actionable intelligence for solar farm operations, predictive maintenance, and site planning.

## Dataset
We make use of the [UNISOLAR Dataset](https://www.kaggle.com/datasets/pythonafroz/solar-power-generation-and-energy-consumption-data/data), which contains high-granularity PV solar energy generation, solar irradiance, and weather data from 42 PV sites deployed across five campuses at La Trobe University, Victoria, Australia.

- The dataset spans over 2 years (2020-01-01 to 2022-04-24) with a 15-minute interval between readings.

## Repository Structure
```text
.
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── models                     # Trained model artifacts (.joblib)
├── reports                    # Performance metrics, deep-dive reports, challenges
├── scripts
│   ├── run_app.bat
│   └── run_pipeline.bat
├── src                        # Source code for data processing and modeling
│   ├── config
│   ├── data
│   ├── features
│   ├── pipelines
│   └── utils
├── README.md
└── requirements.txt
```

## Features & Schema
| Feature             | Source       | Type        | Usage |
| ------------------- | ------------ | ----------- | ----- |
| Timestamp           | Generation   | Datetime    | Temporal feature extraction |
| CampusKey           | Generation   | Categorical | Site Identification |
| SiteKey             | Generation   | Categorical | Site Identification |
| SolarGeneration     | Generation   | Numeric     | **Target Variable** |
| GHI                 | Irradiance   | Numeric     | Predictor |
| CloudOpacity        | Irradiance   | Numeric     | Predictor |
| AirTemperature      | Weather      | Numeric     | Predictor |
| RelativeHumidity    | Weather      | Numeric     | Predictor |
| WindSpeed           | Weather      | Numeric     | Dropped (No predictive value) |
| Latitude / Longitude| Site Details | Numeric     | Geolocation |
| kWp                 | Site Details | Numeric     | Site Capacity |

In addition to the raw features, we engineered several time-series specific features (1-hour lags, 1-hour rolling means for GHI and Cloud Opacity) and temporal indicators (is_daylight, Hour, Quarter) to improve predictive accuracy.

## Environment Setup
To Setup the environment "solar-perf-intel" in windows
1. Run the setup script:
   ```cmd
   scripts\run_setup.bat
   ```

## Running the Pipeline
To execute the data processing and model training pipeline:
1. Ensure the raw CSVs are placed in `data/raw/` or the `.zip` file is extracted.
2. Run the pipeline script:
   ```cmd
   scripts\run_pipeline.bat
   ```
3. To generate the V2 Advanced Analytics reports, run:
   ```cmd
   python -m src.pipelines.v2_advanced_analytics
   ```

## V2 Advanced Analytics
We have expanded the project beyond predictive modeling into a robust intelligence platform. Key V2 features include:
- **Feature Stability Analysis** (understanding temporal shifts in feature importance)
- **Causal Inference & Impact Analysis** (quasi-experimental design to determine causality, e.g., temperature impact)
- **Temporal & Site Clustering** (identifying operational patterns)
- **Granular Error Analysis & Segmentation**
- **Multivariate Policy Analysis** (simulating What-If scenarios like +2°C warming)
- **Forecasting Uncertainty & Drift Detection**

For full details on the implementations, including alternatives considered and rejected, please view the [V2 Documentation](reports/V2/DOCUMENTATION.md).

## Running the streamlit report app
To run the streamlit app:
1. Run the app script:
   ```cmd
   scripts\run_app.py
   ```

## References
[1] S. Wimalaratne, D. Haputhanthri, S. Kahawala, G. Gamage, D. Alahakoon and A. Jennings, "UNISOLAR: An Open Dataset of Photovoltaic Solar Energy Generation in a Large Multi-Campus University Setting," 2022 15th International Conference on Human System Interaction (HSI), 2022, pp. 1-5, doi: 10.1109/HSI55341.2022.9869474.