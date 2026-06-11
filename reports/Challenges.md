# Challenges Faced During the Project

Building a robust predictive model for solar performance generation from high-granularity IoT sensor data presented several distinct challenges:

## 1. Data Alignment and Granularity
- **Challenge:** The dataset contained multiple CSV files (generation, irradiance, weather) recorded at different intervals or with slight timestamp mismatches across 42 PV sites.
- **Solution:** Implemented a robust alignment strategy using pandas, standardizing the temporal resolution to 15-minute intervals and forward-filling/interpolating within safe time windows to prevent data leakage.

## 2. Dealing with Missing Sensor Data
- **Challenge:** Weather sensors occasionally failed or had delayed reporting, resulting in NaN values for Air Temperature, Relative Humidity, and Wind Speed. Dropping these rows would result in significant data loss.
- **Solution:** Utilized a `SimpleImputer` with a median strategy combined with missing-value indicator columns (`AirTemperature_missing`, etc.) so the model could learn patterns associated with sensor downtime.

## 3. High Variance and Non-Linear Relationships
- **Challenge:** Initial linear models (Ridge Regression) struggled to capture the complex relationship between cloud opacity, irradiance, temperature, and actual solar output, yielding a poor R² score of 0.540.
- **Solution:** Pivoted to tree-based ensemble methods, specifically `HistGradientBoostingRegressor`. This algorithm natively handles non-linear interactions and categorical features (like CampusKey), dramatically boosting R² to 0.934.

## 4. Feature Engineering for Time-Series
- **Challenge:** Solar generation is highly dependent on recent past states, not just instantaneous measurements. A snapshot of weather isn't enough to predict generation due to thermal inertia and sensor lag.
- **Solution:** Engineered rolling 1-hour means and 1-hour lagged features for Global Horizontal Irradiance (GHI) and Cloud Opacity. This specific step reduced the MAE (Mean Absolute Error) significantly by giving the model a "memory" of recent weather patterns.

## 5. Pruning Unnecessary Data (Wind Speed)
- **Challenge:** We initially assumed Wind Speed would be a crucial feature due to its cooling effect on PV panels (which generally improves efficiency). However, including it increased dataset dependency without a corresponding drop in RMSE.
- **Solution:** Conducted ablation studies to measure feature importance. We concluded that `WindSpeed` yielded no tangible benefit and removed it from the final pipeline, streamlining data collection requirements for future site deployments.
