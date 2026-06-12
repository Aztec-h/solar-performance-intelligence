# Overall Challenges Faced During the Project (V1 & V2)

Building a robust predictive model and advanced analytics platform for solar performance generation from high-granularity IoT sensor data presented a wide range of distinct challenges, spanning from initial data engineering to complex causal inference.

## 1. Data Alignment and Granularity (V1)
*   **Challenge:** The dataset contained multiple CSV files (generation, irradiance, weather) recorded at different intervals or with slight timestamp mismatches across 42 PV sites.
*   **Solution:** Implemented a robust alignment strategy using pandas, standardizing the temporal resolution to 15-minute intervals and forward-filling/interpolating within safe time windows to prevent data leakage.

## 2. Dealing with Missing Sensor Data (V1)
*   **Challenge:** Weather sensors occasionally failed or had delayed reporting, resulting in NaN values for Air Temperature, Relative Humidity, and Wind Speed. Dropping these rows would result in significant data loss.
*   **Solution:** Utilized a `SimpleImputer` with a median strategy combined with missing-value indicator columns (`AirTemperature_missing`, etc.) so the model could learn patterns associated with sensor downtime.

## 3. High Variance and Non-Linear Relationships (V1)
*   **Challenge:** Initial linear models (Ridge Regression) struggled to capture the complex relationship between cloud opacity, irradiance, temperature, and actual solar output, yielding a poor R² score of 0.540.
*   **Solution:** Pivoted to tree-based ensemble methods, specifically `HistGradientBoostingRegressor`. This algorithm natively handles non-linear interactions and categorical features (like CampusKey), dramatically boosting R² to 0.934.

## 4. Feature Engineering for Time-Series (V1)
*   **Challenge:** Solar generation is highly dependent on recent past states, not just instantaneous measurements. A snapshot of weather isn't enough to predict generation due to thermal inertia and sensor lag.
*   **Solution:** Engineered rolling 1-hour means and 1-hour lagged features for Global Horizontal Irradiance (GHI) and Cloud Opacity. This specific step reduced the MAE (Mean Absolute Error) significantly by giving the model a "memory" of recent weather patterns.

## 5. Pruning Unnecessary Data (V1)
*   **Challenge:** We initially assumed Wind Speed would be a crucial feature due to its cooling effect on PV panels (which generally improves efficiency). However, including it increased dataset dependency without a corresponding drop in RMSE.
*   **Solution:** Conducted ablation studies to measure feature importance. We concluded that `WindSpeed` yielded no tangible benefit and removed it from the final pipeline, streamlining data collection requirements for future site deployments.

## 6. Explainability vs. Computational Overhead (V2)
*   **Challenge:** Tree-based ensemble models act as "black boxes." Fully computing exact SHAP (Shapley Additive exPlanations) values for a large dataset (over a million rows) via `HistGradientBoostingRegressor` is computationally prohibitive, making real-time dashboards sluggish.
*   **Solution:** Instead of computing exact SHAP for every prediction, we pivoted to a *Feature Stability Analysis* framework. This surrogate approach extracts global and localized insights (e.g., tracking how `AirTemperature` importance fluctuates between seasons), maintaining fast inference while still delivering actionable explanations.

## 7. Establishing Causality without Extensive Confounders (V2)
*   **Challenge:** Differentiating between correlation and causation (e.g., "Does heat actually cause panel degradation, or is it just correlated with sunny days?"). Building a complete Causal Graph (using libraries like DoWhy) was unfeasible due to a lack of domain-specific covariate data (like inverter models or panel maintenance logs).
*   **Solution:** Implemented a robust Quasi-Experimental Design on historical data. By utilizing Welch's t-test, we isolated high GHI (>500) periods and compared the generation of panels under high heat (>30°C) versus low heat, proving significant causal temperature-induced degradation while controlling for irradiance.

## 8. Identifying Systematic Failure Modes (V2)
*   **Challenge:** A high overall R² score can mask severe localized failures. Determining exactly *where* and *why* the model failed across hundreds of different dimensions (Campus, Hour, Month, Weather State) was like finding a needle in a haystack.
*   **Solution:** Developed an automated Error Analysis engine that isolates the worst 5% of predictions (based on Absolute Error) and segments them temporally and geographically. This highlighted specific campuses and specific times of day that contributed disproportionately to model drift, enabling targeted debugging.
