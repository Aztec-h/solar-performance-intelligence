# Resume Points & Interview Q&A

## High-Impact Resume Bullets
- **Machine Learning & Time-Series Forecasting:** Developed a predictive machine learning pipeline using `HistGradientBoostingRegressor` to forecast solar power generation across 42 PV sites, achieving an **R² of 0.934** and an **MAE of 0.75 kW** on a 4-month holdout dataset.
- **Advanced Feature Engineering:** Engineered rolling 1-hour means and temporal lag features from high-granularity weather and irradiance data, reducing model Root Mean Squared Error (RMSE) by **6%** and significantly capturing thermal inertia and cloud movement dynamics.
- **Pipeline Optimization:** Conducted ablation studies to measure feature importance, successfully identifying and removing non-impactful data streams (e.g., wind speed). This streamlined the pipeline, reducing data dependencies and computational overhead for production deployments.
- **Data Engineering:** Aligned, cleaned, and processed over 2 years of 15-minute interval IoT sensor data (160+ MB) using `pandas` and `scikit-learn` ColumnTransformers, implementing robust missing-value imputation strategies without data leakage.

---

## Technical Q&A (Interview Prep)

### Q: Why did you choose HistGradientBoosting over a Deep Learning approach like LSTMs or a simple Linear Regression?
**A:** I started with a linear baseline (`Ridge`), but it performed poorly (R² 0.540) because the relationship between temperature, irradiance, and solar output is highly non-linear (e.g., panels lose efficiency at extreme heat). Deep learning like LSTMs requires significantly more compute and tuning. `HistGradientBoosting` offered the perfect middle ground: it natively handles non-linear interactions, seamlessly processes categorical variables (like `CampusKey`), and trains incredibly fast on tabular data while achieving a highly accurate R² of 0.934.

### Q: How did you handle missing sensor data without introducing data leakage?
**A:** Dropping rows with missing weather data would have caused massive data loss. Instead, I used a `SimpleImputer` to fill missing numerical values with the median (calculated only on the training set to prevent leakage). More importantly, I created explicit boolean indicator columns (e.g., `AirTemperature_missing`). This allowed the model to actually learn patterns associated with sensor downtime, rather than just guessing.

### Q: Why did you engineer "lag" and "rolling mean" features?
**A:** Solar generation isn't just about instantaneous weather. If a cloud covers the sun, the panels might have a slight delay in temperature drop, or the sensor might have a slight reporting lag. By engineering 1-hour lagged GHI (Global Horizontal Irradiance) and 1-hour rolling means, I gave the model a "short-term memory." This single addition reduced the model's RMSE from 2.58 down to 2.43.

### Q: Why did you remove Wind Speed from the final model?
**A:** Domain knowledge suggests wind cools solar panels, improving efficiency. However, when I ran experiments with and without wind speed, the MAE and RMSE remained identical (MAE 0.75). Since it added no predictive power but required maintaining a separate data ingestion pipeline, I removed it to simplify the system architecture and reduce maintenance costs.

---

## Business Value Proposition (What it brings to Australian Power Plants)

If asked: *"How is this useful for solar farms in Australia?"*

1. **Site Feasibility Simulation:** Before investing millions into building a new solar farm, investors can feed decades of historical weather data for a specific Australian latitude/longitude into this model. It will accurately simulate the expected power yield, completely de-risking the financial investment.
2. **Predictive Maintenance & Anomaly Detection:** In production, if a site is generating 150 kW, but the model predicts it *should* be generating 200 kW based on the current perfect weather, it immediately flags a mechanical issue (e.g., inverter failure, severe dust on panels). This enables targeted maintenance rather than schedule-based maintenance.
3. **Grid Supply Forecasting (AEMO Integration):** The Australian Energy Market Operator (AEMO) struggles with the "duck curve." By feeding 24-hour weather forecasts into this model, grid operators know exactly how much solar power will hit the grid tomorrow, allowing them to precisely schedule backup coal/gas plants and stabilize prices.
