# Overall Project Resume Points & Interview Q&A (V1 & V2)

## High-Impact Resume Bullets

- **Predictive Modeling & Machine Learning:** Developed an end-to-end predictive pipeline using `HistGradientBoostingRegressor` to forecast solar generation across 42 PV sites, achieving an **R² of 0.934** and an **MAE of 0.75 kW** on a 4-month holdout dataset.
- **Advanced Data Engineering & Feature Extraction:** Aligned and processed 2+ years of 15-minute interval IoT sensor data (>160 MB). Engineered time-series specific features (1-hour lags, rolling means) to capture thermal inertia, reducing overall model RMSE by **6%**.
- **Causal Inference & Impact Analysis:** Designed a quasi-experimental framework using statistical matching and Welch’s t-tests to isolate and quantify the causal degradation of panel efficiency due to high temperatures (>30°C), controlling for confounding variables like solar irradiance.
- **Explainability & Stability Tracking:** Implemented a Feature Stability Analysis surrogate framework (in lieu of computationally expensive exact SHAP values) to track how environmental variables' predictive importance shifts across seasons, providing crucial "white-box" insights to stakeholders.
- **Multivariate Policy Simulation:** Built a "What-If" scenario engine to simulate long-term operational impacts, such as projecting total yield loss under a +2°C global warming scenario to guide future hardware investment strategies.
- **Automated Error & Drift Diagnostics:** Engineered automated systems to isolate the worst 5% of predictions for targeted geographical debugging (Segmentation), and deployed Kolmogorov-Smirnov (KS) tests to monitor for multi-year weather distribution drift.

---

## Technical Q&A (Interview Prep)

### Q: Why did you choose HistGradientBoosting over Deep Learning (LSTMs) or Linear Regression?
**A:** I started with a baseline linear model (`Ridge`), but it performed poorly (R² 0.540) because the relationship between temperature, irradiance, and output is highly non-linear (panels lose efficiency at extreme heat). Deep learning like LSTMs requires significantly more compute, tuning, and infrastructure to deploy. `HistGradientBoosting` offered the perfect middle ground: it natively handles non-linear interactions, seamlessly processes categorical variables (like `CampusKey`), and trains incredibly fast on tabular data while achieving a highly accurate R² of 0.934.

### Q: How did you handle missing sensor data without introducing data leakage?
**A:** Dropping rows with missing weather data would have caused massive data loss. Instead, I used a `SimpleImputer` to fill missing numerical values with the median (calculated *only* on the training set to prevent leakage). More importantly, I created explicit boolean indicator columns (e.g., `AirTemperature_missing`). This allowed the model to actually learn patterns associated with sensor downtime, rather than just guessing.

### Q: Why did you engineer "lag" and "rolling mean" features?
**A:** Solar generation isn't just about instantaneous weather. If a cloud covers the sun, the panels might have a slight delay in temperature drop, and sensors have a reporting lag. By engineering 1-hour lagged GHI (Global Horizontal Irradiance) and 1-hour rolling means, I gave the model a "short-term memory." This specific addition dropped the model's RMSE from 2.58 down to 2.43.

### Q: You mentioned Causal Inference. Why did you use Quasi-Experimental matching instead of a formal Causal Graph (like DoWhy)?
**A:** To build a robust causal graph, you need extensive domain-specific covariate data—things like inverter models, panel angles, or maintenance logs, which we didn't have. Instead, I used a Quasi-Experimental design on historical data. By utilizing statistical t-tests, I isolated periods of extremely high irradiance (GHI > 500) and compared the actual generation output when it was very hot versus mild. This allowed me to prove and quantify temperature-induced degradation while effectively holding the primary driver (the sun) constant, providing highly rigorous insights without over-complicating the architecture.

### Q: Why didn't you use exact SHAP values for explainability?
**A:** Exact SHAP computation on tree ensembles with hundreds of thousands of rows is computationally intensive. In a production environment or real-time dashboard, calculating exact SHapley values for every prediction causes severe latency. I opted for a *Feature Stability Analysis* surrogate approach—I analyzed the global feature importances and how they shifted temporally (e.g., Temperature becoming more critical in Summer). This provided the necessary "white-box" transparency to stakeholders without the crippling computational overhead.

### Q: How does your Multivariate Policy Simulation work?
**A:** I built a simulation wrapper around the trained pipeline. Instead of retraining models for every "What-If" scenario, we feed altered data matrices into the existing, validated model. For example, to test a global warming scenario, I duplicated the test set, applied a +2°C shift to the `AirTemperature` feature, and ran it through the model. We can then calculate the exact percentage drop in expected generation, giving financial stakeholders hard data on potential future yield loss.

---

## Business Value Proposition (What it brings to Australian Power Plants)

If asked: *"How is this useful for solar farms or grid operators in Australia?"*

1. **Site Feasibility & Financial De-Risking:** Before investing millions into a new solar farm, investors can feed decades of historical weather data for a specific Australian longitude/latitude into the simulation engine. It accurately predicts the expected power yield under various policy/climate scenarios, completely de-risking the financial investment.
2. **Predictive Maintenance:** In production, if a site generates 150 kW, but the model predicts it *should* be generating 200 kW based on the exact current weather, the system flags a mechanical anomaly (inverter failure, dust accumulation). The automated error segmentation explicitly isolates *which* campus and *when* this happens, enabling targeted, cost-effective maintenance.
3. **Grid Supply Forecasting (AEMO Integration):** The Australian Energy Market Operator (AEMO) struggles with the "duck curve" caused by sudden influxes of solar power. By feeding 24-hour weather forecasts into this model, grid operators know precisely how much solar power will hit the grid tomorrow. Furthermore, our Forecasting Uncertainty intervals (5th/95th percentiles) provide operators with confidence bounds, allowing them to dynamically schedule backup coal/gas plants safely and stabilize market prices.
