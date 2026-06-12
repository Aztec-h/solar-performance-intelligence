# Future Plan: Advanced Analytics & Modeling 

This document outlines the strategic roadmap for introducing advanced analytical and machine learning capabilities into the Solar Performance Intelligence platform. The focus is on deepening our understanding of the data, improving model explainability, and extracting actionable insights for policy and operational improvements.

## 1. SHAP (Shapley Additive exPlanations) & Explainability
**Objective:** Move beyond "black box" predictions to understand exactly *why* models make specific predictions for solar output.
*   **Global Interpretability:** Generate SHAP summary plots to understand overall feature importance (e.g., how much does cloud cover vs. time of day impact generation across the whole dataset).
*   **Local Interpretability:** Use SHAP force plots for individual predictions to explain anomalous behavior (e.g., "Why was the prediction so low for this specific hour on Campus A?").
*   **Integration:** Add an Explainability dashboard/tab to the frontend app, visualizing SHAP values for recent predictions.
* **Some More**: Feature Stability Analysis for "Are important features changing over seasons?"

## 2. Causal Inference & Hypothesis Testing
**Objective:** Differentiate between correlation and causation to understand the true drivers of solar performance.
*   **A/B Testing (Virtual):** Compare performance between different sites or panels under similar environmental conditions using rigorous statistical testing (e.g., T-tests, ANOVA).
*   **Causal Impact:** If an intervention occurred (e.g., cleaning the panels, changing an inverter), use causal impact analysis to quantify the exact uplift or degradation caused by that event, controlling for weather.
*   **Hypothesis Formulation:** Test assumptions like "Does high humidity significantly degrade performance independently of temperature?"
*   **What we can call it:** Quasi-Experimental Analysis or Intervention Impact Analysis this should be done

## 3. Clustering
**Objective:** Discover hidden patterns and group similar operational days or geographical sites.
*   **Site Clustering:** Group different solar installations based on their generation profiles (e.g., using K-Means or DBSCAN) to identify which sites behave similarly.
*   **Temporal Clustering:** Group days into "profiles" (e.g., perfect clear sky days, highly variable cloudy days, partial shade days) to understand the frequency of different operating conditions.
*   **Usage:** Use these clusters as new categorical features for predictive modeling to improve accuracy.
*   **Some tips:** Use clusters as features. Feed into the forecasting model

## 4. Error Analysis
**Objective:** Systematically identify where the current predictive models are failing to guide targeted improvements.
*   **Residual Analysis:** Plot residuals (actual vs. predicted) against various features (e.g., temperature, time of day) to look for heteroscedasticity or non-linear patterns the model missed.
*   **Worst-Case Isolation:** Filter for the top 5% of highest-error predictions and perform root-cause analysis. Are they all happening at a specific time? Under specific weather conditions?
*   **Error Dashboards:** Create tracking metrics specifically for model degradation over time, broken down by campus or season.
*   **Make A Proper Documentation as well**

## 5. Segmentation
**Objective:** Break down high-level metrics into granular, actionable segments.
*   **Performance by Segment:** Analyze generation efficiency sliced by categorical variables: e.g., by Panel Type, by Inverter Age, by specific Campus or Sub-region.
*   **Time-based Segmentation:** Contrast performance during peak vs. off-peak hours, or weekday vs. weekend patterns (to correlate with campus energy demand).
*   **Actionable Insights:** Allow end-users to drill down into the dashboard by selecting specific segments to view localized performance metrics.


## 6. Multivariate Policy Analysis
**Objective:** Simulate and evaluate potential operational or policy decisions based on multivariate data.
*   **What-If Scenarios:** Build a simulation engine where users can tweak variables (e.g., "What if we upgrade 20% of panels to a higher efficiency rating? What if average summer temperatures rise by 2°C?").
*   **ROI Optimization:** Combine predicted generation with energy pricing models to recommend the optimal times for maintenance or battery storage discharge.
*   **Policy Recommendations:** Use the insights gathered from causal testing and segmentation to draft data-driven policy recommendations for future solar investments on campus.

## 7. Some additional stuff
* **Forecasting Uncertainty**: error margin, can use quantile regression, prediction intervals, or conformal prediction
* **Drift Detection**: Monitor Weather drift, data drift, concept drift, to see weather distribution changed or generation behavior changed
