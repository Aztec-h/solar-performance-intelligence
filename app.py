import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import plotly.express as px
import json

# Configuration
st.set_page_config(page_title="Solar Performance Intelligence", layout="wide", page_icon="☀️")

# Paths
ROOT_DIR = Path(__file__).parent
REPORTS_DIR = ROOT_DIR / "reports"
V2_REPORTS_DIR = REPORTS_DIR / "V2"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

@st.cache_data
def load_report():
    with open(REPORTS_DIR / "V1" / "REPORT.md", "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def load_v2_json(filename):
    filepath = V2_REPORTS_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_v2_csv(filename):
    filepath = V2_REPORTS_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None

@st.cache_data
def load_metrics():
    return pd.read_csv(REPORTS_DIR / "V1" / "model_metrics.csv")

@st.cache_data
def load_predictions(sample_n=10000):
    df = pd.read_parquet(DATA_DIR / "model_holdout_predictions.parquet")
    # Sample data for performance on frontend
    if len(df) > sample_n:
        return df.sample(sample_n, random_state=42)
    return df

@st.cache_data
def load_campus_error():
    if (REPORTS_DIR / "model_campus_error.csv").exists():
        return pd.read_csv(REPORTS_DIR / "model_campus_error.csv")
    return None

st.title("☀️ Solar Performance Intelligence App")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Final Report", "📊 Model Metrics & Predictions", "🧠 V2 Advanced Analytics", "🧪 Interactive Predictor"])

with tab1:
    st.markdown(load_report())

with tab2:
    st.header("Model Evaluation Dashboard")
    st.write("This dashboard displays the performance of the finalized Machine Learning Pipeline.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Experiment Metrics")
        metrics_df = load_metrics()
        st.dataframe(metrics_df, use_container_width=True)
        
    with col2:
        st.subheader("RMSE Comparison")
        fig = px.bar(metrics_df, x="experiment", y="RMSE", title="Root Mean Squared Error by Model")
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    st.subheader("Holdout Predictions Sample")
    
    try:
        preds_df = load_predictions()
        st.write("Visualizing a random sample of holdout predictions (Actual vs Predicted)")
        
        fig2 = px.scatter(
            preds_df, 
            x="SolarGeneration", 
            y="Prediction", 
            color="CampusKey",
            labels={"SolarGeneration": "Actual Generation (kW)", "Prediction": "Predicted Generation (kW)"},
            title="Actual vs Predicted Solar Generation"
        )
        # Add diagonal line for perfect prediction
        max_val = max(preds_df["SolarGeneration"].max(), preds_df["Prediction"].max())
        fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
        
        st.plotly_chart(fig2, use_container_width=True)
        
        campus_err = load_campus_error()
        if campus_err is not None:
            st.subheader("Error by Campus")
            st.dataframe(campus_err, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load prediction visualizations: {e}")

with tab3:
    st.header("🧠 V2 Advanced Analytics & Intelligence")
    st.write("This section provides deep-dive causal, interpretative, and prescriptive insights from our V2 analytics engine.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Policy Simulation: +2°C Global Warming")
        policy_data = load_v2_json("policy_analysis_what_if.json")
        if policy_data:
            st.metric(
                label="Simulated Mean Generation Impact", 
                value=f"{policy_data['Simulated_Mean_Generation']:.2f} kW", 
                delta=f"{policy_data['Impact_Percentage']:.2f}% vs Base"
            )
            st.info("Simulation holds all other variables constant to isolate temperature effect.")
            
        st.subheader("Causal Inference: High Heat vs High Irradiance")
        causal_data = load_v2_json("causal_inference_impact.json")
        if causal_data:
            st.write(causal_data["hypothesis"])
            if causal_data.get("mean_generation_loss"):
                st.error(f"Mean Generation Loss: {causal_data['mean_generation_loss']:.2f} kW")
                st.write(f"**Conclusion**: {causal_data['conclusion']} (p={causal_data['p_value']:.4f})")
            
    with col2:
        st.subheader("SHAP Explainability & Feature Stability")
        shap_data = load_v2_json("shap_feature_stability.json")
        if shap_data:
            st.success(shap_data["insight"])
            st.write("**Top Global Predictors:**")
            st.write(", ".join(shap_data["top_features_overall"]))
            st.write(f"**Temporal Shifts:** {shap_data['season_shifts']}")

    st.divider()
    
    st.subheader("Granular Error Segmentation & Clustering")
    col3, col4 = st.columns(2)
    
    with col3:
        seg_data = load_v2_csv("segmentation_performance.csv")
        if seg_data is not None:
            st.write("Mean Absolute Error Segmented by Campus and Hour")
            fig3 = px.density_heatmap(seg_data, x="Hour", y="CampusKey", z="MAE", title="MAE Heatmap", color_continuous_scale="Viridis")
            st.plotly_chart(fig3, use_container_width=True)
            
    with col4:
        cluster_data = load_v2_json("temporal_clustering.json")
        if cluster_data:
            st.write("Temporal Weather Profile Clusters (Mean Values)")
            # Convert dict of dicts to dataframe for display
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df.T, use_container_width=True)
            
    st.divider()
    
    st.subheader("Forecasting Uncertainty & Drift Detection")
    col5, col6 = st.columns(2)
    
    with col5:
        uncertainty_data = load_v2_csv("forecasting_uncertainty_intervals.csv")
        if uncertainty_data is not None:
            st.write("90% Prediction Intervals by Hour (Error Bounds)")
            # Plot the uncertainty bounds using plotly
            fig4 = px.line(uncertainty_data, x="Hour", y=["Lower_5%", "Upper_95%"], 
                           title="Empirical Error Margins by Hour of Day",
                           labels={"value": "Error (kW)", "variable": "Percentile"})
            st.plotly_chart(fig4, use_container_width=True)
            
    with col6:
        drift_data = load_v2_json("data_drift_detection.json")
        if drift_data:
            st.write("Year-over-Year Weather Distribution Drift")
            if "Error" not in drift_data:
                st.metric(label=f"Drift KS Statistic ({drift_data['Comparison']})", 
                          value=f"{drift_data['KS_Statistic']:.4f}")
                st.write(f"**P-Value:** {drift_data['P_Value']:.4f}")
                if drift_data["Drift_Detected"]:
                    st.warning("⚠️ Statistically significant drift detected between operating years.")
                else:
                    st.success("✅ No significant distributional drift detected.")
            else:
                st.info(drift_data["Error"])
                
    st.divider()
    st.subheader("Critical Failure Mode Isolation (Worst 5% Predictions)")
    worst_cases = load_v2_csv("error_analysis_worst_5pct.csv")
    if worst_cases is not None:
        st.write("Analyzing the top 5% highest absolute errors across all holdout predictions.")
        fig5 = px.bar(worst_cases, x="CampusKey", y="Count", color="MeanError",
                      title="Frequency of Worst Predictions by Campus",
                      labels={"Count": "Number of Severe Errors", "MeanError": "Average Error in these cases (kW)"})
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.header("Interactive Solar Predictor")
    st.write("Simulate an API request to the backend model to get real-time generation predictions.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            campus_key = st.selectbox("Campus Key", ["C1", "C2", "C3", "C4", "C5"])
            site_key = st.selectbox("Site Key", ["S1", "S2", "S3", "S4", "S5"])
            ghi = st.number_input("Global Horizontal Irradiance (Ghi)", value=500.0)
            cloud_opacity = st.number_input("Cloud Opacity (%)", value=20.0, min_value=0.0, max_value=100.0)
            
        with col2:
            air_temp = st.number_input("Air Temperature (°C)", value=25.0)
            rel_humidity = st.number_input("Relative Humidity (%)", value=50.0, min_value=0.0, max_value=100.0)
            kwp = st.number_input("Installed Capacity (kWp)", value=100.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            month = st.slider("Month", 1, 12, 6)
            
        submit = st.form_submit_button("Predict Generation")
        
    if submit:
        # Load Model
        try:
            model_artifact = joblib.load(MODELS_DIR / "final_solar_generation_model.joblib")
            model = model_artifact["model"]
            features = model_artifact["features"]
            
            # Construct feature dictionary with defaults for engineered features
            is_daylight = 1 if 7 <= hour <= 19 else 0
            
            input_data = {
                "Ghi": [ghi],
                "CloudOpacity": [cloud_opacity],
                "AirTemperature": [air_temp],
                "RelativeHumidity": [rel_humidity],
                "Hour": [hour],
                "Month": [month],
                "DayOfYear": [150], # Default middle of year
                "DayOfWeek": [3], # Default Wednesday
                "kWp": [kwp],
                "has_capacity_data": [1 if kwp > 0 else 0],
                "Quarter": [(month - 1) // 3 + 1],
                "is_daylight": [is_daylight],
                "AirTemperature_missing": [0],
                "RelativeHumidity_missing": [0],
                "kWp_missing": [0],
                "Ghi_lag_1h": [ghi], # Default to current
                "CloudOpacity_lag_1h": [cloud_opacity], # Default to current
                "Ghi_rolling_1h_mean": [ghi], # Default to current
                "CloudOpacity_rolling_1h_mean": [cloud_opacity], # Default to current
                "CampusKey": [campus_key],
                "SiteKey": [site_key]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Ensure all required features exist
            missing_cols = set(features) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
                
            input_df = input_df[features]
            prediction = model.predict(input_df)[0]
            prediction = max(0, prediction) # Clip negative
            
            st.success(f"### Predicted Generation: {prediction:.2f} kW")
            st.info("Note: Lagged and rolling features are approximated to the current values in this manual simulation.")
        except Exception as e:
            st.error(f"Error predicting: {e}")
