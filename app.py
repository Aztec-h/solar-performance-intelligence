import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import plotly.express as px

# Configuration
st.set_page_config(page_title="Solar Performance Intelligence", layout="wide", page_icon="☀️")

# Paths
ROOT_DIR = Path(__file__).parent
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

@st.cache_data
def load_report():
    with open(REPORTS_DIR / "REPORT.md", "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def load_metrics():
    return pd.read_csv(REPORTS_DIR / "model_metrics.csv")

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

tab1, tab2, tab3 = st.tabs(["📝 Final Report", "📊 Model Metrics & Predictions", "🧪 Interactive Predictor"])

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
