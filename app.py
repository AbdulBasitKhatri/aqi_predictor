import os
import joblib
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import hopsworks
import math
from zoneinfo import ZoneInfo

# ---------------------------------------------------------
# 1. INITIAL SETUP & HOPSWORKS CONNECTION
# ---------------------------------------------------------
load_dotenv()

LAT = 24.86
LON = 67.01
df = None
project = None

st.set_page_config(
    page_title="Karachi AQI Predictor Dashboard",
    page_icon="🌬️",
    layout="wide"
)

#fetch API key from .env file
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

if not HOPSWORKS_API_KEY:
    st.error("❌ `HOPSWORKS_API_KEY` not found in your environment (.env file).")
    st.stop()

if not OPENWEATHER_API_KEY:
    st.error("❌ `OPENWEATHER_API_KEY` not found in your environment (.env file).")
    st.stop()

#fetch current weather data from OpenWeather API
@st.cache_data(show_spinner="Fetching current weather data from OpenWeather API...")
def get_weather():
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": LAT,
            "lon": LON,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

@st.cache_data(show_spinner="Fetching current air pollution data...")
def get_air_pollution():
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"

        response = requests.get(
            url,
            params={
                "lat": LAT,
                "lon": LON,
                "appid": OPENWEATHER_API_KEY
            }
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"Error fetching air pollution data: {e}")
        return None

@st.cache_data(show_spinner="Loading...")
def fetch_feature_group_data(api_key):
    global df, project
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_fg", version=1)
    df = fg.read()
    df.to_csv("feature_group_data.csv", index=False)

@st.cache_resource(show_spinner="Connecting to Hopsworks Model Registry...")
def download_latest_model_assets(api_key):
    try:
        # Authenticate and login to Hopsworks
        global project
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        
        # Get all versions of the model, sort to identify the highest/latest version
        models = mr.get_models(name="karachi_aqi_predictor")
        if not models:
            st.error("No models found with the name 'karachi_aqi_predictor'.")
            return None, None
            
        latest_model_meta = max(models, key=lambda m: m.version)
        
        # Download all model files (artifacts) into a local folder
        download_dir = latest_model_meta.download()
        fetch_feature_group_data(HOPSWORKS_API_KEY)
        return latest_model_meta.version, download_dir
        
    except Exception as e:
        st.error(f"Error connecting to Hopsworks or pulling assets: {e}")
        return None, None

#run the cached function to pull files down
model_version, local_asset_dir = download_latest_model_assets(HOPSWORKS_API_KEY)

if not local_asset_dir:
    st.stop()

#helper macro to point to downloaded assets safely
def get_asset_path(filename):
    return os.path.join(local_asset_dir, filename)


# ---------------------------------------------------------
# 2. APP HEADER & LAYOUT
# ---------------------------------------------------------
st.title("🌬️ Karachi AQI Inference & Analysis Dashboard")
st.caption(f"Connected to Hopsworks Registry | **Model Name:** karachi_aqi_predictor | **Active Version:** v{model_version}")
st.write("---")

# ---------------------------------------------------------
# 1 & 2. CURRENT WEATHER & POLLUTION SECTION
# ---------------------------------------------------------
st.header("🌤️ Current Weather & Air Quality", divider="rainbow")

weather_data = get_weather()
pollution_data = get_air_pollution()

if weather_data and pollution_data:
    
    #weather overview card
    with st.container(border=True):
        st.markdown("### **Weather Overview**")
        weather_cols = st.columns(4)
        
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]
        condition = weather_data["weather"][0]["description"].title()
        
        weather_cols[0].metric(label="Temperature", value=f"{temp:.1f} °C", delta="🌡️")
        weather_cols[1].metric(label="Humidity", value=f"{humidity}%", delta="💧")
        weather_cols[2].metric(label="Wind Speed", value=f"{wind_speed} m/s", delta="🌬️")
        weather_cols[3].metric(label="Condition", value=condition, delta="☁️")

    #air quality card
    with st.container(border=True):
        pollution = pollution_data["list"][0]
        aqi = pollution["main"]["aqi"]
        
        aqi_map = {
            1: ("Good", "🟢", "Normal / Safe outdoor conditions"),
            2: ("Fair", "🟡", "Minor Risk / Acceptable air quality"),
            3: ("Moderate", "🟠", "Unhealthy for Sensitive Groups"),
            4: ("Poor", "🔴", "Unhealthy / Wear a mask outdoors"),
            5: ("Very Poor", "🟣", "Hazardous / Avoid outdoor activities")
        }
        
        status, emoji, health_alert = aqi_map.get(aqi, ("Unknown", "⚪", "No Data Available"))
        
        st.markdown(f"### **Air Quality Index:  {emoji} Level {aqi} ({status})**")
        st.caption(f"⚠️ **Health Guidance:** {health_alert}")
        
        st.divider()
        
        components_data = pollution["components"]
        pollutant_cols = st.columns(6)
        
        #displaying individual pollutant metrics cleanly with their units
        pollutants = [
            ("PM₂.₅", "pm2_5"), ("PM₁₀", "pm10"), ("NO₂", "no2"),
            ("SO₂", "so2"), ("O₃", "o3"), ("CO", "co")
        ]
        
        for idx, (label, key) in enumerate(pollutants):
            with pollutant_cols[idx]:
                st.metric(label=label, value=f"{components_data[key]:.1f}")
                st.caption("µg/m³")


# ---------------------------------------------------------
# 3. METRICS SECTION (From metrics.csv)
# ---------------------------------------------------------
st.header("📊 Model Performance Metrics", divider="blue")
metrics_path = get_asset_path("metrics.csv")

if os.path.exists(metrics_path):
    df_metrics = pd.read_csv(metrics_path)
    
    with st.container(border=True):
        st.markdown("### **Validation Performance**")
        cols = st.columns(min(len(df_metrics.columns), 4))
        
        for i, col_name in enumerate(df_metrics.columns):
            val = df_metrics[col_name].iloc[0]
            formatted_val = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
            
            with cols[i % len(cols)]:
                #formats headers beautifully (e.g., "mae" -> "MAE", "rmse" -> "RMSE")
                clean_label = col_name.upper().replace('_', ' ')
                st.metric(label=clean_label, value=formatted_val)
else:
    st.warning("⚠️ `metrics.csv` was not found in the downloaded model registry folder.")


# ---------------------------------------------------------
# 4. 3-DAY INFERENCE SECTION (From best_aqi_model.pkl)
# ---------------------------------------------------------
st.header("🔮 3-Day AQI Forecast Inference", divider="green")
model_path = get_asset_path("best_aqi_model.pkl")

#hopsworks case-insensitive column handling fix
available_cols = {col.lower(): col for col in df.columns}
aqi_source_col = available_cols.get("aqi", "aqi")
time_source_col = available_cols.get("timestamp", "timestamp")

df[time_source_col] = pd.to_datetime(df[time_source_col])
df = df.sort_values(time_source_col).reset_index(drop=True)

#build features safely onto baseline arrays
df["aqi_lag_1"] = df[aqi_source_col].shift(1)
df["aqi_lag_3"] = df[aqi_source_col].shift(3)
df["aqi_lag_6"] = df[aqi_source_col].shift(6)
df["aqi_rolling_mean"] = df[aqi_source_col].rolling(6).mean()
df["aqi_rolling_std"] = df[aqi_source_col].rolling(6).std()

def get_aqi_status(aqi_val):
    aqi_round = int(aqi_val)
    if aqi_round <= 1: return "Good", "🟢"
    elif aqi_round == 2: return "Fair", "🟡"
    elif aqi_round == 3: return "Moderate", "🟠"
    elif aqi_round == 4: return "Poor", "🔴"
    else: return "Very Poor", "🟣"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        
        if hasattr(model.named_steps["scaler"], "feature_names_in_"):
            feature_names = list(model.named_steps["scaler"].feature_names_in_)
        else:
            feature_names = [
                "pm25", "pm10", "no2", "so2", "o3", "co",
                "temp", "humidity", "wind_speed",
                "hour", "day", "month", "day_of_week",
                "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24",
                "aqi_rolling_mean_6h", "aqi_rolling_std_6h", "aqi_rolling_mean_24h"
            ]

        df_features = df.copy()
        df_features["aqi_lag_1"] = df_features[aqi_source_col].shift(1)
        df_features["aqi_lag_3"] = df_features[aqi_source_col].shift(3)
        df_features["aqi_lag_6"] = df_features[aqi_source_col].shift(6)
        df_features["aqi_lag_24"] = df_features[aqi_source_col].shift(24)
        
        df_features["aqi_rolling_mean_6h"] = df_features[aqi_source_col].rolling(6).mean()
        df_features["aqi_rolling_std_6h"] = df_features[aqi_source_col].rolling(6).std()
        df_features["aqi_rolling_mean_24h"] = df_features[aqi_source_col].rolling(24).mean()

        df_clean = df_features.dropna(subset=["aqi_lag_24", "aqi_rolling_mean_24h"]).copy()
        if df_clean.empty:
            df_clean = df_features.copy()
            
        latest_row = df_clean.iloc[-1].copy()
        latest_aqi_values = df_clean.tail(24)[aqi_source_col].astype(float).tolist()

        #recursive 72-hour forecast loop
        predictions_pool = []
        history = latest_aqi_values.copy() 
        current_features = latest_row.to_dict()

        for step in range(72):
            current_features["aqi_lag_1"] = history[-1]
            current_features["aqi_lag_3"] = history[-3]
            current_features["aqi_lag_6"] = history[-6]
            current_features["aqi_lag_24"] = history[-24]
            
            current_features["aqi_rolling_mean_6h"] = np.mean(history[-6:])
            current_features["aqi_rolling_std_6h"] = np.std(history[-6:]) if np.std(history[-6:]) > 0 else 0.0
            current_features["aqi_rolling_mean_24h"] = np.mean(history[-24:])
            
            base_date = pd.to_datetime(latest_row[time_source_col]) if time_source_col in latest_row else pd.to_datetime(latest_row.name)
            future_time = base_date + datetime.timedelta(hours=step+1)
            time_factor = math.sin(2 * np.pi * future_time.hour / 24.0)
            current_features["temp"] = latest_row["temp"] + (time_factor * 3.0)
            current_features["humidity"] = max(10, min(100, latest_row["humidity"] - (time_factor * 10)))
            
            decay_rate = 0.95 ** step
            for pollutant in ["pm25", "pm10", "no2", "so2", "o3", "co"]:
                if pollutant in current_features:
                    current_features[pollutant] = latest_row[pollutant] * decay_rate

            if "hour" in current_features: current_features["hour"] = future_time.hour
            if "day" in current_features: current_features["day"] = future_time.day
            if "month" in current_features: current_features["month"] = future_time.month
            if "day_of_week" in current_features: current_features["day_of_week"] = future_time.weekday()
            if "dayofweek" in current_features: current_features["dayofweek"] = future_time.weekday()

            df_inference_row = pd.DataFrame([current_features], columns=feature_names)
            pred_val = model.predict(df_inference_row)
            next_hour_pred = float(pred_val[0]) if hasattr(pred_val, "__len__") else float(pred_val)
            predictions_pool.append(next_hour_pred)
            history.append(next_hour_pred)

        #aggregate hourly predictions into daily max AQI forecasts for the next 3 days
        tomorrow_aqi = np.max(predictions_pool[0:24])
        day_after_aqi = np.max(predictions_pool[24:48])
        two_days_after_aqi = np.max(predictions_pool[48:72])
        predictions = [tomorrow_aqi, day_after_aqi, two_days_after_aqi]
        
        today_karachi = datetime.now(ZoneInfo("Asia/Karachi"))
        forecast_dates = [today_karachi + timedelta(days=i) for i in range(1, 4)]

        #forecast display cards
        pred_cols = st.columns(3)
        for index, target_date in enumerate(forecast_dates):
            raw_val = predictions[index]
            aqi_val = max(1, min(5, math.trunc(raw_val))) if not np.isnan(raw_val) else 1
            status, emoji = get_aqi_status(aqi_val)
            day_label = target_date.strftime("%A, %b %d") # E.g., "Wednesday, Jun 03"

            with pred_cols[index]:
                #native, light/dark responsive dashboard cards
                with st.container(border=True):
                    st.markdown(f"#### 📅 {day_label}")
                    st.metric(label="Predicted AQI:", value=f"{emoji} Level {aqi_val}")
                    st.markdown(f"**Status:** `{status}`")

    except Exception as e:
        st.error(f"❌ Failed to execute model inference step: {e}")
else:
    st.error("❌ `best_aqi_model.pkl` missing. Could not load model object for real-time inference.")

st.write("---")


# ---------------------------------------------------------
# 5. MODEL INTERPRETABILITY & ANALYSIS (SHAP & LIME)
# ---------------------------------------------------------
st.header("🧠 Model Explainability & Analysis")
tab1, tab2, tab3 = st.tabs(["SHAP Summary", "SHAP Feature Importance", "LIME Instance Explanation"])

with tab1:
    st.subheader("Global Explanations: SHAP Summary")
    shap_summary_path = get_asset_path("shap_summary.png")
    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.info("ℹ️ `shap_summary.png` not available in this model package version.")

with tab2:
    st.subheader("Feature Importance Distribution")
    shap_bar_path = get_asset_path("shap_bar.png")
    feat_importance_csv = get_asset_path("feature_importance.csv")
    
    col_img, col_data = st.columns([2, 1])
    
    with col_img:
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, caption="SHAP Feature Importance (Bar Plot)", use_container_width=True)
        else:
            st.info("ℹ️ `shap_bar.png` not available.")
            
    with col_data:
        if os.path.exists(feat_importance_csv):
            st.markdown("**Feature Weights Dataframe**")
            df_feat = pd.read_csv(feat_importance_csv)
            st.dataframe(df_feat, use_container_width=True)

with tab3:
    st.subheader("Local Explanations: LIME Interface")
    lime_path = get_asset_path("lime_explanation.html")
    
    if os.path.exists(lime_path):
        st.write("The interactive window below breaks down exactly how feature thresholds swung the specific instance prediction:")
        with open(lime_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        styled_html = f"""
        <div style="background-color: #FFFFFF; color: #000000; padding: 15px; border-radius: 8px; min-height: 760px;">
            {html_content}
        </div>
        """
        
        components.html(styled_html, height=800, scrolling=True)
    else:
        st.info("ℹ️ `lime_explanation.html` interactive artifact not found in this version folder.")