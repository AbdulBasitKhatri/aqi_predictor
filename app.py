import os
import joblib
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import hopsworks
import math

# ---------------------------------------------------------
# 1. INITIAL SETUP & HOPSWORKS CONNECTION
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Karachi AQI Predictor Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# Fetch API key from .env file
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    st.error("❌ `HOPSWORKS_API_KEY` not found in your environment (.env file).")
    st.stop()

@st.cache_resource(show_spinner="Connecting to Hopsworks Model Registry...")
def download_latest_model_assets(api_key):
    try:
        # Authenticate and login to Hopsworks
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
        return latest_model_meta.version, download_dir
        
    except Exception as e:
        st.error(f"Error connecting to Hopsworks or pulling assets: {e}")
        return None, None

# Run the cached function to pull files down
model_version, local_asset_dir = download_latest_model_assets(HOPSWORKS_API_KEY)

if not local_asset_dir:
    st.stop()

# Helper macro to point to downloaded assets safely
def get_asset_path(filename):
    return os.path.join(local_asset_dir, filename)


# ---------------------------------------------------------
# 2. APP HEADER & LAYOUT
# ---------------------------------------------------------
st.title("🌬️ Karachi AQI Inference & Analysis Dashboard")
st.caption(f"Connected to Hopsworks Registry | **Model Name:** karachi_aqi_predictor | **Active Version:** v{model_version}")
st.write("---")


# ---------------------------------------------------------
# 3. METRICS SECTION (From metrics.csv)
# ---------------------------------------------------------
st.header("📊 Model Performance Metrics")
metrics_path = get_asset_path("metrics.csv")

if os.path.exists(metrics_path):
    df_metrics = pd.read_csv(metrics_path)
    
    # Render metrics cleanly in columns dynamically
    cols = st.columns(min(len(df_metrics.columns), 4))
    for i, col_name in enumerate(df_metrics.columns):
        val = df_metrics[col_name].iloc[0]
        # Format if float, else keep as string
        formatted_val = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
        with cols[i % len(cols)]:
            st.metric(label=col_name.upper().replace('_', ' '), value=formatted_val)
else:
    st.warning("⚠️ `metrics.csv` was not found in the downloaded model registry folder.")

st.write("---")


# ---------------------------------------------------------
# 4. 3-DAY INFERENCE SECTION (From best_aqi_model.pkl)
# ---------------------------------------------------------
st.header("🔮 3-Day AQI Forecast Inference")
model_path = get_asset_path("best_aqi_model.pkl")

# Connect to Hopsworks and load feature group
@st.cache_data(show_spinner="Fetching latest live observations from Feature Store...")
def fetch_feature_group_data(api_key):
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_fg", version=1)
    return fg.read()

df = fetch_feature_group_data(HOPSWORKS_API_KEY)

# --- HOPSWORKS CASE-INSENSITIVE SAFETY FIX ---
available_cols = {col.lower(): col for col in df.columns}

aqi_source_col = available_cols.get("aqi", "aqi")
time_source_col = available_cols.get("timestamp", "timestamp")

df[time_source_col] = pd.to_datetime(df[time_source_col])
df = df.sort_values(time_source_col).reset_index(drop=True)

# Build features safely onto baseline arrays
df["aqi_lag_1"] = df[aqi_source_col].shift(1)
df["aqi_lag_3"] = df[aqi_source_col].shift(3)
df["aqi_lag_6"] = df[aqi_source_col].shift(6)
df["aqi_rolling_mean"] = df[aqi_source_col].rolling(6).mean()
df["aqi_rolling_std"] = df[aqi_source_col].rolling(6).std()

def get_aqi_status(aqi_val):
    """Categorize OpenWeather standard 1-5 AQI brackets"""
    # Round to the nearest whole index since OpenWeather scales discretely from 1 to 5
    aqi_round = int(round(aqi_val))
    
    if aqi_round <= 1:
        return "Good", "🟢"
    elif aqi_round == 2:
        return "Fair", "🟡"
    elif aqi_round == 3:
        return "Moderate", "🟠"
    elif aqi_round == 4:
        return "Poor", "🔴"
    else:
        return "Very Poor", "🟣"


if os.path.exists(model_path):
    try:
        # 1. Load the model pipeline
        model = joblib.load(model_path)
        
        # 2. Extract EXACT feature names your model was trained on
        if hasattr(model.named_steps["scaler"], "feature_names_in_"):
            feature_names = list(model.named_steps["scaler"].feature_names_in_)
        else:
            # Complete fallback matching your explicit time structure 
            feature_names = ["aqi", "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_rolling_mean", "aqi_rolling_std"]

        # Ensure we drop lookback rows containing NaNs
        df_clean = df.dropna(subset=["aqi_lag_6", "aqi_rolling_std"]).copy()
        if df_clean.empty:
            df_clean = df.copy()
            
        # Get the single latest real row (retains structural weather/pollutants info)
        latest_row = df_clean.iloc[-1].copy()
        latest_aqi_values = df_clean.tail(6)[aqi_source_col].astype(float).tolist()

        # 4. Generate prediction arrays recursively (72 hours ahead)
        predictions_pool = []
        history = latest_aqi_values.copy()
        
        # Create a mutable dictionary based on the latest real data point
        current_features = latest_row.to_dict()

        for step in range(72):
            # Dynamic rolling time updates
            current_features["aqi"] = history[-1]
            current_features["aqi_lag_1"] = history[-2]
            current_features["aqi_lag_3"] = history[-4]
            current_features["aqi_lag_6"] = history[-6]
            current_features["aqi_rolling_mean"] = np.mean(history[-6:])
            current_features["aqi_rolling_std"] = np.std(history[-6:]) if np.std(history[-6:]) > 0 else 0.0
            
            # Map calendar attributes smoothly out into the future
            future_time = pd.to_datetime(latest_row[time_source_col]) + datetime.timedelta(hours=step+1)
            
            # Handle variable names dynamically if present in your model training
            if "hour" in current_features: current_features["hour"] = future_time.hour
            if "day" in current_features: current_features["day"] = future_time.day
            if "month" in current_features: current_features["month"] = future_time.month
            if "day_of_week" in current_features: current_features["day_of_week"] = future_time.weekday()
            if "dayofweek" in current_features: current_features["dayofweek"] = future_time.weekday()

            # Align inputs exactly to match feature_names format
            df_inference_row = pd.DataFrame([current_features], columns=feature_names)
            
            # Predict the next hour AQI
            next_hour_pred = float(model.predict(df_inference_row)[0])
            predictions_pool.append(next_hour_pred)
            
            # Append output back into tracking window state
            history.append(next_hour_pred)

        # 5. Extract daily target forecasts from your hourly output arrays
        tomorrow_aqi = np.max(predictions_pool[0:24])
        day_after_aqi = np.max(predictions_pool[24:48])
        three_days_aqi = np.max(predictions_pool[48:72])
        
        predictions = [tomorrow_aqi, day_after_aqi, three_days_aqi]
        
        today = datetime.date.today()
        forecast_dates = [today + datetime.timedelta(days=i) for i in range(1, 4)]

        # Display forecast cards side-by-side
        pred_cols = st.columns(3)
        for index, target_date in enumerate(forecast_dates):
            aqi_val = max(1, min(5, math.floor(predictions[index])))  
            status, emoji = get_aqi_status(aqi_val)
            day_label = target_date.strftime("%B %d, %Y")

            with pred_cols[index]:
                st.markdown(
                    f"""
                    <div style="padding:20px; border-radius:10px; border:1px solid #4A4A4A; background-color:#1E1E1E; text-align:center;">
                        <h4>📅 {day_label}</h4>
                        <h1 style="margin:10px 0;">{emoji} {aqi_val}</h1>
                        <p style="font-weight:bold; color:#A0A0A0;">Status: {status}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Format final view to display features cleanly 
        df_display_vector = pd.DataFrame([current_features], columns=feature_names)

        st.write(" ")

    except Exception as e:
        st.error(f"❌ Failed to execute model inference step: {e}")
else:
    st.error(
        "❌ `best_aqi_model.pkl` missing. Could not load model object for real-time inference."
    )

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