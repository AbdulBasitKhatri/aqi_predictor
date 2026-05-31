import os
import joblib
import warnings

import hopsworks
import pandas as pd
import numpy as np

from dotenv import load_dotenv

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#linear models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

#tree models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)

#support vector regression
from sklearn.svm import SVR

#neighbors
from sklearn.neighbors import KNeighborsRegressor

#advanced gradient boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#explainability
import shap
import lime
import lime.lime_tabular

#visualization
import matplotlib.pyplot as plt

load_dotenv()

warnings.filterwarnings("ignore")

#configurations for Hopsworks API key and model directory
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

MODEL_DIR = "aqi_model"

os.makedirs(MODEL_DIR, exist_ok=True)

#connect to Hopsworks and load feature group
print("Connecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

#load feature group data
print("Loading feature group...")

fg = fs.get_feature_group(
    name="karachi_aqi_fg",
    version=1
)

df = fg.read()

print(f"Dataset shape: {df.shape}")

# Preprocessing
print("Performing preprocessing...")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# --- FIX 1: Resample to uniform hourly timeline to fix random hours ---
df.set_index("timestamp", inplace=True)

# Resample to hourly frequency. For duplicate hours, we take the mean.
df_hourly = df.resample("h").mean(numeric_only=True)

# Interpolate the missing random hours linearly so the time sequence is perfectly continuous
df_hourly = df_hourly.interpolate(method="linear")

# --- FIX 2: Create precise hourly calendar features ---
df_hourly["hour"] = df_hourly.index.hour
df_hourly["day"] = df_hourly.index.day
df_hourly["month"] = df_hourly.index.month
df_hourly["day_of_week"] = df_hourly.index.dayofweek

# --- FIX 3: Clean Hourly Lag and Rolling features ---
df_hourly["aqi_lag_1"] = df_hourly["aqi"].shift(1)   # 1 hour ago
df_hourly["aqi_lag_3"] = df_hourly["aqi"].shift(3)   # 3 hours ago
df_hourly["aqi_lag_6"] = df_hourly["aqi"].shift(6)   # 6 hours ago
df_hourly["aqi_lag_24"] = df_hourly["aqi"].shift(24) # 24 hours ago (yesterday at same time)

# Rolling windows adjusted for hourly scales (6-hour window and 24-hour window)
df_hourly["aqi_rolling_mean_6h"] = df_hourly["aqi"].rolling(6).mean()
df_hourly["aqi_rolling_std_6h"] = df_hourly["aqi"].rolling(6).std()
df_hourly["aqi_rolling_mean_24h"] = df_hourly["aqi"].rolling(24).mean()

# --- FIX 4: Single target for the NEXT HOUR ---
df_hourly["target_aqi"] = df_hourly["aqi"].shift(-1)

# Remove rows with missing values caused by lookbacks/target shifts
df_hourly = df_hourly.dropna()

print(f"Dataset after preprocessing: {df_hourly.shape}")

# Features optimized for hourly modeling
FEATURES = [
    "pm25", "pm10", "no2", "so2", "o3", "co",
    "temp", "humidity", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h", "aqi_rolling_mean_24h"
]

TARGET = "target_aqi"

X = df_hourly[FEATURES]
y = df_hourly[TARGET]

# Split data into train and test sets chronologically
split_index = int(len(df_hourly) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

#make a dictionary of models to train
models = {
    "LinearRegression": LinearRegression(),

    "Ridge": Ridge(),

    "Lasso": Lasso(),

    "ElasticNet": ElasticNet(),

    "DecisionTree": DecisionTreeRegressor(
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    ),

    "AdaBoost": AdaBoostRegressor(
        random_state=42
    ),

    "SVR": SVR(),

    "KNeighbors": KNeighborsRegressor(),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
}

#train and evaluate models
results = []

best_model = None
best_model_name = None
best_score = -999999

print("\nStarting training...\n")

for name, model in models.items():

    print(f"Training {name}...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    try:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_model_name = name

    except Exception as e:
        print(f"Error training {name}: {e}")

#sort results by R2 score
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="r2",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)

# Save metrics
results_df.to_csv(
    os.path.join(MODEL_DIR, "metrics.csv"),
    index=False
)

print(f"\nBest Model: {best_model_name}")
print(f"Best R2 Score: {best_score:.4f}")

#save the best model
model_path = os.path.join(
    MODEL_DIR,
    "best_aqi_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved.")

#feature importance for the best model
model_object = best_model.named_steps["model"]

if hasattr(model_object, "feature_importances_"):

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model_object.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            MODEL_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    print("\nTop Features:")
    print(importance_df.head(10))

#SHAP analysis for the best model
print("\nGenerating SHAP analysis...")

try:

    model_for_shap = best_model.named_steps["model"]

    explainer = shap.Explainer(
        model_for_shap,
        X_train
    )

    shap_values = explainer(X_test)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "shap_summary.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "shap_bar.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP analysis completed.")

except Exception as e:
    print(f"SHAP generation failed: {e}")

#LIME analysis for the best model
print("\nGenerating LIME analysis...")

try:

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURES,
        mode="regression"
    )

    lime_exp = lime_explainer.explain_instance(
        X_test.iloc[0].values,
        best_model.predict
    )

    lime_exp.save_to_file(
        os.path.join(
            MODEL_DIR,
            "lime_explanation.html"
        )
    )

    print("LIME analysis completed.")

except Exception as e:
    print(f"LIME generation failed: {e}")


predictions = best_model.predict(X_test)

predictions_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})

predictions_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "predictions.csv"
    ),
    index=False
)

print("\nUploading model to Hopsworks Model Registry...")

try:

    mr = project.get_model_registry()

    model = mr.python.create_model(
        name="karachi_aqi_predictor",
        metrics={
            "r2_score": float(best_score),
            "mae": float(
                results_df.iloc[0]["mae"]
            ),
            "rmse": float(
                results_df.iloc[0]["rmse"]
            )
        },
        description="""
        Karachi AQI forecasting model.

        Features:
        - Pollution data
        - Weather data
        - Time-series lag features
        - Rolling statistics

        Includes:
        - SHAP explainability
        - LIME explainability
        - Metrics
        - Predictions
        """
    )

    model.save(MODEL_DIR)

    print("Model uploaded successfully!")

except Exception as e:
    print(f"Model registry upload failed: {e}")

print("\nTraining pipeline completed successfully.")