import os
from datetime import datetime

import pandas as pd
import requests
import hopsworks
from dotenv import load_dotenv

load_dotenv()

#configurations for API keys, city, and date range for backfilling
API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

CITY = "Karachi"
LAT = 24.86
LON = 67.01

#fetch current air pollution data from OpenWeather API
def get_air_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"

    response = requests.get(url, params={
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY
    })

    return response.json()

#fetch current weather data from OpenWeather API
def get_weather():
    url = "https://api.openweathermap.org/data/2.5/weather"

    response = requests.get(url, params={
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY,
        "units": "metric"
    })

    return response.json()

#feature engineering
def create_features(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour.astype("int32")
    df["day"] = df["timestamp"].dt.day.astype("int32")
    df["month"] = df["timestamp"].dt.month.astype("int32")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int32")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(bool)

    df["pollutant_sum"] = (
        df["pm25"] +
        df["pm10"] +
        df["no2"] +
        df["so2"] +
        df["o3"] +
        df["co"]
    )

    df["temp_humidity_index"] = (
        df["temp"] * df["humidity"] / 100
    )

    df["wind_temp_ratio"] = (
        df["wind_speed"] / (df["temp"].abs() + 0.1)
    )

    aqi_change_rate = df["aqi"].diff().fillna(0).astype("float64")
    df["aqi_change_rate"] = aqi_change_rate
    df["target_aqi"] = df["aqi"].shift(-1)
    df["humidity"] = df["humidity"].astype("int64")

    #integer columns
    df["aqi"] = df["aqi"].astype("int64")

    #double columns
    double_cols = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co",
        "temp",
        "wind_speed",
        "pollutant_sum",
        "temp_humidity_index",
        "wind_temp_ratio",
        "aqi_change_rate",
        "target_aqi"
    ]

    for col in double_cols:
        df[col] = df[col].astype("float64")

    return df

#build dataframe with current data
def build_dataframe():

    pollution = get_air_pollution()
    weather = get_weather()

    air = pollution["list"][0]

    row = {
        "city": CITY,
        "timestamp": datetime.fromtimestamp(air["dt"]),

        "aqi": air["main"]["aqi"],

        "pm25": air["components"]["pm2_5"],
        "pm10": air["components"]["pm10"],
        "no2": air["components"]["no2"],
        "so2": air["components"]["so2"],
        "o3": air["components"]["o3"],
        "co": air["components"]["co"],

        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "wind_speed": weather["wind"]["speed"]
    }

    df = pd.DataFrame([row])

    return create_features(df)

#upload to hopsworks feature store
def upload_to_hopsworks(df):

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_fg",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True
    )

    fg.insert(df)

if __name__ == "__main__":
    #build dataframe
    df = build_dataframe()
    print(df.head())

    upload_to_hopsworks(df)
    print("Current AQI uploaded successfully!")