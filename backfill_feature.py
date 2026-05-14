import os
from datetime import datetime

from numpy import int32
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

START_DATE = "2026-01-01"
END_DATE = "2026-05-15"

#fetch historical data from OpenWeather API
def get_historical_data():

    start = int(datetime.fromisoformat(START_DATE).timestamp())
    end = int(datetime.fromisoformat(END_DATE).timestamp())

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    response = requests.get(url, params={
        "lat": LAT,
        "lon": LON,
        "start": start,
        "end": end,
        "appid": API_KEY
    })

    return response.json()["list"]

#feature engineering
def create_features(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5

    df["pollutant_sum"] = (
        df["pm25"] +
        df["pm10"] +
        df["no2"] +
        df["so2"] +
        df["o3"] +
        df["co"]
    )

    #historical weather data is not available in the free tier of OpenWeather API, so I set these features to 0 for now
    df["temp"] = 0.0
    df["humidity"] = 0.0
    df["wind_speed"] = 0.0

    df["temp_humidity_index"] = 0.0
    df["wind_temp_ratio"] = 0.0

    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

    df["target_aqi"] = df["aqi"].shift(-1)

    df["humidity"] = df["humidity"].astype(int32)
    df["aqi"] = df["aqi"].astype(int32)
    df["hour"] = df["hour"].astype(int32)
    df["day"] = df["day"].astype(int32)
    df["month"] = df["month"].astype(int32)
    df["day_of_week"] = df["day_of_week"].astype(int32)
    return df


#build dataframe with historical data
def build_dataframe():

    data = get_historical_data()

    rows = []

    for item in data:

        rows.append({
            "city": CITY,
            "timestamp": datetime.fromtimestamp(item["dt"]),

            "aqi": item["main"]["aqi"],

            "pm25": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "no2": item["components"]["no2"],
            "so2": item["components"]["so2"],
            "o3": item["components"]["o3"],
            "co": item["components"]["co"]
        })
    df = pd.DataFrame(rows)


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


#build dataframe with historical data
df = build_dataframe()
#print(df.head())

upload_to_hopsworks(df)
print("Historical AQI uploaded successfully!")