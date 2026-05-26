import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="karachi_aqi_fg",
    version=1
)

df = fg.read()

print(df.head())
print(df.info())

#Check missing values
print(df.isnull().sum())

#Check distribution of AQI
plt.figure(figsize=(8,5))
sns.histplot(df["aqi"], bins=20, kde=True)
plt.title("AQI Distribution")
plt.show()

#Check correlation between features
plt.figure(figsize=(14,10))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.show()

#Check relationship between pollutants and AQI
pollutants = ["pm25", "pm10", "no2", "so2", "o3", "co"]
for col in pollutants:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df["aqi"])
    plt.title(f"{col} vs AQI")
    plt.show()

#Check if AQI differs on weekends vs weekdays
sns.boxplot(
    x=df["is_weekend"],
    y=df["aqi"]
)
plt.title("Weekend vs Weekday AQI")
plt.show()