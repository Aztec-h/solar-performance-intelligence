import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR / "unisolar"
OUTPUT_FILE = OUTPUT_DIR / "merged_dataset.csv"

irradiance_key_map = {
    "Bundoora": 1,
    "Wodonga": 2,
    "Bendigo": 3,
    "Mildura": 4,
    "Shepparton": 5
}

irradiance_df = pd.read_csv(
    DATA_DIR / "unisolar" / "Solar_Irradiance.csv"
)

generation_df = pd.read_csv(
    DATA_DIR / "unisolar" / "Solar_Energy_Generation.csv"
)

weather_df = pd.read_csv(
    DATA_DIR / "unisolar" / "Weather_Data_reordered_all.csv"
)

# irradiance_df
irradiance_df["CampusKey"] = (
    irradiance_df["Campus"]
    .map(irradiance_key_map)
)

irradiance_df["Timestamp_UTC"] = pd.to_datetime(
    irradiance_df["Timestamp_UTC"]
)

# generation_df
generation_df["Timestamp"] = pd.to_datetime(
    generation_df["Timestamp"]
)

weather_df["Timestamp"] = pd.to_datetime(
    weather_df["Timestamp"]
)

print(irradiance_df.head())
print(irradiance_df.columns)

print(
    generation_df["Timestamp"]
    .sort_values()
    .diff()
    .value_counts()
    .head()
)

print(
    weather_df["Timestamp"]
    .sort_values()
    .diff()
    .value_counts()
    .head()
)

print(
    irradiance_df["Timestamp_UTC"]
    .sort_values()
    .diff()
    .value_counts()
    .head()
)

print(
    generation_df["Timestamp"].min(),
    generation_df["Timestamp"].max()
)

print(
    weather_df["Timestamp"].min(),
    weather_df["Timestamp"].max()
)

print(
    irradiance_df["Timestamp_UTC"].min(),
    irradiance_df["Timestamp_UTC"].max()
)