from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

import pandas as pd

generation = pd.read_csv(RAW_DIR / "Solar_Energy_Generation.csv")

generation["Timestamp"] = pd.to_datetime(generation["Timestamp"])

nulls_by_hour = generation[
    generation["SolarGeneration"].isna()
]["Timestamp"].dt.hour.value_counts().sort_index()

print(nulls_by_hour)
