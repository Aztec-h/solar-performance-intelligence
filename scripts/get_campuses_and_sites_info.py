import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR=Path(__file__).resolve().parent
ARCHIVE_DIR=SCRIPT_DIR.parent / "data" / "archive"
DATA_DIR=SCRIPT_DIR.parent / "data" / "unisolar"
OUTPUT_FILE=SCRIPT_DIR / "campus_and_site_info.txt"

def get_unique_campuses(df: pd.DataFrame):
    if "Campus" in df.columns:
        return df["Campus"].unique()
    else:
        return None

def get_unique_campus_keys(df: pd.DataFrame):
    if "CampusKey" in df.columns:
        return df["CampusKey"].unique()
    else:
        return None

def get_unique_site_keys(df: pd.DataFrame):
    if "SiteKey" in df.columns:
        return df["SiteKey"].unique()
    else:
        return None
    
if __name__=="__main__":
    if not DATA_DIR.exists():
        print("Error: the 'data/unisolar' dir does not exists")
    else:
        print("starting the csv analysis for summary")
        csv_files=list(DATA_DIR.glob("*.csv"))
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('======| Cool campus summary over all csvs gng | =====\n\n')
            for file in csv_files:
                df = pd.read_csv(file)
                
                f.write(f"File: {file.relative_to(DATA_DIR)}\n")
                f.write(f"Unique Campuses: {get_unique_campuses(df)}\n")
                f.write(f"Unique CampusKeys: {get_unique_campus_keys(df)}\n")
                f.write(f"Unique SiteKeys: {get_unique_site_keys(df)}\n\n")
            f.write("=" * 40)
        print("poof done !!!")