import os
from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"
output_file = script_dir / "data_summary.txt"

if not data_dir.exists():
    print("Error: the 'data' folder was not found.")
else:
    csv_files = list(data_dir.rglob("*.csv"))
    
    print("starting the csv analysis for summary")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== SOLAR PERFORMANCE DATA SUMMARY ===\n\n")
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            
            f.write(f"File Path: {file_path.relative_to(data_dir)}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {df.columns.to_list()}\n")
            f.write(df.head(2).to_string() + "\n")
            f.write("-" * 40 + "\n\n")
        
        f.write("Done! Summary successfully written to: {output_file}")
    print("poof we done")