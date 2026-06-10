import pandas as pd


class DataCleaner:

    @staticmethod
    def clean_geneartion(df):
        
        df = df.copy()
        
        df["Hour"] = (
            df["Timestamp"]
            .dt.hour
        )
        
        night_mask = (
            (df["Hour"] < 7) | (df["Hour"] > 19)
        )
        
        df.loc[
            night_mask &
            df["SolarGeneration"].isna(),
            "SolarGeneration"
        ] = 0
        
        return df