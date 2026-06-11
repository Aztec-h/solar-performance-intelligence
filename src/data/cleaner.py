import pandas as pd


class DataCleaner:

    @staticmethod
    def clean_geneartion(df: pd.DataFrame) -> pd.DataFrame:
        
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
        
        return df.drop(
            columns=["Hour"]
        )
        
    
    @staticmethod
    def clean_weather(
        df: pd.DataFrame,
        interpolation_limit: int = 8
    ) -> pd.DataFrame:
        df = df.copy()
        
        weather_cols = [
            "AirTemperature",
            "ApparentTemperature",
            "DewPointTemperature",
            "RelativeHumidity",
        ]
        
        df = (
            df.sort_values(["CampusKey", "Timestamp"])
        )
        
        for col in weather_cols:
            df.groupby("CampusKey")[col].transform(
                lambda s: s.interpolate(
                    limit=interpolation_limit
                )
            )
        
        return df
    
    @staticmethod
    def clean_irradiance(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        
        df = df.copy()
        
        group_cols = [
            "CampusKey",
            "Timestamp"
        ]
        
        value_cols = [
            col for col in df.columns
            if col not in group_cols
        ]
        
        agg_map = {
            col: "mean"
            if pd.api.types.is_numeric_dtype(df[col])
            else "first"
            for col in value_cols
        }
        
        return (
            df
            .sort_values(group_cols)
            .groupby(
                group_cols,
                as_index=False
            )
            .agg(agg_map)
        )

    @staticmethod
    def clean_sites(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        df = df.copy()
        df["has_capacity_data"] = df["kWp"].notna().astype("int8")

        cols_to_drop = [
            "Panel",
            "Inverter",
            "Optimizers",
            "Metric"
        ]
        
        existing_cols = [
            col for col in cols_to_drop if col in df.columns
        ]
        
        return df.drop(
            columns=existing_cols
        )
        
