import pandas as pd

class DataMerger:
    
    @staticmethod
    def resample_irradiance(
        irradiance: pd.DataFrame
    ) -> pd.DataFrame:
        
        irradiance = irradiance.copy()
        
        numeric_cols = irradiance.select_dtypes(include=['number']).columns.tolist()
        
        if "CampusKey" in numeric_cols:
            numeric_cols.remove("CampusKey")
            
        irradiance = (
            irradiance
            .groupby(
                [
                    "CampusKey",
                    "Timestamp"
                ],
                as_index=False
            )[numeric_cols]
            .mean()
        )
        
        irradiance = (
            irradiance
            .sort_values(["CampusKey", "Timestamp"])
            .set_index("Timestamp")
            .groupby("CampusKey")[numeric_cols]
            .apply(lambda group: group.resample("15min").interpolate(method="linear"))
            .reset_index()
        )
        
        return irradiance

    @staticmethod
    def merge_weather_irradiance(
        weather: pd.DataFrame,
        irradiance: pd.DataFrame
    ) -> pd.DataFrame:
        
        return weather.merge(
            irradiance,
            on=[
                "CampusKey",
                "Timestamp"
            ],
            how="left"
        )
    
    @staticmethod
    def merge_generation(
        weather_irradiance: pd.DataFrame,
        generation: pd.DataFrame
    ) -> pd.DataFrame:
        
        return generation.merge(
            weather_irradiance,
            on=[
                "CampusKey",
                "Timestamp"
            ],
            how="left"
        )
    
    @staticmethod
    def merge_sites(
        master: pd.DataFrame,
        sites: pd.DataFrame
    ) -> pd.DataFrame:
        
        return master.merge(
            sites,
            on=[
                "CampusKey",
                "SiteKey"
            ],
            how="left"
        )
        
    @classmethod
    def build_master_dataset(
        cls,
        generation,
        weather,
        irradiance,
        sites
    ) -> pd.DataFrame:
        
        irradiance = (
            cls
            .resample_irradiance(irradiance)
        )
        
        weather_irradiance = (
            cls
            .merge_weather_irradiance(weather, irradiance)
        )
        
        master = (
            cls
            .merge_generation(weather_irradiance, generation)
        )
        
        master = (
            cls
            .merge_sites(master, sites)
        )
        
        return master
        
