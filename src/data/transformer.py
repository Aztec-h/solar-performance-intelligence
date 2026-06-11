import pandas as pd

from src.config.constants import CAMPUS_MAP

LOCAL_TIMEZONE = "Australia/Melbourne"

class DataTransformer:
    
    @staticmethod
    def transform_generation(df):
        
        df = df.copy()
        
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp']
        )
        
        df = df.sort_values(
            ["SiteKey", "Timestamp"]
        )
        
        return df
    
    @staticmethod
    def transform_weather(df):
        
        df = df.copy()
        
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp']
        )
        
        df = df.sort_values(
            ['CampusKey', 'Timestamp']
        )
        
        return df
    
    @staticmethod
    def transform_irradiance(df):
        
        df = df.copy()
        
        df['CampusKey'] = (
            df['Campus']
            .map(CAMPUS_MAP)
        )
        
        df['Timestamp'] = (
            pd.to_datetime(
                df['Timestamp_UTC'],
                utc=True
            )
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )
        
        df = df.drop(
            columns=['Timestamp_UTC']
        )
        
        df = df.sort_values(
            ['CampusKey', 'Timestamp']
        )
        
        return df
    
    @staticmethod
    def transform_sites(df):
        
        df = df.copy()
        
        df = df.sort_values(
            ['CampusKey', 'SiteKey']
        )
        
        return df
    
    
    # HELPER FUNCS :3
    # memory optimizers
    @staticmethod
    def optimize_dataframe(df):
        
        df = df.copy()
        
        for col in df.select_dtypes(
            include=['int64']
        ):
            df[col] = pd.to_numeric(
                df[col],
                downcast='integer'
            )
        
        for col in df.select_dtypes(
            include=['float64']
        ):
            df[col] = pd.to_numeric(
                df[col],
                downcast='float'
            )
            
        return df
