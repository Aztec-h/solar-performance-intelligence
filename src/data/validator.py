import pandas as pd
from dataclasses import dataclass

class DataProfiler:
    
    @staticmethod
    def get_shape(df):
        return df.shape

    @staticmethod
    def get_memory_usage(df):
        return (
            df.memory_usage(deep=True)
            .sum()
            / 1024 ** 2
        )
    
    @staticmethod
    def get_duplicates(df):
        return df.duplicated().sum()
    
    @staticmethod
    def get_null_report(df):
        
        report = pd.DataFrame({
            "missing_count": df.isnull().sum()
        })
        
        report["missing_pct"] = (
            report["missing_count"]
            / len(df)
            * 100
        )
        
        return report.sort_values(
            "missing_pct",
            ascending=False
        )
    
    @staticmethod
    def get_date_range(
        df,
        timestamp_col
    ):
        return {
            "start": df[timestamp_col].min(),
            "end": df[timestamp_col].max()
        }
        
    @staticmethod
    def get_unique_counts(
        df,
        columns
    ):
        
        report = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            report[col] = df[col].nunique()
        
        return report
    
class DataValidator:
    
    @staticmethod
    def validate_columns(
        df,
        required_columns
    ):
        missing = (
            set(required_columns)
            - set(df.columns)
        )
        
        if missing:
            raise ValueError(
                f"missing columns: {missing}"
            )
            
    @staticmethod
    def validate_timestamp_column(
        df,
        column
    ):
        if not pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            raise TypeError(
                f"{column} must be datetime"
            )
            
    @staticmethod
    def validate_null_threshold(
        df,
        threshold=20
    ):
        pct = (
            df.isnull()
            .mean()
            * 100
        )
        bad_cols = pct[
            pct > threshold
        ]
        
        if len(bad_cols):
            raise ValueError(
                f"Columns exceed threshold:\n{bad_cols}"
            )
