class DataValidator:
    
    @staticmethod
    def check_columns(
        df,
        required_columns
    ):
        missing = (
            set(required_columns),
            - set(df.columns)
        )
        
        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )