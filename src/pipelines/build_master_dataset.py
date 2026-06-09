from src.config.settings import RAW_DIR
from src.data.loader import DataLoader
from src.data.validator import (
    DataProfiler,
    DataValidator
)
from src.data.transformer import (
    DataTransformer
)

DATASET_CONFIG = {
    "generation": {
        "timestamp_col": "Timestamp",
        "unique_cols": [
            "CampusKey",
            "SiteKey"
        ]
    },

    "weather": {
        "timestamp_col": "Timestamp",
        "unique_cols": [
            "CampusKey"
        ]
    },

    "irradiance": {
        "timestamp_col": "Timestamp",
        "unique_cols": [
            "Campus",
            "CampusKey"
        ]
    },

    "sites": {
        "unique_cols": [
            "CampusKey",
            "SiteKey"
        ]
    }
}

def profile_dataset(
    name,
    df,
    timestamp_col=None,
    unique_cols=None
):
    print("\n" + "=" * 60)
    print(f"> {name.upper()}")
    print("=" * 60)
    
    print("Shape: ", DataProfiler.get_shape(df))
    
    print(
        "Memory(Mb): ",
        round(
            DataProfiler
            .get_memory_usage(df),
            2
        ))

    print(
        "Duplicate Rows: ",
        DataProfiler.get_duplicates(df)
    )

    
    if timestamp_col:
        print(
            "\nDate Range:"
        )

        print(
            DataProfiler.get_date_range(
                df,
                timestamp_col,
            )
        )
    
    if unique_cols:

        print(
            "\nUnique Counts:"
        )

        print(
            DataProfiler
            .get_unique_counts(
                df,
                unique_cols,
            )
        )
    
    print(
        "\nNull Report:\n",
        DataProfiler
        .get_null_report(df)
        .head(10)
    )
    

def main():
    loader = DataLoader(RAW_DIR)
    datasets = loader.load_all()
    
    generation = (
        DataTransformer
        .transform_generation(
            datasets["generation"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
    )
    
    weather = (
        DataTransformer
        .transform_weather(
            datasets["weather"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
    )

    irradiance = (
        DataTransformer
        .transform_irradiance(
            datasets["irradiance"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
    )

    sites = (
        DataTransformer
        .transform_sites(
            datasets["sites"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
    )
    
    transformed_datasets = {
        "generation": generation,
        "weather": weather,
        "irradiance": irradiance,
        "sites": sites
    }
        
    #quick validation
    print("\nGeneration")
    print(generation.dtypes)

    print("\nWeather")
    print(weather.dtypes)

    print("\nIrradiance")
    print(irradiance.dtypes)

    print("\nSites")
    print(sites.dtypes)
    
    for name, df in transformed_datasets.items():

        config = DATASET_CONFIG[name]

        profile_dataset(
            name=name,
            df=df,
            timestamp_col=config.get(
                "timestamp_col"
            ),
            unique_cols=config.get(
                "unique_cols"
            )
        )
        
if __name__=="__main__":
    main()