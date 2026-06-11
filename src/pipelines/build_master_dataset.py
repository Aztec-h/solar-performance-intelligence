import pandas as pd
from src.config.settings import RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from src.data.loader import DataLoader
from src.utils.profiler import profile_dataset
from src.data.transformer import DataTransformer
from src.data.cleaner import DataCleaner
from src.data.merger import DataMerger

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
    
def test(master_df, irradiance):
    print("=" * 60)
    print("TARGET QUALITY ANALYSIS")
    print("=" * 60)

    target_nulls = (
        master_df
        .groupby("CampusKey")["SolarGeneration"]
        .apply(
            lambda s: s.isna().mean() * 100
        )
        .sort_values(ascending=False)
    )

    print("\nSolarGeneration Missing % per Campus:")
    print(target_nulls)

    print("\n")


    print("=" * 60)
    print("GHI COVERAGE ANALYSIS")
    print("=" * 60)

    ghi_stats = (
        master_df
        .groupby("CampusKey")
        .agg(
            total_rows=("Ghi", "size"),
            missing_ghi=("Ghi", lambda s: s.isna().sum())
        )
    )

    ghi_stats["missing_pct"] = (
        ghi_stats["missing_ghi"]
        / ghi_stats["total_rows"]
        * 100
    )

    print(ghi_stats)

    print("\n")


    print("=" * 60)
    print("TIMESTAMP COVERAGE")
    print("=" * 60)

    coverage = (
        master_df
        .groupby("CampusKey")
        .agg(
            start=("Timestamp", "min"),
            end=("Timestamp", "max")
        )
    )

    print(coverage)

    print("\n")

    print("=" * 60)
    print("CAMPUS 5 DEEP DIVE")
    print("=" * 60)

    campus5 = (
        master_df[
            master_df["CampusKey"] == 5
        ]
    )

    print(
        "\nRows:",
        len(campus5)
    )

    print(
        "\nSolarGeneration Missing %:",
        campus5["SolarGeneration"]
        .isna()
        .mean() * 100
    )

    print(
        "\nGHI Missing %:",
        campus5["Ghi"]
        .isna()
        .mean() * 100
    )

    print(
        "\nWeather Missing %:"
    )

    print(
        campus5[
            [
                "AirTemperature",
                "RelativeHumidity",
                "WindSpeed"
            ]
        ]
        .isna()
        .mean()
        * 100
    )

    irr5 = (
        irradiance[
            irradiance["CampusKey"] == 5
        ]
    )

    print("=" * 60)
    print("IRRADIANCE SOURCE CHECK")
    print("=" * 60)

    print(
        "Rows:",
        len(irr5)
    )

    print(
        "Start:",
        irr5["Timestamp"].min()
    )

    print(
        "End:",
        irr5["Timestamp"].max()
    )

    print(
        "\nMissing GHI:"
    )

    print(
        irr5["Ghi"]
        .isna()
        .mean()
        * 100
    )

    for campus in range(1, 6):
        temp = irradiance[
            irradiance["CampusKey"] == campus
        ]

        print("\n")
        print("=" * 50)
        print(f"Campus {campus}")
        print("=" * 50)

        print(
            temp["Timestamp"]
            .min()
        )

        print(
            temp["Timestamp"]
            .max()
        )

        print(
            temp["Timestamp"]
            .diff()
            .value_counts()
            .head()
        )

    bad = master_df[
        (master_df["CampusKey"] == 5)
        &
        (master_df["Ghi"].isna())
    ]

    print(
        bad[
            [
                "CampusKey",
                "Timestamp"
            ]
        ].head(20)
    )
    
    ts = bad.iloc[0]["Timestamp"]

    irradiance[
        (irradiance["CampusKey"] == 5)
        &
        (
            irradiance["Timestamp"]
            .between(
                ts - pd.Timedelta("1H"),
                ts + pd.Timedelta("1H")
            )
        )
    ]

def main():
    loader = DataLoader(RAW_DIR)
    datasets = loader.load_all()
    
    generation = (
        DataTransformer
        .transform_generation(
            datasets["generation"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
        .pipe(DataCleaner.clean_geneartion)
    )
    
    weather = (
        DataTransformer
        .transform_weather(
            datasets["weather"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
        .pipe(DataCleaner.clean_weather)
    )

    irradiance = (
        DataTransformer
        .transform_irradiance(
            datasets["irradiance"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
        .pipe(DataCleaner.clean_irradiance)
    )

    sites = (
        DataTransformer
        .transform_sites(
            datasets["sites"]
        )
        # .pipe(DataTransformer.optimize_dataframe)
        .pipe(DataCleaner.clean_sites)
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
        output_path = (
            INTERIM_DIR / f"{name}_cleaned.parquet"
        )
        
        df.to_parquet(
            output_path,
            index=False
        )
        print(f"Saved to -> {output_path}")

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
    
    master_df = (
        DataMerger
        .build_master_dataset(
            generation=transformed_datasets["generation"],
            weather=transformed_datasets["weather"],
            irradiance=transformed_datasets["irradiance"],
            sites=transformed_datasets["sites"]
        )
    )
    
    profile_dataset(
        name="master",
        df=master_df,
        timestamp_col="Timestamp",
        unique_cols=[
            "CampusKey",
            "SiteKey"
        ]
    )
    
    test(master_df=master_df, irradiance=transformed_datasets["irradiance"])
        
    master_df.to_parquet(
        PROCESSED_DIR / "master_dataset.parquet",
        index=False
    )
        
if __name__=="__main__":
    main()