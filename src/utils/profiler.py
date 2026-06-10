from src.data.validator import DataProfiler

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