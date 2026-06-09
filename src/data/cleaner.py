import pandas as pd


class DataCleaner:

    @staticmethod
    def analyze_missingness(df):

        report = pd.DataFrame({
            "missing_count": df.isnull().sum()
        })

        report["missing_pct"] = (
            report["missing_count"]
            / len(df)
            * 100
        )

        return report.sort_values(
            by="missing_pct",
            ascending=False
        )