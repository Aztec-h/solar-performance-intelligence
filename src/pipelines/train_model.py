from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config.settings import PROCESSED_DIR, PROJECT_ROOT


TARGET = "SolarGeneration"
TRAIN_END = "2022-01-01"
RANDOM_STATE = 42

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
FINAL_MODEL_PATH = MODEL_DIR / "final_solar_generation_model.joblib"
PREDICTIONS_PATH = PROCESSED_DIR / "model_holdout_predictions.parquet"


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }


def add_model_features(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Quarter"] = df["Timestamp"].dt.quarter

    df["is_daylight"] = df["Hour"].between(7, 19).astype("int8")
    df["WindSpeed_missing"] = df["WindSpeed"].isna().astype("int8")
    df["AirTemperature_missing"] = df["AirTemperature"].isna().astype("int8")
    df["RelativeHumidity_missing"] = df["RelativeHumidity"].isna().astype("int8")
    df["kWp_missing"] = df["kWp"].isna().astype("int8")

    df = df.sort_values(["CampusKey", "SiteKey", "Timestamp"])
    grouped = df.groupby(["CampusKey", "SiteKey"], sort=False)

    periods_1h = 4
    df["Ghi_lag_1h"] = grouped["Ghi"].shift(periods_1h)
    df["CloudOpacity_lag_1h"] = grouped["CloudOpacity"].shift(periods_1h)
    df["Ghi_rolling_1h_mean"] = (
        grouped["Ghi"]
        .transform(lambda s: s.shift(1).rolling(periods_1h, min_periods=1).mean())
    )
    df["CloudOpacity_rolling_1h_mean"] = (
        grouped["CloudOpacity"]
        .transform(lambda s: s.shift(1).rolling(periods_1h, min_periods=1).mean())
    )

    return df


def make_base_frame(master_df):
    model_df = add_model_features(master_df)
    return model_df.dropna(
        subset=[
            TARGET,
            "Ghi",
            "CloudOpacity",
        ]
    )


def split_time_holdout(model_df):
    train_df = model_df[model_df["Timestamp"] < TRAIN_END].copy()
    test_df = model_df[model_df["Timestamp"] >= TRAIN_END].copy()
    return train_df, test_df


def make_ridge_pipeline(numeric_features, categorical_features):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_hgb_pipeline(numeric_features, categorical_features):
    all_features = numeric_features + categorical_features
    categorical_mask = [
        feature in categorical_features
        for feature in all_features
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    categorical_features=categorical_mask,
                    learning_rate=0.08,
                    max_iter=120,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_experiment(name, model, train_df, test_df, features):
    model.fit(train_df[features], train_df[TARGET])
    predictions = model.predict(test_df[features])
    predictions = np.clip(predictions, 0, None)
    metrics = regression_metrics(test_df[TARGET], predictions)
    metrics["experiment"] = name
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_df)
    return metrics, predictions, model


def run_experiments(model_df):
    train_df, test_df = split_time_holdout(model_df)

    categorical_features = [
        "CampusKey",
        "SiteKey",
    ]

    baseline_numeric = [
        "Ghi",
        "CloudOpacity",
        "AirTemperature",
        "RelativeHumidity",
        "Hour",
        "Month",
        "DayOfYear",
        "DayOfWeek",
        "kWp",
        "has_capacity_data",
    ]

    engineered_numeric = baseline_numeric + [
        "Quarter",
        "is_daylight",
        "AirTemperature_missing",
        "RelativeHumidity_missing",
        "kWp_missing",
        "Ghi_lag_1h",
        "CloudOpacity_lag_1h",
        "Ghi_rolling_1h_mean",
        "CloudOpacity_rolling_1h_mean",
    ]

    wind_numeric = engineered_numeric + [
        "WindSpeed",
        "WindSpeed_missing",
    ]

    experiments = [
        (
            "dummy_mean",
            DummyRegressor(strategy="mean"),
            baseline_numeric + categorical_features,
        ),
        (
            "ridge_baseline_no_wind",
            make_ridge_pipeline(baseline_numeric, categorical_features),
            baseline_numeric + categorical_features,
        ),
        (
            "hgb_baseline_no_wind",
            make_hgb_pipeline(baseline_numeric, categorical_features),
            baseline_numeric + categorical_features,
        ),
        (
            "hgb_engineered_no_wind",
            make_hgb_pipeline(engineered_numeric, categorical_features),
            engineered_numeric + categorical_features,
        ),
        (
            "hgb_engineered_with_wind_missing",
            make_hgb_pipeline(wind_numeric, categorical_features),
            wind_numeric + categorical_features,
        ),
    ]

    metrics_records = []
    fitted_models = {}
    prediction_store = {}

    for name, model, features in experiments:
        print(f"Running {name}...")
        metrics, predictions, fitted_model = evaluate_experiment(
            name=name,
            model=model,
            train_df=train_df,
            test_df=test_df,
            features=features,
        )
        metrics_records.append(metrics)
        fitted_models[name] = {
            "model": fitted_model,
            "features": features,
        }
        prediction_store[name] = predictions
        print(metrics)

    metrics_df = (
        pd.DataFrame(metrics_records)
        .set_index("experiment")
        .sort_values("RMSE")
    )

    best_name = metrics_df.index[0]
    best_bundle = fitted_models[best_name]
    best_predictions = prediction_store[best_name]

    holdout_predictions = test_df[
        [
            "CampusKey",
            "SiteKey",
            "Timestamp",
            TARGET,
            "Ghi",
            "CloudOpacity",
            "AirTemperature",
            "RelativeHumidity",
            "WindSpeed",
            "WindSpeed_missing",
            "kWp",
            "has_capacity_data",
        ]
    ].copy()
    holdout_predictions["Prediction"] = best_predictions
    holdout_predictions["Error"] = holdout_predictions[TARGET] - holdout_predictions["Prediction"]
    holdout_predictions["AbsoluteError"] = holdout_predictions["Error"].abs()
    holdout_predictions["SelectedExperiment"] = best_name

    final_artifact = {
        "model": best_bundle["model"],
        "features": best_bundle["features"],
        "target": TARGET,
        "train_end": TRAIN_END,
        "selected_experiment": best_name,
        "metrics": metrics_df,
    }

    return metrics_df, holdout_predictions, final_artifact


def create_diagnostics(master_df, model_df, holdout_predictions):
    coverage = (
        master_df
        .groupby(master_df["Timestamp"].dt.year)[
            [
                TARGET,
                "Ghi",
                "CloudOpacity",
                "AirTemperature",
                "RelativeHumidity",
                "WindSpeed",
                "kWp",
            ]
        ]
        .apply(lambda group: group.notna().mean().round(3))
    )

    rows_by_year = master_df.groupby(master_df["Timestamp"].dt.year).size()
    model_rows_by_year = model_df.groupby(model_df["Timestamp"].dt.year).size()

    campus_error = (
        holdout_predictions
        .groupby("CampusKey")
        .agg(
            rows=(TARGET, "size"),
            mae=("AbsoluteError", "mean"),
            bias=("Error", "mean"),
            mean_actual=(TARGET, "mean"),
            mean_prediction=("Prediction", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    hour_error = (
        holdout_predictions
        .groupby(holdout_predictions["Timestamp"].dt.hour)
        .agg(
            rows=(TARGET, "size"),
            mae=("AbsoluteError", "mean"),
            bias=("Error", "mean"),
            mean_actual=(TARGET, "mean"),
            mean_prediction=("Prediction", "mean"),
        )
        .sort_index()
    )

    return {
        "coverage": coverage,
        "rows_by_year": rows_by_year,
        "model_rows_by_year": model_rows_by_year,
        "campus_error": campus_error,
        "hour_error": hour_error,
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    master_path = PROCESSED_DIR / "master_dataset.parquet"
    master_df = pd.read_parquet(master_path)
    master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"])

    print("Loaded master dataset:", master_df.shape)
    print("Date range:", master_df["Timestamp"].min(), "to", master_df["Timestamp"].max())

    model_df = make_base_frame(master_df)
    metrics_df, holdout_predictions, final_artifact = run_experiments(model_df)

    metrics_df.to_csv(METRICS_PATH)
    holdout_predictions.to_parquet(PREDICTIONS_PATH, index=False)
    joblib.dump(final_artifact, FINAL_MODEL_PATH)

    diagnostics = create_diagnostics(master_df, model_df, holdout_predictions)
    for name, diagnostic in diagnostics.items():
        diagnostic_path = REPORT_DIR / f"model_{name}.csv"
        diagnostic.to_csv(diagnostic_path)

    print("\nFinal experiment ranking:")
    print(metrics_df)
    print("\nSelected final model:", final_artifact["selected_experiment"])
    print("Saved metrics to:", METRICS_PATH)
    print("Saved predictions to:", PREDICTIONS_PATH)
    print("Saved final model to:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
