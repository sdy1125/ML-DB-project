import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from pipelines.modeling.common import (
    PreparedData,
    get_feature_names,
    make_preprocessor,
    regression_metrics,
    save_actual_vs_predicted,
)


def is_indicator_feature(feature: str, target_col: str) -> bool:
    return not (
        feature.startswith(f"{target_col}_lag_")
        or feature.startswith(f"{target_col}_rolling_")
        or feature == f"delta_{target_col}"
    )


def add_weight_columns(weight_df: pd.DataFrame) -> pd.DataFrame:
    weight_df = weight_df.copy()
    weight_df["abs_coefficient"] = weight_df["coefficient"].abs()
    total_abs = weight_df["abs_coefficient"].sum()
    weight_df["normalized_weight"] = (
        weight_df["abs_coefficient"] / total_abs if total_abs > 0 else 0.0
    )
    weight_df["direction"] = np.select(
        [weight_df["coefficient"] > 0, weight_df["coefficient"] < 0],
        ["positive", "negative"],
        default="zero",
    )
    return weight_df.sort_values("normalized_weight", ascending=False)


def save_top_weights_figure(weight_df: pd.DataFrame, output_path: Path, title: str) -> None:
    top = weight_df.head(20).sort_values("normalized_weight")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["normalized_weight"], color="#357a38")
    plt.xlabel("Normalized weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_model1(prepared: PreparedData, dirs: dict[str, Path]) -> dict[str, Any]:
    logging.info("Training Model 1: Elastic Net Regression")
    model1_feature_cols = [
        feature
        for feature in prepared.feature_cols
        if is_indicator_feature(feature, prepared.target_col)
    ]
    excluded_target_history = [
        feature for feature in prepared.feature_cols if feature not in model1_feature_cols
    ]
    if not model1_feature_cols:
        raise ValueError("Model 1 requires at least one non-target-history feature.")
    if excluded_target_history:
        logging.info(
            "Model 1 excludes target history features for interpretation: %s",
            excluded_target_history,
        )

    X_train = prepared.train_df[model1_feature_cols]
    y_train = prepared.train_df[prepared.target_col]
    X_test = prepared.test_df[model1_feature_cols]
    y_test = prepared.test_df[prepared.target_col]
    preprocessor, numeric_features, _ = make_preprocessor(
        prepared.train_df,
        model1_feature_cols,
        prepared.country_col,
    )
    cv_splits = min(5, max(2, prepared.train_df[prepared.year_col].nunique() - 1))
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                ElasticNetCV(
                    alphas=np.logspace(-4, 2, 25),
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=TimeSeriesSplit(n_splits=cv_splits),
                    max_iter=100000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    pred_df = prepared.test_df[[col for col in [prepared.country_col, prepared.year_col] if col]].copy()
    pred_df["actual"] = y_test.to_numpy()
    pred_df["predicted"] = predictions
    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
    pred_path = dirs["predictions"] / "model1_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    feature_names = get_feature_names(model.named_steps["preprocess"])
    coefficients = model.named_steps["model"].coef_
    weight_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    weight_df = weight_df[weight_df["feature"].isin(numeric_features)].copy()
    weight_df = add_weight_columns(weight_df)

    weight_path = dirs["tables"] / "model1_hidden_weights.csv"
    weight_df.to_csv(weight_path, index=False, encoding="utf-8")

    save_top_weights_figure(
        weight_df,
        dirs["figures"] / "model1_top_hidden_weights.png",
        "Model 1 - Top Hidden Weights",
    )
    save_actual_vs_predicted(
        pred_df,
        "actual",
        "predicted",
        "Model 1 - Actual vs Predicted",
        dirs["figures"] / "model1_actual_vs_predicted.png",
    )
    joblib.dump(model, dirs["models"] / "model1_elasticnet.pkl")
    logging.info("Model 1 metrics: %s", metrics)
    return {
        "metrics": metrics,
        "predictions": pred_path,
        "weights": weight_path,
        "top_weights": weight_df.head(10),
        "exog_candidates": weight_df["feature"].head(5).tolist(),
        "features_used": model1_feature_cols,
        "excluded_target_history": excluded_target_history,
    }
