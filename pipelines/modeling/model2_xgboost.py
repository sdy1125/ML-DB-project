import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from pipelines.modeling.common import (
    PreparedData,
    get_feature_names,
    make_preprocessor,
    regression_metrics,
    save_actual_vs_predicted,
)


def train_model2(prepared: PreparedData, dirs: dict[str, Path]) -> dict[str, Any]:
    logging.info("Training Model 2: XGBoost Regression")
    missing_dependency_note = dirs["tables"] / "model2_missing_dependency.txt"
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        missing_dependency_note.write_text(
            "Model 2 requires xgboost. Install it with: pip install xgboost\n",
            encoding="utf-8",
        )
        logging.error("xgboost is not installed. Run: pip install xgboost")
        return {"error": str(exc), "note": missing_dependency_note}
    if missing_dependency_note.exists():
        missing_dependency_note.unlink()

    X_train = prepared.train_df[prepared.feature_cols]
    y_train = prepared.train_df[prepared.target_col]
    X_test = prepared.test_df[prepared.feature_cols]
    y_test = prepared.test_df[prepared.target_col]
    preprocessor, _, _ = make_preprocessor(prepared.train_df, prepared.feature_cols, prepared.country_col)

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    xgb_model.fit(X_train_prepared, y_train)
    predictions = xgb_model.predict(X_test_prepared)
    metrics = regression_metrics(y_test, predictions)

    pred_df = prepared.test_df[[col for col in [prepared.country_col, prepared.year_col] if col]].copy()
    pred_df["actual"] = y_test.to_numpy()
    pred_df["predicted"] = predictions
    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
    pred_path = dirs["predictions"] / "model2_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    feature_names = get_feature_names(preprocessor)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": xgb_model.feature_importances_}
    )
    total_importance = importance_df["importance"].sum()
    importance_df["normalized_importance"] = (
        importance_df["importance"] / total_importance if total_importance > 0 else 0.0
    )
    importance_df = importance_df.sort_values("normalized_importance", ascending=False)
    importance_path = dirs["tables"] / "model2_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False, encoding="utf-8")

    top = importance_df.head(20).sort_values("normalized_importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["normalized_importance"], color="#2f6f9f")
    plt.xlabel("Normalized importance")
    plt.title("Model 2 - XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "model2_feature_importance.png", dpi=160)
    plt.close()
    save_actual_vs_predicted(
        pred_df,
        "actual",
        "predicted",
        "Model 2 - Actual vs Predicted",
        dirs["figures"] / "model2_actual_vs_predicted.png",
    )
    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["predicted"], pred_df["residual"], alpha=0.55)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Model 2 - Residual Plot")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "model2_residual_plot.png", dpi=160)
    plt.close()
    joblib.dump(
        {"preprocess": preprocessor, "model": xgb_model, "feature_cols": prepared.feature_cols},
        dirs["models"] / "model2_xgboost.pkl",
    )
    logging.info("Model 2 metrics: %s", metrics)
    return {
        "metrics": metrics,
        "predictions": pred_path,
        "importance": importance_path,
        "top_importance": importance_df.head(10),
    }
