import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COUNTRY_ALIASES = {
    "country",
    "nation",
    "economy",
    "country_name",
    "countryname",
}
YEAR_ALIASES = {"year", "time"}
TARGET_ALIASES = {
    "score",
    "target",
    "imd_score",
    "sdg_score",
    "overall_score",
    "goal16",
    "goal_16",
}
VIETNAM_ALIASES = {"viet nam", "vietnam", "viet_nam"}
FORECAST_YEARS = list(range(2025, 2031))


@dataclass
class ModelingConfig:
    project_root: Path
    data_dir: Path
    output_dir: Path
    country_col: str | None = None
    year_col: str | None = None
    target_col: str | None = None
    include_country_dummy: bool = False
    max_lagged_features: int = 5
    random_state: int = 42


@dataclass
class PreparedData:
    source_path: Path
    raw_shape: tuple[int, int]
    frame: pd.DataFrame
    country_col: str | None
    year_col: str
    target_col: str
    feature_cols: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def snake_case(name: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "models": output_dir / "models",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "predictions": output_dir / "predictions",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def scan_project_structure(root: Path) -> dict[str, bool]:
    keys = [
        "data",
        "data/clean",
        "notebooks",
        "src",
        "scripts",
        "models",
        "outputs",
        "pipelines",
        "pipelines/modeling",
    ]
    structure = {key: (root / key).exists() for key in keys}
    logging.info("Project structure detected: %s", structure)
    return structure


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    non_zero = y_true.replace(0, np.nan).dropna()
    mape = None
    if len(non_zero) == len(y_true):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def make_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    country_col: str | None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])]
    categorical_features = [
        col
        for col in feature_cols
        if col not in numeric_features and country_col and col == country_col
    ]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def save_actual_vs_predicted(
    predictions: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(predictions[actual_col], predictions[pred_col], alpha=0.55)
    low = min(predictions[actual_col].min(), predictions[pred_col].min())
    high = max(predictions[actual_col].max(), predictions[pred_col].max())
    plt.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def clip_score(values: np.ndarray | pd.Series, lower: float = 0.0, upper: float = 100.0) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower, upper)

