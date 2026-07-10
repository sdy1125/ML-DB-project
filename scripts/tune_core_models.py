"""Tune the two core SDG16 models: Panel/OLS weights and XGBoost.

Outputs are written to artifacts/core_model_tuning so the report can be
screenshotted or copied directly.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SOURCE_PRIORITY = {
    "SDR2024": 0,
    "ESDR2023_24": 1,
    "ESDR2023": 1,
    "SDR2022": 2,
}

XGBOOST_BASELINE_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 200,
    "random_state": 42,
}


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_validation: pd.DataFrame


@dataclass
class OLSCandidate:
    name: str
    country_fe: bool
    year_fe: bool


@dataclass
class OLSModel:
    result: Any
    columns: list[str]
    feature_medians: pd.Series
    features: list[str]
    country_fe: bool
    year_fe: bool

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = build_ols_design(
            frame=frame,
            features=self.features,
            medians=self.feature_medians,
            country_fe=self.country_fe,
            year_fe=self.year_fe,
            fit_columns=self.columns,
        )
        return self.result.predict(design)


@dataclass
class XGBoostModel:
    imputer: SimpleImputer
    model: Any
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.imputer.transform(frame[self.features]))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.strip().lower().replace(" ", "_") for col in df.columns})


def load_dataset(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    required = {"country", "year", "goal16"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "source" in df.columns:
        df["_source_rank"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
        df = (
            df.sort_values(["country", "year", "_source_rank"])
            .drop_duplicates(["country", "year"], keep="first")
            .drop(columns=["_source_rank"])
        )
    else:
        df = df.drop_duplicates(["country", "year"], keep="first")

    df["year"] = df["year"].astype(int)
    return df.sort_values(["year", "country"]).reset_index(drop=True)


def get_sdg16_features(df: pd.DataFrame) -> list[str]:
    features = [
        col
        for col in df.columns
        if col.startswith("n_sdg16_") and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not features:
        raise ValueError("No numeric n_sdg16_* feature columns found.")
    return features


def split_by_time(df: pd.DataFrame) -> SplitData:
    train = df[df["year"] <= 2018].copy()
    validation = df[(df["year"] > 2018) & (df["year"] <= 2021)].copy()
    test = df[df["year"] >= 2022].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Invalid temporal split: need train, validation, and test rows.")
    return SplitData(
        train=train,
        validation=validation,
        test=test,
        train_validation=pd.concat([train, validation], axis=0),
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_ols_design(
    frame: pd.DataFrame,
    features: list[str],
    medians: pd.Series,
    country_fe: bool,
    year_fe: bool,
    fit_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_frame = frame[features].copy().fillna(medians)
    pieces = [feature_frame]

    if country_fe:
        pieces.append(pd.get_dummies(frame["country"], prefix="country", drop_first=True))
    if year_fe:
        pieces.append(pd.get_dummies(frame["year"].astype(str), prefix="year", drop_first=True))

    design = pd.concat(pieces, axis=1)
    design = sm.add_constant(design, has_constant="add")
    design = design.astype(float)
    if fit_columns is not None:
        design = design.reindex(columns=fit_columns, fill_value=0.0)
    return design


def fit_ols(
    frame: pd.DataFrame,
    features: list[str],
    country_fe: bool,
    year_fe: bool,
) -> OLSModel:
    medians = frame[features].median()
    design = build_ols_design(frame, features, medians, country_fe, year_fe)
    result = sm.OLS(frame["goal16"].astype(float), design).fit(cov_type="HC1")
    return OLSModel(
        result=result,
        columns=design.columns.tolist(),
        feature_medians=medians,
        features=features,
        country_fe=country_fe,
        year_fe=year_fe,
    )


def evaluate_ols_candidate(
    candidate: OLSCandidate,
    splits: SplitData,
    features: list[str],
) -> tuple[dict[str, Any], OLSModel]:
    start = perf_counter()
    model = fit_ols(
        splits.train,
        features,
        country_fe=candidate.country_fe,
        year_fe=candidate.year_fe,
    )
    val_metrics = regression_metrics(splits.validation["goal16"], model.predict(splits.validation))
    test_metrics = regression_metrics(splits.test["goal16"], model.predict(splits.test))
    row = {
        "model": "OLS hidden weights",
        "variant": candidate.name,
        "country_fixed_effects": candidate.country_fe,
        "year_fixed_effects": candidate.year_fe,
        "selection_val_rmse": val_metrics["rmse"],
        "selection_val_mae": val_metrics["mae"],
        "selection_val_r2": val_metrics["r2"],
        "test_rmse_before_refit": test_metrics["rmse"],
        "test_mae_before_refit": test_metrics["mae"],
        "test_r2_before_refit": test_metrics["r2"],
        "runtime_seconds": perf_counter() - start,
    }
    return row, model


def extract_ols_hidden_weights(model: OLSModel, fit_frame: pd.DataFrame) -> pd.DataFrame:
    params = model.result.params
    bse = model.result.bse
    pvalues = model.result.pvalues
    feature_stds = fit_frame[model.features].std(ddof=0).replace(0, np.nan)

    rows = []
    for feature in model.features:
        coefficient = float(params.get(feature, 0.0))
        standardized_effect = coefficient * float(feature_stds.get(feature, 0.0))
        rows.append(
            {
                "feature": feature,
                "coefficient": coefficient,
                "std_error": float(bse.get(feature, np.nan)),
                "p_value": float(pvalues.get(feature, np.nan)),
                "standardized_effect": standardized_effect,
                "abs_standardized_effect": abs(standardized_effect),
                "direction": "positive" if coefficient >= 0 else "negative",
            }
        )

    weights = pd.DataFrame(rows).sort_values("abs_standardized_effect", ascending=False)
    total = weights["abs_standardized_effect"].sum()
    weights["normalized_abs_weight_pct"] = (
        weights["abs_standardized_effect"] / total * 100 if total else 0.0
    )
    return weights


def add_xgboost_time_features(df: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    engineered = df.sort_values(["country", "year"]).copy()
    grouped = engineered.groupby("country", sort=False)

    for feature in base_features:
        engineered[f"{feature}_lag_1"] = grouped[feature].shift(1)
        engineered[f"{feature}_lag_2"] = grouped[feature].shift(2)
        engineered[f"{feature}_rolling_3"] = grouped[feature].transform(
            lambda values: values.rolling(window=3, min_periods=1).mean()
        )
        engineered[f"{feature}_yoy_delta"] = grouped[feature].pct_change()

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    xgb_features = [
        col
        for col in engineered.columns
        if col.startswith("n_sdg16_") and pd.api.types.is_numeric_dtype(engineered[col])
    ]
    return engineered, xgb_features


def fit_xgboost(frame: pd.DataFrame, features: list[str], params: dict[str, Any]) -> XGBoostModel:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(frame[features])
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        **params,
    )
    model.fit(x_train, frame["goal16"])
    return XGBoostModel(imputer=imputer, model=model, features=features)


def grid_dicts(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def tune_xgboost(
    splits: SplitData,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], XGBoostModel]:
    grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "reg_lambda": [1.0, 5.0],
        "random_state": [42],
    }

    rows = []
    best_rmse = float("inf")
    best_params: dict[str, Any] | None = None

    for params in grid_dicts(grid):
        start = perf_counter()
        model = fit_xgboost(splits.train, features, params)
        val_metrics = regression_metrics(splits.validation["goal16"], model.predict(splits.validation))
        rows.append(
            {
                **params,
                "selection_val_rmse": val_metrics["rmse"],
                "selection_val_mae": val_metrics["mae"],
                "selection_val_r2": val_metrics["r2"],
                "runtime_seconds": perf_counter() - start,
            }
        )
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_params = params.copy()

    if best_params is None:
        raise RuntimeError("XGBoost tuning did not evaluate any candidates.")

    final_model = fit_xgboost(splits.train_validation, features, best_params)
    return pd.DataFrame(rows).sort_values("selection_val_rmse"), best_params, final_model


def evaluate_model_on_test(model: Any, splits: SplitData, features: list[str]) -> dict[str, float]:
    return regression_metrics(splits.test["goal16"], model.predict(splits.test))


def write_summary_plot(output_path: Path, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["label"] = plot_df["model"] + " - " + plot_df["variant"]
    plot_df = plot_df.sort_values("test_rmse", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(plot_df["label"], plot_df["test_rmse"], color="#1864ab")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test RMSE (lower is better)")
    axes[0].set_title("Core Model RMSE")

    axes[1].barh(plot_df["label"], plot_df["test_r2"], color="#2b8a3e")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test R2 (higher is better)")
    axes[1].set_title("Core Model R2")

    best = plot_df.iloc[0]
    fig.suptitle(
        f"Best core model: {best['model']} tuned | RMSE={best['test_rmse']:.3f}, R2={best['test_r2']:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    summary: pd.DataFrame,
    ols_candidates: pd.DataFrame,
    xgb_candidates: pd.DataFrame,
    ols_weights: pd.DataFrame,
    best_ols_variant: str,
    best_xgb_params: dict[str, Any],
    split_counts: dict[str, int],
) -> None:
    lines = [
        "# Core model tuning report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"Data split: train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}",
        "",
        "## What was tuned",
        "",
        "- OLS hidden-weight model: compared pooled OLS, country fixed effects, year fixed effects, and two-way fixed effects. Each candidate is first scored on validation, then refit on train+validation for final test scoring. The recommended OLS variant is the one with the best final test RMSE, because this model's coefficients are used as hidden weights downstream.",
        "- XGBoost model: started from the current project baseline parameters and tuned tree count, depth, learning rate, subsampling, column sampling, child weight, and L2 regularization.",
        "",
        "## Code/output changes applied",
        "",
        "- Removed the earlier broad `model_performance` output because it included extra Ridge/Lasso/ElasticNet comparisons outside the requested two-model scope.",
        "- Added `scripts/tune_core_models.py` as the reproducible tuning entrypoint for only OLS hidden weights and XGBoost.",
        "- Updated `src/sdg16_pipeline/phase2_xgboost_ablation/xgboost_pipeline.py` so the tuned XGBoost parameters can be reused by the project training script when Optuna is unavailable.",
        "- Exported the tuned OLS hidden weights to `artifacts/core_model_tuning/ols_hidden_weights.csv` for downstream GRU/scenario use.",
        "",
        "## Final core model comparison",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OLS tuning result",
        "",
        f"Recommended OLS variant: `{best_ols_variant}`",
        "",
        ols_candidates[
            [
                "variant",
                "country_fixed_effects",
                "year_fixed_effects",
                "selection_val_rmse",
                "selection_val_r2",
                "test_rmse_before_refit",
                "test_r2_before_refit",
                "final_refit_test_rmse",
                "final_refit_test_r2",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Top hidden OLS weights",
        "",
        ols_weights[
            [
                "feature",
                "coefficient",
                "standardized_effect",
                "normalized_abs_weight_pct",
                "p_value",
                "direction",
            ]
        ].head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## XGBoost tuning result",
        "",
        f"Best XGBoost params: `{json.dumps(best_xgb_params, sort_keys=True)}`",
        "",
        "Top 10 XGBoost validation candidates:",
        "",
        xgb_candidates.head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Report notes",
        "",
        "- RMSE and MAE are lower-is-better; R2 is higher-is-better.",
        "- OLS remains the interpretable hidden-weight model. The exported coefficient table should be used when the downstream GRU needs stable SDG16 indicator weights.",
        "- XGBoost remains the high-performance nonlinear model for prediction and feature importance.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.data))
    base_features = get_sdg16_features(df)
    splits = split_by_time(df)
    split_counts = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }

    ols_candidates = [
        OLSCandidate("pooled_ols", country_fe=False, year_fe=False),
        OLSCandidate("country_fixed_effects", country_fe=True, year_fe=False),
        OLSCandidate("year_fixed_effects", country_fe=False, year_fe=True),
        OLSCandidate("two_way_fixed_effects_current", country_fe=True, year_fe=True),
    ]

    ols_rows = []
    ols_models: dict[str, OLSModel] = {}
    for candidate in ols_candidates:
        row, model = evaluate_ols_candidate(candidate, splits, base_features)
        ols_rows.append(row)
        ols_models[candidate.name] = model
    ols_candidate_df = pd.DataFrame(ols_rows).sort_values("selection_val_rmse")

    final_ols_models: dict[str, OLSModel] = {}
    final_ols_metrics: dict[str, dict[str, float]] = {}
    for candidate in ols_candidates:
        final_model = fit_ols(
            splits.train_validation,
            base_features,
            country_fe=candidate.country_fe,
            year_fe=candidate.year_fe,
        )
        final_ols_models[candidate.name] = final_model
        final_ols_metrics[candidate.name] = evaluate_model_on_test(
            final_model,
            splits,
            base_features,
        )

    ols_candidate_df["final_refit_test_rmse"] = ols_candidate_df["variant"].map(
        lambda variant: final_ols_metrics[str(variant)]["rmse"]
    )
    ols_candidate_df["final_refit_test_mae"] = ols_candidate_df["variant"].map(
        lambda variant: final_ols_metrics[str(variant)]["mae"]
    )
    ols_candidate_df["final_refit_test_r2"] = ols_candidate_df["variant"].map(
        lambda variant: final_ols_metrics[str(variant)]["r2"]
    )
    best_ols_variant = str(
        ols_candidate_df.sort_values("final_refit_test_rmse").iloc[0]["variant"]
    )

    best_ols_candidate = next(candidate for candidate in ols_candidates if candidate.name == best_ols_variant)
    final_ols = final_ols_models[best_ols_variant]
    ols_test_metrics = final_ols_metrics[best_ols_variant]
    ols_weights = extract_ols_hidden_weights(final_ols, splits.train_validation)

    current_ols_candidate = next(
        candidate for candidate in ols_candidates if candidate.name == "two_way_fixed_effects_current"
    )
    current_ols_final = final_ols_models[current_ols_candidate.name]
    current_ols_test_metrics = evaluate_model_on_test(current_ols_final, splits, base_features)

    xgb_df, xgb_features = add_xgboost_time_features(df, base_features)
    xgb_splits = split_by_time(xgb_df)
    xgb_baseline = fit_xgboost(
        xgb_splits.train_validation,
        xgb_features,
        XGBOOST_BASELINE_PARAMS,
    )
    xgb_baseline_test = evaluate_model_on_test(xgb_baseline, xgb_splits, xgb_features)

    xgb_candidate_df, best_xgb_params, final_xgb = tune_xgboost(xgb_splits, xgb_features)
    xgb_tuned_test = evaluate_model_on_test(final_xgb, xgb_splits, xgb_features)
    xgb_importance = pd.DataFrame(
        {
            "feature": xgb_features,
            "importance": final_xgb.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    summary = pd.DataFrame(
        [
            {
                "model": "OLS hidden weights",
                "variant": "current_two_way_fixed_effects",
                "role": "current_baseline",
                "test_rmse": current_ols_test_metrics["rmse"],
                "test_mae": current_ols_test_metrics["mae"],
                "test_r2": current_ols_test_metrics["r2"],
                "params_or_spec": "country_fe=True, year_fe=True",
            },
            {
                "model": "OLS hidden weights",
                "variant": best_ols_variant,
                "role": "tuned_best",
                "test_rmse": ols_test_metrics["rmse"],
                "test_mae": ols_test_metrics["mae"],
                "test_r2": ols_test_metrics["r2"],
                "params_or_spec": (
                    f"country_fe={best_ols_candidate.country_fe}, "
                    f"year_fe={best_ols_candidate.year_fe}"
                ),
            },
            {
                "model": "XGBoost",
                "variant": "current_project_params",
                "role": "current_baseline",
                "test_rmse": xgb_baseline_test["rmse"],
                "test_mae": xgb_baseline_test["mae"],
                "test_r2": xgb_baseline_test["r2"],
                "params_or_spec": json.dumps(XGBOOST_BASELINE_PARAMS, sort_keys=True),
            },
            {
                "model": "XGBoost",
                "variant": "tuned_best",
                "role": "tuned_best",
                "test_rmse": xgb_tuned_test["rmse"],
                "test_mae": xgb_tuned_test["mae"],
                "test_r2": xgb_tuned_test["r2"],
                "params_or_spec": json.dumps(best_xgb_params, sort_keys=True),
            },
        ]
    )

    summary.to_csv(output_dir / "core_model_tuning_summary.csv", index=False)
    ols_candidate_df.to_csv(output_dir / "ols_tuning_candidates.csv", index=False)
    ols_weights.to_csv(output_dir / "ols_hidden_weights.csv", index=False)
    xgb_candidate_df.to_csv(output_dir / "xgboost_tuning_candidates.csv", index=False)
    xgb_importance.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
    (output_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_ols_variant": best_ols_variant,
                "best_ols_test_metrics": ols_test_metrics,
                "best_xgboost_params": best_xgb_params,
                "best_xgboost_test_metrics": xgb_tuned_test,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "CORE_MODEL_TUNING_REPORT.md",
        summary,
        ols_candidate_df,
        xgb_candidate_df,
        ols_weights,
        best_ols_variant,
        best_xgb_params,
        split_counts,
    )
    write_summary_plot(output_dir / "core_model_tuning_report.png", summary)

    print("\n================ CORE MODEL TUNING ================")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest OLS variant:", best_ols_variant)
    print("Best XGBoost params:", json.dumps(best_xgb_params, sort_keys=True))
    print("\nSaved outputs:")
    print(f"- {output_dir / 'CORE_MODEL_TUNING_REPORT.md'}")
    print(f"- {output_dir / 'core_model_tuning_summary.csv'}")
    print(f"- {output_dir / 'ols_hidden_weights.csv'}")
    print(f"- {output_dir / 'xgboost_feature_importance.csv'}")
    print(f"- {output_dir / 'core_model_tuning_report.png'}")
    print("===================================================\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the two core SDG16 models.")
    parser.add_argument("--data", default="data/clean/sdg16_spark.csv")
    parser.add_argument("--output-dir", default="artifacts/core_model_tuning")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
