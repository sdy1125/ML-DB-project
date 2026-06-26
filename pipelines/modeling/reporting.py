from pathlib import Path
from typing import Any

import pandas as pd

from pipelines.modeling.common import PreparedData


def format_metrics(metrics: dict[str, float | None]) -> str:
    lines = []
    for key, value in metrics.items():
        lines.append(f"- {key}: {'not applicable' if value is None else f'{value:.4f}'}")
    return "\n".join(lines)


def top_weight_lines(weight_df: pd.DataFrame) -> str:
    lines = "\n".join(
        f"- {row.feature}: normalized_weight={row.normalized_weight:.4f}, direction={row.direction}"
        for row in weight_df.itertuples()
    )
    return lines or "- No non-zero hidden weights were found."


def policy_impact_lines(impact_df: pd.DataFrame) -> str:
    if impact_df.empty:
        return "- Not available because Model 3 did not use exogenous variables."
    summary = (
        impact_df.groupby(["scenario", "indicator"], as_index=False)["score_delta"]
        .mean()
        .sort_values("score_delta", ascending=False)
        .head(10)
    )
    return "\n".join(
        f"- {row.scenario}, {row.indicator}: average score_delta={row.score_delta:.4f}"
        for row in summary.itertuples()
    )


def write_summary_report(
    prepared: PreparedData,
    project_structure: dict[str, bool],
    model1: dict[str, Any],
    model2: dict[str, Any],
    model3: dict[str, Any],
    dirs: dict[str, Path],
) -> Path:
    report_path = dirs["tables"] / "model_summary_report.md"
    top_weights = model1.get("top_weights", pd.DataFrame())
    excluded_target_history = model1.get("excluded_target_history", [])
    policy_impact_df = model3.get("policy_impact_df", pd.DataFrame())

    if "metrics" in model2:
        model2_section = (
            "## Model 2 - XGBoost Regression\n"
            "XGBoost is used as the accuracy boost model for nonlinear relationships.\n\n"
            f"Metrics:\n{format_metrics(model2['metrics'])}\n\n"
            f"Feature importance: `{model2['importance']}`\n"
        )
    else:
        model2_section = (
            "## Model 2 - XGBoost Regression\n"
            "Model 2 was not trained because `xgboost` is not installed in this environment.\n\n"
            "Install with: `pip install xgboost`\n"
        )

    report = f"""# Modeling Summary Report

## Data Used
- Source file: `{prepared.source_path}`
- Raw shape: {prepared.raw_shape[0]} rows x {prepared.raw_shape[1]} columns
- Prepared shape: {prepared.frame.shape[0]} rows x {prepared.frame.shape[1]} columns
- Target column: `{prepared.target_col}`
- Country column: `{prepared.country_col}`
- Year column: `{prepared.year_col}`
- Project structure detected: {project_structure}

## Model Roles
- Model 1, Elastic Net Regression: explain hidden weights of input indicators.
- Model 2, XGBoost Regression: improve prediction accuracy with nonlinear patterns.
- Model 3, ARIMAX/ARIMA: forecast Vietnam score for 2025-2030 under three scenarios.

## Model 1 - Elastic Net Regression
Elastic Net estimates regularized coefficients on standardized SDG indicators. Target-history features are excluded before training so the coefficients support indicator interpretation instead of autoregressive prediction.

Metrics:
{format_metrics(model1["metrics"])}

Excluded target-history features:
{", ".join(f"`{feature}`" for feature in excluded_target_history) if excluded_target_history else "`None`"}

Top hidden weights from Model 1 feature set:
{top_weight_lines(top_weights)}

Hidden weights file: `{model1["weights"]}`

{model2_section}

## Model 3 - ARIMAX/ARIMA Forecast
Model used: `{model3["model_used"]}` with order `{model3["order"]}`.

Exogenous features used: `{model3.get("exogenous_features", [])}`

Forecast file: `{model3["forecast"]}`

Policy impact file (+1% per exogenous indicator): `{model3["policy_impact"]}`

Top average +1% policy impacts:
{policy_impact_lines(policy_impact_df)}

## Assumptions
- Data was split by time, with the most recent years reserved for testing.
- Missing values were forward/backward-filled within country when panel data was available, then train-set median imputation was used inside model pipelines.
- Lag and rolling features were generated within country for panel data.
- Model 1 excludes historical features of the target score, such as target lags and target rolling means, because its role is interpretation of SDG indicators.
- Country dummies are disabled by default so Model 1 hidden weights focus on numeric indicators.
- Model 3 uses Model 1 hidden weights only to order candidate exogenous variables; it does not compare or select a best model.
- Model 3 policy impact is a what-if sensitivity test: one exogenous indicator is increased by 1% while the selected scenario context is held fixed.

## Current Limitations
- Model 2 requires the external `xgboost` package.
- Forecast scenarios depend on simple historical trend extrapolation for exogenous variables.
- The pipeline does not compare the three models or choose a best model because each model has a separate role.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def print_run_summary(
    prepared: PreparedData,
    project_structure: dict[str, bool],
    model1: dict[str, Any],
    model2: dict[str, Any],
    model3: dict[str, Any],
    report_path: Path,
    dirs: dict[str, Path],
) -> None:
    print("\n=== PROJECT STRUCTURE DETECTED ===")
    for key, exists in project_structure.items():
        print(f"{key}: {'yes' if exists else 'no'}")
    print("\n=== DATA ===")
    print(f"Data file used: {prepared.source_path}")
    print(f"Target column: {prepared.target_col}")
    print(f"Country column: {prepared.country_col}")
    print(f"Year column: {prepared.year_col}")
    print(f"Prepared shape: {prepared.frame.shape[0]} rows x {prepared.frame.shape[1]} columns")
    print("\n=== MODEL 1 OUTPUTS ===")
    print(f"Metrics: {model1['metrics']}")
    print(f"Hidden weights: {model1['weights']}")
    print(f"Predictions: {model1['predictions']}")
    print("\n=== MODEL 2 OUTPUTS ===")
    if "metrics" in model2:
        print(f"Metrics: {model2['metrics']}")
        print(f"Feature importance: {model2['importance']}")
        print(f"Predictions: {model2['predictions']}")
    else:
        print("XGBoost was not trained. Install dependency with: pip install xgboost")
        print(f"Note: {model2['note']}")
    print("\n=== MODEL 3 OUTPUTS ===")
    print(model3["forecast_df"].to_string(index=False))
    print(f"Exogenous features: {model3.get('exogenous_features', [])}")
    print(f"Forecast: {model3['forecast']}")
    print(f"Policy impact (+1%): {model3['policy_impact']}")
    print("\n=== CREATED OUTPUT ROOTS ===")
    for key in ["models", "tables", "figures", "predictions", "logs"]:
        print(f"{key}: {dirs[key]}")
    print(f"Report: {report_path}")
    print("\nRun again from project root:")
    print("python -m pipelines.modeling_pipeline")
