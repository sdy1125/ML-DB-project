import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipelines.modeling.common import (
    FORECAST_YEARS,
    VIETNAM_ALIASES,
    PreparedData,
    clip_score,
)


MAX_ARIMAX_EXOG = 3
MAX_REASONABLE_SCENARIO_WIDTH = 10.0


def find_vietnam_frame(prepared: PreparedData) -> pd.DataFrame:
    if not prepared.country_col:
        return prepared.frame.sort_values(prepared.year_col).copy()
    country_series = prepared.frame[prepared.country_col].astype(str).str.lower().str.strip()
    mask = country_series.isin(VIETNAM_ALIASES)
    if not mask.any():
        mask = country_series.str.contains("viet", na=False)
    if not mask.any():
        raise ValueError("Cannot find Vietnam rows for Model 3 forecast.")
    return prepared.frame.loc[mask].sort_values(prepared.year_col).copy()


def aggregate_yearly_history(history: pd.DataFrame, year_col: str) -> pd.DataFrame:
    numeric_cols = history.select_dtypes(include=[np.number]).columns.tolist()
    if year_col not in numeric_cols:
        numeric_cols.append(year_col)
    return (
        history[numeric_cols]
        .groupby(year_col, as_index=False)
        .mean(numeric_only=True)
        .sort_values(year_col)
    )


def trend_future_values(history: pd.DataFrame, cols: list[str], year_col: str, years: list[int]) -> pd.DataFrame:
    result = pd.DataFrame({year_col: years})
    x = history[year_col].to_numpy(dtype=float)
    for col in cols:
        y = pd.to_numeric(history[col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        finite_mask = np.isfinite(y)
        if len(np.unique(x)) >= 2 and finite_mask.sum() >= 2:
            slope, intercept = np.polyfit(x[finite_mask], y[finite_mask], 1)
            result[col] = intercept + slope * np.asarray(years, dtype=float)
        else:
            result[col] = y[finite_mask][-1] if finite_mask.any() else 0.0
    return result[cols]


def choose_exogenous_features(
    prepared: PreparedData,
    history: pd.DataFrame,
    preferred_features: list[str] | None,
) -> list[str]:
    target_col = prepared.target_col
    valid = [
        col
        for col in prepared.feature_cols
        if col in history.columns
        and pd.api.types.is_numeric_dtype(history[col])
        and not col.startswith(f"{target_col}_")
        and col != f"delta_{target_col}"
        and "_lag_" not in col
        and not col.startswith("delta_")
        and history[col].notna().mean() >= 0.7
    ]
    preferred = [feature for feature in (preferred_features or []) if feature in valid]
    fallback = [feature for feature in valid if feature not in preferred]
    # Light optimization: use Model 1 hidden weights to order exogenous variables.
    return (preferred + fallback)[:MAX_ARIMAX_EXOG]


def forecast_with_exog(fitted: Any, steps: int, exog: pd.DataFrame | None) -> np.ndarray:
    forecast = fitted.get_forecast(
        steps=steps,
        exog=exog,
    )
    return clip_score(forecast.predicted_mean.to_numpy())


def build_plus_one_pct_impact_table(
    fitted: Any,
    forecast_years: list[int],
    output_mask: np.ndarray,
    scenario_exog: dict[str, pd.DataFrame],
    scenario_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, base_exog in scenario_exog.items():
        baseline = scenario_scores[scenario]
        for indicator in base_exog.columns:
            intervention_exog = base_exog.copy()
            intervention_exog[indicator] = intervention_exog[indicator] * 1.01
            intervention = forecast_with_exog(fitted, len(forecast_years), intervention_exog)[output_mask]
            for year, base_score, intervention_score in zip(
                FORECAST_YEARS,
                baseline,
                intervention,
            ):
                rows.append(
                    {
                        "year": year,
                        "scenario": scenario,
                        "indicator": indicator,
                        "change": "+1%",
                        "baseline_score": base_score,
                        "intervention_score": intervention_score,
                        "score_delta": intervention_score - base_score,
                    }
                )
    return pd.DataFrame(rows)


def train_model3(
    prepared: PreparedData,
    dirs: dict[str, Path],
    preferred_exog: list[str] | None = None,
) -> dict[str, Any]:
    logging.info("Training Model 3: ARIMAX/ARIMA forecast for Vietnam")
    vietnam = find_vietnam_frame(prepared)
    year_col = prepared.year_col
    target_col = prepared.target_col
    vietnam = aggregate_yearly_history(vietnam, year_col)
    history = vietnam[vietnam[year_col] < min(FORECAST_YEARS)].copy()
    if len(history) < 6:
        raise ValueError("Not enough Vietnam yearly observations for ARIMA forecast.")

    candidate_exog = choose_exogenous_features(prepared, history, preferred_exog)
    use_exog = len(candidate_exog) >= 2
    y = history[target_col].astype(float)
    exog = history[candidate_exog].astype(float) if use_exog else None
    orders = [(1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 0, 1), (2, 0, 1)]
    best: tuple[float, Any, tuple[int, int, int]] | None = None
    warnings.filterwarnings("ignore")
    for order in orders:
        try:
            fitted = SARIMAX(
                y,
                exog=exog,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            if best is None or fitted.aic < best[0]:
                best = (float(fitted.aic), fitted, order)
        except Exception as exc:
            logging.warning("SARIMAX order %s failed: %s", order, exc)
    if best is None:
        raise ValueError("All SARIMAX/ARIMA orders failed.")

    _, fitted, order = best
    model_used = "ARIMAX" if use_exog else "ARIMA"
    last_history_year = int(history[year_col].max())
    forecast_years = list(range(last_history_year + 1, max(FORECAST_YEARS) + 1))
    output_mask = np.asarray([year in FORECAST_YEARS for year in forecast_years])
    if not output_mask.any():
        raise ValueError("Forecast horizon does not overlap configured output years.")

    future_exog = trend_future_values(history, candidate_exog, year_col, forecast_years) if use_exog else None
    forecast = fitted.get_forecast(steps=len(forecast_years), exog=future_exog)
    interval = forecast.conf_int(alpha=0.2).to_numpy()

    if use_exog:
        scenario_exog = {
            "pessimistic": future_exog * 0.95,
            "base": future_exog.copy(),
            "optimistic": future_exog * 1.05,
        }
        scenario_scores = {
            scenario: forecast_with_exog(fitted, len(forecast_years), exog_values)[output_mask]
            for scenario, exog_values in scenario_exog.items()
        }
        pessimistic = scenario_scores["pessimistic"]
        mean = scenario_scores["base"]
        optimistic = scenario_scores["optimistic"]
        scenario_width = float(np.nanmean(np.abs(optimistic - pessimistic)))
        history_std = float(history[target_col].std(ddof=0))
        max_width = max(MAX_REASONABLE_SCENARIO_WIDTH, history_std * 3)
        if scenario_width > max_width:
            pessimistic = interval[:, 0][output_mask]
            optimistic = interval[:, 1][output_mask]
            scenario_note = (
                "Scenario range uses the ARIMAX forecast interval because direct +/-5% "
                "exogenous scenarios were too unstable for the short Vietnam time series."
            )
        else:
            scenario_note = "Scenario range uses +/-5% adjustment on projected exogenous indicators."
        impact_df = build_plus_one_pct_impact_table(
            fitted,
            forecast_years,
            output_mask,
            scenario_exog,
            scenario_scores,
        )
        note = (
            f"{scenario_note} Exogenous indicators were ordered by Model 1 hidden weights "
            f"and filtered to raw indicators: {candidate_exog}."
        )
    else:
        mean = clip_score(forecast.predicted_mean.to_numpy())[output_mask]
        pessimistic = interval[:, 0][output_mask]
        optimistic = interval[:, 1][output_mask]
        impact_df = pd.DataFrame(
            columns=[
                "year",
                "scenario",
                "indicator",
                "change",
                "baseline_score",
                "intervention_score",
                "score_delta",
            ]
        )
        note = "Scenario uses forecast interval because exogenous variables were not reliable enough."

    pessimistic = clip_score(pessimistic)
    mean = clip_score(mean)
    optimistic = clip_score(optimistic)
    scenario_low = np.minimum(pessimistic, optimistic)
    scenario_high = np.maximum(pessimistic, optimistic)

    forecast_df = pd.DataFrame(
        {
            "year": FORECAST_YEARS,
            "pessimistic": scenario_low,
            "base": mean,
            "optimistic": scenario_high,
            "model_used": model_used,
            "note": f"{note} Selected order={order}.",
        }
    )
    forecast_path = dirs["predictions"] / "model3_vietnam_forecast_2025_2030.csv"
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8")
    impact_path = dirs["tables"] / "model3_policy_impact_plus1pct.csv"
    impact_df.to_csv(impact_path, index=False, encoding="utf-8")

    plt.figure(figsize=(9, 6))
    plt.plot(history[year_col], history[target_col], marker="o", label="History")
    plt.plot(forecast_df["year"], forecast_df["base"], marker="o", label="Base")
    plt.fill_between(
        forecast_df["year"],
        forecast_df["pessimistic"],
        forecast_df["optimistic"],
        alpha=0.2,
        label="Scenario range",
    )
    plt.xlabel("Year")
    plt.ylabel(target_col)
    plt.title("Model 3 - Vietnam Forecast 2025-2030")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "model3_vietnam_forecast_2025_2030.png", dpi=160)
    plt.close()
    joblib.dump(fitted, dirs["models"] / "model3_arimax_or_prophet.pkl")
    logging.info("Model 3 used %s with order %s and exog %s", model_used, order, candidate_exog)
    return {
        "forecast": forecast_path,
        "policy_impact": impact_path,
        "policy_impact_df": impact_df,
        "forecast_df": forecast_df,
        "model_used": model_used,
        "order": order,
        "exogenous_features": candidate_exog,
    }
