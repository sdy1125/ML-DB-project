"""Evaluate and compare available SDG16 model artifacts.

This script is intentionally lightweight: it reads metrics already produced by
the Spark linear baseline and the XGBoost SHAP runner, then exports one
comparison table for reporting/tuning decisions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("artifacts/model_comparison")
LINEAR_METADATA = Path("artifacts/linear_regression/metadata.json")
SHAP_SUMMARY = Path("artifacts/shap/shap_summary.json")
XGBOOST_METADATA = Path("artifacts/xgboost/metadata.json")
GRU_SUMMARY = Path("artifacts/gru/gru_summary.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_block(metrics: dict[str, Any], preferred_split: str = "test") -> tuple[str, dict[str, float]]:
    if preferred_split in metrics and isinstance(metrics[preferred_split], dict):
        return preferred_split, metrics[preferred_split]
    if "validation" in metrics and isinstance(metrics["validation"], dict):
        return "validation", metrics["validation"]
    return "overall", metrics


def _row(
    model_name: str,
    model_type: str,
    artifact_path: Path,
    metrics: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    split, values = _metric_block(metrics)
    rmse = values.get("rmse")
    mae = values.get("mae")
    r2 = values.get("r2")
    return {
        "model_name": model_name,
        "model_type": model_type,
        "evaluation_split": split,
        "rmse": round(float(rmse), 6) if rmse is not None else None,
        "mae": round(float(mae), 6) if mae is not None else None,
        "r2": round(float(r2), 6) if r2 is not None else None,
        "artifact_path": str(artifact_path),
        "notes": notes,
    }


def collect_model_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    linear = _load_json(LINEAR_METADATA)
    if linear:
        rows.append(
            _row(
                model_name="Spark Linear Regression",
                model_type=linear.get("model_type", "linear_regression"),
                artifact_path=LINEAR_METADATA,
                metrics=linear.get("metrics", {}),
                notes="Baseline/reverse-engineering weights from Spark MLlib.",
            )
        )

    shap = _load_json(SHAP_SUMMARY)
    if shap:
        rows.append(
            _row(
                model_name="XGBoost SHAP Runner",
                model_type="xgboost_regressor_pred_contribs",
                artifact_path=SHAP_SUMMARY,
                metrics=shap.get("metrics", {}),
                notes="Current best candidate; exact tree contributions via XGBoost pred_contribs.",
            )
        )

    xgboost = _load_json(XGBOOST_METADATA)
    if xgboost:
        rows.append(
            _row(
                model_name="Optional XGBoost Tuned Pipeline",
                model_type="xgboost_regressor_optuna_optional",
                artifact_path=XGBOOST_METADATA,
                metrics=xgboost.get("metrics", {}),
                notes="Optional phase-2 pipeline; may use default params if Optuna is unavailable.",
            )
        )

    gru = _load_json(GRU_SUMMARY)
    if gru:
        rows.append(
            _row(
                model_name="GRU Sequence Forecaster",
                model_type=gru.get("model_type", "pytorch_gru_regressor"),
                artifact_path=GRU_SUMMARY,
                metrics=gru.get("metrics", {}),
                notes="Phase-3 sequence model using 5-year country histories; useful for 2024-2030 forecasting.",
            )
        )

    return rows


def select_best_model(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("rmse") is not None]
    if not scored:
        return None
    return sorted(
        scored,
        key=lambda row: (
            float(row["rmse"]),
            -(float(row["r2"]) if row.get("r2") is not None else -999.0),
        ),
    )[0]


def write_outputs(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    json_path = OUTPUT_DIR / "model_comparison.json"
    summary_path = OUTPUT_DIR / "best_model.json"

    fieldnames = [
        "model_name",
        "model_type",
        "evaluation_split",
        "rmse",
        "mae",
        "r2",
        "artifact_path",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        json.dumps(best or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    rows = collect_model_rows()
    if not rows:
        print("No model metrics found. Run the Spark pipeline or SHAP/XGBoost runner first.")
        return 1

    best = select_best_model(rows)
    write_outputs(rows, best)

    print("Model comparison")
    print("================")
    for row in sorted(rows, key=lambda item: item.get("rmse") or 999999):
        print(
            f"- {row['model_name']}: split={row['evaluation_split']}, "
            f"RMSE={row['rmse']}, MAE={row['mae']}, R2={row['r2']}"
        )
    if best:
        print()
        print(
            "Best model by lowest RMSE: "
            f"{best['model_name']} (RMSE={best['rmse']}, R2={best['r2']})"
        )
    print(f"\nOutputs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
