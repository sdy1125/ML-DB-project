"""Run all available model tuning/evaluation steps, then print comparison.

This is the one-command entrypoint when you want fresh metrics for every model:

    python scripts/optimize_and_compare_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], required: bool = True) -> bool:
    print()
    print("=" * 80)
    print(f"{name}")
    print("=" * 80)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode == 0:
        print(f"[OK] {name}")
        return True
    print(f"[FAILED] {name} exited with code {completed.returncode}")
    if required:
        raise SystemExit(completed.returncode)
    return False


def main() -> int:
    python = sys.executable

    run_step(
        "1) Run Phase-1 Panel OLS + Fixed Effects",
        [python, "scripts/run_panel_ols.py"],
        required=True,
    )
    run_step(
        "2) Tune/evaluate XGBoost + SHAP runner",
        [python, "scripts/run_shap_sdg16_vietnam.py"],
        required=True,
    )
    run_step(
        "3) Tune/evaluate GRU sequence forecaster",
        [python, "scripts/run_gru_forecast.py"],
        required=True,
    )
    run_step(
        "4) Train optional Phase-2 XGBoost pipeline",
        [python, "scripts/train_xgboost.py"],
        required=False,
    )
    run_step(
        "5) Run Phase-5 subnational drill-down",
        [python, "scripts/run_subnational.py"],
        required=False,
    )
    run_step(
        "6) Print final model comparison",
        [python, "scripts/evaluate_models.py"],
        required=True,
    )

    print()
    print("Fresh comparison files:")
    print("- artifacts/model_comparison/model_comparison.csv")
    print("- artifacts/model_comparison/model_comparison.json")
    print("- artifacts/model_comparison/best_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
