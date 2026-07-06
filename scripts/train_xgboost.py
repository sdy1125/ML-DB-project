"""Train the optional Phase 2 XGBoost model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdg16_pipeline.phase2_xgboost_ablation.xgboost_pipeline import XGBoostPipeline


def main() -> int:
    data_path = Path("data/clean/sdg16_spark.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return 1

    pipeline = XGBoostPipeline()
    metrics = pipeline.train(optimize=True)
    ablation_metrics = pipeline.ablation_study(top_n=3)
    pipeline.save_model()

    print("XGBoost training completed.")
    print(f"Metrics: {metrics}")
    print(f"Ablation: {ablation_metrics}")
    print("Artifacts written to artifacts/xgboost/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

