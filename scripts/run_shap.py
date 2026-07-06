"""Run the optional Phase 4 SHAP analyzer for the XGBoost artifact."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> int:
    try:
        from src.sdg16_pipeline.phase4_shap_explanation.shap_analyzer import SHAPAnalyzer
    except ModuleNotFoundError as exc:
        print(
            "Optional SHAP dependencies are not installed. "
            "Run: pip install -r requirements/experiments.txt"
        )
        print(f"Missing module: {exc.name}")
        return 1

    print(
        "SHAPAnalyzer is available. For the current lightweight project path, "
        "use scripts/run_shap_sdg16_vietnam.py. For the optional XGBoost artifact, "
        "import SHAPAnalyzer from src.sdg16_pipeline.phase4_shap_explanation."
    )
    _ = SHAPAnalyzer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
