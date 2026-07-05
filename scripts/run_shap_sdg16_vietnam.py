"""Backward-compatible entrypoint for Phase 4 SHAP analysis."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdg16_pipeline.phase4_shap_explanation.run_shap_sdg16_vietnam import main


if __name__ == "__main__":
    main()
