"""Run Phase 1 Panel OLS / Fixed Effects.

This is the official Phase 1 entrypoint for the SDG16 workflow:

    python scripts/run_panel_ols.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdg16_pipeline.phase1_panel_ols.panel_ols_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
