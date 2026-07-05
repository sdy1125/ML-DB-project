"""Backward-compatible entrypoint for the Spark Phase 1 pipeline."""

from src.sdg16_pipeline.phase1_panel_ols.run_pipeline import *  # noqa: F401,F403
from src.sdg16_pipeline.phase1_panel_ols.run_pipeline import main


if __name__ == "__main__":
    main()

