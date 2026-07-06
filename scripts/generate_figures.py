"""Generate optional final report figures."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdg16_pipeline.final_output.generate_figures import generate_figures


if __name__ == "__main__":
    raise SystemExit(generate_figures())

