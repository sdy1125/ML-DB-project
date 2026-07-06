"""Run the optional Phase 5 Vietnam subnational drill-down."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdg16_pipeline.phase5_subnational_drilldown.subnational_analyzer import (
    SubnationalAnalyzer,
)


def main() -> int:
    analyzer = SubnationalAnalyzer()
    analyzer.load_data()
    rankings = analyzer.rank_provinces()
    panel_results = analyzer.run_panel_regression()
    mapping = analyzer.map_sdg_to_papi()

    Path("figures").mkdir(parents=True, exist_ok=True)
    analyzer.create_heatmap("figures/provincial_heatmap.png")
    analyzer.save_results()

    print("Subnational analysis completed.")
    print(f"Ranking year: {rankings.get('year')}")
    print(f"Panel observations: {panel_results.get('nobs')}")
    print(f"Mapped dimensions: {list(mapping.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

