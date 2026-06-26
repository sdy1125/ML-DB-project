import argparse
from pathlib import Path

from pipelines.modeling.common import (
    ModelingConfig,
    configure_logging,
    ensure_output_dirs,
    scan_project_structure,
)
from pipelines.modeling.data import prepare_data
from pipelines.modeling.model1_elasticnet import train_model1
from pipelines.modeling.model2_xgboost import train_model2
from pipelines.modeling.model3_forecast import train_model3
from pipelines.modeling.reporting import print_run_summary, write_summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Elastic Net, XGBoost, and ARIMAX/ARIMA modeling pipeline."
    )
    parser.add_argument("--data-dir", default="data/clean", help="Folder containing clean CSV/XLSX data.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for model outputs.")
    parser.add_argument("--country-col", default=None, help="Override country column name.")
    parser.add_argument("--year-col", default=None, help="Override year column name.")
    parser.add_argument("--target-col", default=None, help="Override target score column name.")
    parser.add_argument(
        "--include-country-dummy",
        action="store_true",
        help="Include country as a categorical model feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    output_dir = project_root / args.output_dir
    dirs = ensure_output_dirs(output_dir)
    configure_logging(dirs["logs"] / "modeling_pipeline.log")

    config = ModelingConfig(
        project_root=project_root,
        data_dir=project_root / args.data_dir,
        output_dir=output_dir,
        country_col=args.country_col,
        year_col=args.year_col,
        target_col=args.target_col,
        include_country_dummy=args.include_country_dummy,
    )
    project_structure = scan_project_structure(project_root)
    prepared = prepare_data(config)
    model1 = train_model1(prepared, dirs)
    model2 = train_model2(prepared, dirs)
    model3 = train_model3(
        prepared,
        dirs,
        preferred_exog=model1.get("exog_candidates", []),
    )
    report_path = write_summary_report(prepared, project_structure, model1, model2, model3, dirs)
    print_run_summary(prepared, project_structure, model1, model2, model3, report_path, dirs)


if __name__ == "__main__":
    main()
