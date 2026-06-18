import argparse
import os

from pyspark.sql import SparkSession

from pipelines.config import load_config
from pipelines.spark_jobs.prepare_data import prepare_dataset
from pipelines.spark_jobs.train_linear import train_and_export


def create_spark_session() -> SparkSession:
    master = os.getenv("SPARK_MASTER_URL", "local[*]")
    return (
        SparkSession.builder.appName("sdg16-pipeline")
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SDG16 Spark data/ML pipeline")
    parser.add_argument(
        "stage",
        choices=["prepare", "train", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        default="/workspace/configs/project.yaml",
        help="Path to project YAML config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        features = None
        if args.stage in {"prepare", "all"}:
            _, features = prepare_dataset(spark, config)
        if args.stage in {"train", "all"}:
            train_and_export(spark, config, features)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

