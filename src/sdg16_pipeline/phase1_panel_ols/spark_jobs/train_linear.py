import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.sdg16_pipeline.phase1_panel_ols.config import ProjectConfig


def split_by_time(
    df: DataFrame, config: ProjectConfig
) -> tuple[DataFrame, DataFrame, DataFrame]:
    train = df.where(F.col(config.time_column) <= config.train_end_year)
    validation = df.where(
        (F.col(config.time_column) > config.train_end_year)
        & (F.col(config.time_column) <= config.validation_end_year)
    )
    test = df.where(F.col(config.time_column) >= config.test_start_year)
    return train, validation, test


def evaluate(predictions: DataFrame, target: str) -> dict[str, float]:
    metrics = {}
    for metric in ("rmse", "mae", "r2"):
        evaluator = RegressionEvaluator(
            labelCol=target,
            predictionCol="prediction",
            metricName=metric,
        )
        metrics[metric] = float(evaluator.evaluate(predictions))
    return metrics


def train_and_export(
    spark: SparkSession,
    config: ProjectConfig,
    features: list[str] | None = None,
) -> dict:
    df = spark.read.parquet(config.clean_output)
    features = features or [
        column for column in df.columns if column.startswith(config.optional_prefix)
    ]
    if not features:
        raise ValueError("No features available for training.")

    train, validation, test = split_by_time(df, config)
    counts = {
        "train": train.count(),
        "validation": validation.count(),
        "test": test.count(),
    }
    if counts["train"] == 0 or counts["test"] == 0:
        raise ValueError(f"Invalid temporal split: {counts}")

    imputed_columns = [f"{feature}__imputed" for feature in features]
    imputer = Imputer(
        strategy=config.missing_strategy,
        inputCols=features,
        outputCols=imputed_columns,
    )
    assembler = VectorAssembler(
        inputCols=imputed_columns,
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=config.standardization,
    )
    regressor = LinearRegression(
        featuresCol="features",
        labelCol=config.target,
        elasticNetParam=config.elastic_net_param,
        regParam=config.reg_param,
        maxIter=200,
    )
    pipeline = Pipeline(stages=[imputer, assembler, scaler, regressor])
    model = pipeline.fit(train)

    validation_metrics = (
        evaluate(model.transform(validation), config.target)
        if counts["validation"]
        else {}
    )
    test_metrics = evaluate(model.transform(test), config.target)

    imputer_model, _, scaler_model, regression_model = model.stages
    imputer_values = imputer_model.surrogateDF.select(*features).first().asDict()
    scaler_means = list(scaler_model.mean.toArray())
    scaler_stds = list(scaler_model.std.toArray())

    # The scaler's means are computed after imputation and are the correct values
    # for reproducing online inference.
    feature_means = {
        feature: float(scaler_means[index])
        for index, feature in enumerate(features)
    }
    feature_stds = {
        feature: float(scaler_stds[index]) if scaler_stds[index] else 1.0
        for index, feature in enumerate(features)
    }
    coefficients = {
        feature: float(regression_model.coefficients[index])
        for index, feature in enumerate(features)
    }

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    spark_model_path = str(artifact_dir / "spark_pipeline")
    model.write().overwrite().save(spark_model_path)

    metadata = {
        "model_type": "spark_mllib_linear_regression",
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "target": config.target,
        "features": features,
        "intercept": float(regression_model.intercept),
        "coefficients": coefficients,
        "feature_defaults": {
            feature: float(imputer_values[feature]) for feature in features
        },
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "split": {
            "train_end_year": config.train_end_year,
            "validation_end_year": config.validation_end_year,
            "test_start_year": config.test_start_year,
            "row_counts": counts,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata["metrics"], indent=2))
    return metadata
