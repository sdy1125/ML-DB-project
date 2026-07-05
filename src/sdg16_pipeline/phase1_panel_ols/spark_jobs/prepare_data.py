import re
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.sql.window import Window

from src.sdg16_pipeline.phase1_panel_ols.config import ProjectConfig


COLUMN_ALIASES = {
    "country": {"country", "country_name", "countryname"},
    "year": {"year", "time", "period"},
    "goal16": {"goal16", "goal_16", "sdg16_score", "score_goal16"},
}


def canonicalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    for canonical, aliases in COLUMN_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def normalize_columns(df: DataFrame) -> DataFrame:
    for old_name in df.columns:
        new_name = canonicalize_name(old_name)
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
    return df


def read_and_merge(spark: SparkSession, config: ProjectConfig) -> DataFrame:
    paths = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark.sparkContext._jsc.hadoopConfiguration()
    ).globStatus(
        spark.sparkContext._jvm.org.apache.hadoop.fs.Path(config.input_glob)
    )
    if not paths:
        raise FileNotFoundError(f"No CSV files match {config.input_glob}")

    frames = []
    for status in paths:
        path = status.getPath().toString()
        frame = spark.read.option("header", True).option("inferSchema", True).csv(path)
        frame = normalize_columns(frame).withColumn("_source_file", F.lit(path))
        frames.append(frame)

    merged = reduce(
        lambda left, right: left.unionByName(right, allowMissingColumns=True),
        frames,
    )
    if config.duplicate_strategy == "error":
        duplicates = (
            merged.groupBy(*config.deduplication_keys)
            .count()
            .where(F.col("count") > 1)
            .limit(1)
            .count()
        )
        if duplicates:
            raise ValueError(
                "Duplicate country-year keys found. Configure data.source_priority."
            )
        return merged

    rank_expression = F.lit(len(config.source_priority) + 1)
    for rank, token in reversed(list(enumerate(config.source_priority))):
        rank_expression = F.when(
            F.lower(F.col("_source_file")).contains(token.lower()),
            F.lit(rank),
        ).otherwise(rank_expression)
    window = Window.partitionBy(*config.deduplication_keys).orderBy(
        F.col("_source_rank").asc(),
        F.col("_source_file").desc(),
    )
    return (
        merged.withColumn("_source_rank", rank_expression)
        .withColumn("_row_number", F.row_number().over(window))
        .where(F.col("_row_number") == 1)
        .drop("_source_rank", "_row_number")
    )


def select_and_clean(df: DataFrame, config: ProjectConfig) -> tuple[DataFrame, list[str]]:
    required_columns = {
        config.entity_column,
        config.time_column,
        config.target,
    }
    missing_required = sorted(required_columns - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    feature_candidates = [
        column
        for column in df.columns
        if column.startswith(config.optional_prefix)
        and isinstance(df.schema[column].dataType, NumericType)
    ]
    if not feature_candidates:
        raise ValueError(
            f"No numeric feature starts with prefix '{config.optional_prefix}'."
        )

    row_count = df.count()
    if row_count == 0:
        raise ValueError("Merged dataset is empty.")

    missing_counts = df.select(
        *[
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in feature_candidates
        ]
    ).first().asDict()

    retained_features = [
        column
        for column in feature_candidates
        if missing_counts[column] / row_count <= config.max_missing_ratio
        or column in config.required_features
    ]
    dropped = sorted(set(feature_candidates) - set(retained_features))
    if dropped:
        print(f"Dropping high-missing features: {dropped}")

    selected = df.select(
        config.entity_column,
        F.col(config.time_column).cast("int").alias(config.time_column),
        F.col(config.target).cast("double").alias(config.target),
        *[F.col(column).cast("double").alias(column) for column in retained_features],
    ).where(F.col(config.target).isNotNull())

    # Median is safer than replacing missing observations with zero, because zero
    # may be a meaningful SDG score. Imputation is fitted only on the train split.
    return selected, retained_features


def prepare_dataset(
    spark: SparkSession, config: ProjectConfig
) -> tuple[DataFrame, list[str]]:
    merged = read_and_merge(spark, config)
    clean, features = select_and_clean(merged, config)
    output = config.clean_output
    clean.write.mode("overwrite").parquet(output)
    print(f"Wrote {clean.count()} rows and {len(features)} features to {output}")
    return clean, features
