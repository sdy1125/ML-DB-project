import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.modeling.common import (
    COUNTRY_ALIASES,
    TARGET_ALIASES,
    YEAR_ALIASES,
    ModelingConfig,
    PreparedData,
    snake_case,
)


def read_candidate(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def detect_column(columns: list[str], aliases: set[str]) -> str | None:
    normalized = {snake_case(col): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for col in columns:
        col_norm = snake_case(col)
        if any(alias in col_norm for alias in aliases):
            return col
    return None


def score_dataset_candidate(path: Path) -> tuple[int, int, int, int, int, pd.DataFrame]:
    df = read_candidate(path)
    renamed = df.rename(columns={col: snake_case(col) for col in df.columns})
    year_col = detect_column(list(renamed.columns), YEAR_ALIASES)
    country_col = detect_column(list(renamed.columns), COUNTRY_ALIASES)
    target_col = detect_column(list(renamed.columns), TARGET_ALIASES)
    numeric_cols = renamed.select_dtypes(include=[np.number]).columns.tolist()
    valid_rows = int(renamed.dropna(how="all").shape[0])
    return (
        1 if target_col else 0,
        1 if year_col else 0,
        1 if country_col else 0,
        len(numeric_cols),
        valid_rows,
        renamed,
    )


def find_input_file(data_dir: Path) -> tuple[Path, pd.DataFrame]:
    candidates = sorted(
        [
            path
            for suffix in ("*.csv", "*.xlsx", "*.xls")
            for path in data_dir.glob(suffix)
            if not path.name.startswith("~$")
        ]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .csv/.xlsx file found in {data_dir}. Put a clean data file in data/clean."
        )

    scored: list[tuple[tuple[int, int, int, int, int], Path, pd.DataFrame]] = []
    for path in candidates:
        try:
            *score, df = score_dataset_candidate(path)
            scored.append((tuple(score), path, df))
            logging.info("Candidate %s scored %s", path, tuple(score))
        except Exception as exc:
            logging.warning("Skip %s because it cannot be read: %s", path, exc)
    if not scored:
        raise FileNotFoundError(f"No readable .csv/.xlsx file found in {data_dir}.")

    scored.sort(key=lambda item: item[0], reverse=True)
    _, selected_path, selected_df = scored[0]
    logging.info("Using data file: %s", selected_path)
    return selected_path, selected_df


def resolve_columns(
    df: pd.DataFrame,
    country_col: str | None,
    year_col: str | None,
    target_col: str | None,
) -> tuple[str | None, str, str]:
    country = snake_case(country_col) if country_col else detect_column(list(df.columns), COUNTRY_ALIASES)
    year = snake_case(year_col) if year_col else detect_column(list(df.columns), YEAR_ALIASES)
    target = snake_case(target_col) if target_col else detect_column(list(df.columns), TARGET_ALIASES)
    if not year:
        raise ValueError("Cannot detect year column. Use --year-col to configure it.")
    if not target:
        raise ValueError("Cannot detect target score column. Use --target-col to configure it.")
    return country, year, target


def fill_by_time(df: pd.DataFrame, country_col: str | None, year_col: str) -> pd.DataFrame:
    if country_col:
        df = df.sort_values([country_col, year_col]).copy()
        value_cols = [col for col in df.columns if col != country_col]
        grouped = df.groupby(country_col, sort=False)
        df[value_cols] = grouped[value_cols].ffill()
        df[value_cols] = df.groupby(country_col, sort=False)[value_cols].bfill()
        return df
    df = df.sort_values(year_col).copy()
    return df.ffill().bfill()


def add_time_features(
    df: pd.DataFrame,
    country_col: str | None,
    year_col: str,
    target_col: str,
    max_lagged_features: int,
) -> pd.DataFrame:
    sort_cols = [year_col] if country_col is None else [country_col, year_col]
    df = df.sort_values(sort_cols).copy()
    group = df.groupby(country_col, group_keys=False) if country_col else None

    def shift(series: pd.Series, periods: int) -> pd.Series:
        return group[series.name].shift(periods) if group is not None else series.shift(periods)

    df[f"{target_col}_lag_1"] = shift(df[target_col], 1)
    df[f"{target_col}_lag_2"] = shift(df[target_col], 2)
    if group is not None:
        df[f"{target_col}_rolling_mean_3"] = group[target_col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    else:
        df[f"{target_col}_rolling_mean_3"] = df[target_col].shift(1).rolling(3, min_periods=1).mean()
    df[f"delta_{target_col}"] = df[target_col] - df[f"{target_col}_lag_1"]

    numeric_features = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {year_col, target_col}
        and not col.startswith(f"{target_col}_")
        and col != f"delta_{target_col}"
    ][:max_lagged_features]
    for col in numeric_features:
        df[f"{col}_lag_1"] = shift(df[col], 1)
        df[f"delta_{col}"] = df[col] - df[f"{col}_lag_1"]
    return df


def drop_unusable_columns(
    df: pd.DataFrame,
    country_col: str | None,
    year_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    protected = {col for col in [country_col, year_col, target_col] if col}
    for col in list(df.columns):
        if col in protected:
            continue
        series = df[col]
        if series.isna().all():
            dropped.append(col)
        elif series.nunique(dropna=True) <= 1:
            dropped.append(col)
        elif series.dtype == "object":
            mean_len = series.dropna().astype(str).str.len().mean()
            if col != country_col and (pd.isna(mean_len) or mean_len > 40 or "id" in col):
                dropped.append(col)
        elif target_col in col and col not in {
            f"{target_col}_lag_1",
            f"{target_col}_lag_2",
            f"{target_col}_rolling_mean_3",
            f"delta_{target_col}",
        }:
            dropped.append(col)
    if dropped:
        logging.info("Dropped unusable columns: %s", dropped)
    return df.drop(columns=dropped), dropped


def split_by_time(df: pd.DataFrame, year_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(year) for year in df[year_col].dropna().unique())
    if len(years) < 3:
        raise ValueError("Need at least 3 distinct years for time-based train/test split.")
    test_count = max(1, int(round(len(years) * 0.2)))
    test_years = set(years[-test_count:])
    train = df[~df[year_col].isin(test_years)].copy()
    test = df[df[year_col].isin(test_years)].copy()
    logging.info("Train years: %s-%s", train[year_col].min(), train[year_col].max())
    logging.info("Test years: %s-%s", test[year_col].min(), test[year_col].max())
    return train, test


def prepare_data(config: ModelingConfig) -> PreparedData:
    source_path, df = find_input_file(config.data_dir)
    raw_shape = df.shape
    df = df.rename(columns={col: snake_case(col) for col in df.columns})
    country_col, year_col, target_col = resolve_columns(
        df,
        config.country_col,
        config.year_col,
        config.target_col,
    )
    logging.info("Detected columns | country=%s | year=%s | target=%s", country_col, year_col, target_col)

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[year_col, target_col]).copy()
    df[year_col] = df[year_col].astype(int)
    df = fill_by_time(df, country_col, year_col)
    df = add_time_features(df, country_col, year_col, target_col, config.max_lagged_features)
    df = fill_by_time(df, country_col, year_col)
    df, _ = drop_unusable_columns(df, country_col, year_col, target_col)

    train_df, test_df = split_by_time(df, year_col)
    excluded = {target_col, year_col, f"delta_{target_col}"}
    if country_col and not config.include_country_dummy:
        excluded.add(country_col)
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and (pd.api.types.is_numeric_dtype(df[col]) or col == country_col)
    ]
    if not feature_cols:
        raise ValueError("No usable feature columns found after preprocessing.")
    logging.info("Prepared data shape: %s; feature count: %s", df.shape, len(feature_cols))
    return PreparedData(
        source_path=source_path,
        raw_shape=raw_shape,
        frame=df,
        country_col=country_col,
        year_col=year_col,
        target_col=target_col,
        feature_cols=feature_cols,
        train_df=train_df,
        test_df=test_df,
    )
