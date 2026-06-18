from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ProjectConfig:
    target: str
    entity_column: str
    time_column: str
    input_glob: str
    clean_output: str
    deduplication_keys: list[str]
    duplicate_strategy: str
    source_priority: list[str]
    missing_strategy: str
    max_missing_ratio: float
    required_features: list[str]
    optional_prefix: str
    train_end_year: int
    validation_end_year: int
    test_start_year: int
    elastic_net_param: float
    reg_param: float
    standardization: bool
    artifact_dir: str


def load_config(path: str | Path) -> ProjectConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig(
        target=payload["project"]["target"],
        entity_column=payload["project"]["entity_column"],
        time_column=payload["project"]["time_column"],
        input_glob=payload["data"]["input_glob"],
        clean_output=payload["data"]["clean_output"],
        deduplication_keys=payload["data"]["deduplication_keys"],
        duplicate_strategy=payload["data"].get(
            "duplicate_strategy", "source_priority"
        ),
        source_priority=payload["data"].get("source_priority", []),
        missing_strategy=payload["data"]["missing_strategy"],
        max_missing_ratio=float(payload["data"]["max_missing_ratio"]),
        required_features=payload["features"]["required"],
        optional_prefix=payload["features"]["optional_prefix"],
        train_end_year=int(payload["training"]["train_end_year"]),
        validation_end_year=int(payload["training"]["validation_end_year"]),
        test_start_year=int(payload["training"]["test_start_year"]),
        elastic_net_param=float(payload["training"]["elastic_net_param"]),
        reg_param=float(payload["training"]["reg_param"]),
        standardization=bool(payload["training"]["standardization"]),
        artifact_dir=payload["training"]["artifact_dir"],
    )
