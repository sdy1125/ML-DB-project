from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "SDG16 Intelligence Platform"
    environment: str = "development"
    log_level: str = "INFO"

    clean_data_dir: Path = Path("/workspace/data/clean")
    clean_sdg16_path: Path = Path("/workspace/data/clean/sdg16_spark.csv")
    knowledge_dir: Path = Path("/workspace/data/knowledge")
    artifact_dir: Path = Path("/workspace/artifacts")
    model_metadata_path: Path = Path(
        "/workspace/artifacts/linear_regression/metadata.json"
    )
    rag_chunks_path: Path = Path("/workspace/data/knowledge/processed/chunks.jsonl")
    subnational_data_path: Path = Path("/workspace/data/subnational/sdg16_provinces.csv")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "sdg16_knowledge"
    embedding_provider: str = "disabled"
    embedding_model: str = "text-embedding-3-small"

    llm_provider: str = "disabled"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:7b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
