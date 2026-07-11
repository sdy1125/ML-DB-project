from pathlib import Path

from apps.api.app.services.insight_service import InsightService
from apps.api.app.services.llm_service import LlmService
from apps.api.app.services.model_service import ModelService
from apps.api.app.services.rag_service import RagService


def test_final_insight_uses_metadata_model_and_local_rag():
    service = InsightService(
        model_service=ModelService(Path("artifacts/linear_regression/metadata.json")),
        rag_service=RagService(
            qdrant_url="http://localhost:6333",
            collection="test",
            embedding_provider="disabled",
            embedding_model="",
            chunks_path=Path("data/knowledge/processed/chunks.jsonl"),
        ),
        llm_service=LlmService("disabled", "", "", "", "", ""),
        clean_sdg16_path=Path("data/clean/sdg16_spark.csv"),
    )

    insight = service.build_final_insight(country="Vietnam", use_llm=False)

    assert insight.country == "Vietnam"
    assert insight.model_version == "xgboost_shap_runner+gru_forecast+subnational_drilldown"
    assert "xgboost_shap_summary" in insight.explainability_source
    assert insight.current_score > 0
    assert len(insight.weakest_indicators) == 5
    assert len(insight.forecasts) == 21
    assert insight.forecasts[0].scenario == "pessimistic"
    assert insight.forecasts[0].year > insight.year
    assert "gru_baseline" in insight.forecasts[0].assumptions
    assert insight.forecasts[-1].scenario == "optimistic"
    assert insight.forecasts[-1].year == 2030
    assert insight.province.data_status in {"demo_provincial_dataset", "ready"}
    assert insight.province.weakest_province is not None
    assert insight.province.weakest_score is not None
