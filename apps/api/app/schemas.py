from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    country: str = "Vietnam"
    year: int = Field(default=2025, ge=2000, le=2100)
    features: dict[str, float]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    coefficient: float
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    country: str
    year: int
    predicted_score: float
    model_version: str
    warning: str | None = None


class ExplainResponse(PredictionResponse):
    baseline: float
    contributions: list[FeatureContribution]


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    limit: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    country: str = "Vietnam"
    year: int = 2025
    features: dict[str, float] | None = None


class AskResponse(BaseModel):
    answer: str
    prediction: PredictionResponse | None = None
    evidence: list[SearchHit] = []
    provider: str


class IndicatorInsight(BaseModel):
    feature: str
    label: str
    value: float
    coefficient: float
    contribution: float
    direction: str


class ScenarioForecast(BaseModel):
    scenario: str
    year: int
    predicted_score: float
    delta_vs_current: float
    assumptions: dict[str, float]


class ProvinceInsight(BaseModel):
    data_status: str
    message: str
    weakest_province: str | None = None
    weakest_score: float | None = None


class FinalInsightResponse(BaseModel):
    country: str
    year: int
    current_score: float
    observed_score: float | None = None
    model_version: str
    explainability_source: str
    weakest_indicators: list[IndicatorInsight]
    strongest_indicators: list[IndicatorInsight]
    forecasts: list[ScenarioForecast]
    province: ProvinceInsight
    evidence: list[SearchHit]
    recommendation: str
    provider: str
    panel_ols_diagnostics: dict[str, Any] | None = None
    leakage_report: dict[str, Any] | None = None
    gru_forecast_baseline: list[dict[str, Any]] = []
