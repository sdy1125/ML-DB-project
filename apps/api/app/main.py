from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from apps.api.app.config import get_settings
from apps.api.app.schemas import (
    AskRequest,
    AskResponse,
    ExplainResponse,
    FinalInsightResponse,
    PredictionRequest,
    PredictionResponse,
    SearchHit,
    SearchRequest,
)
from apps.api.app.services.insight_service import InsightService
from apps.api.app.services.llm_service import LlmService
from apps.api.app.services.model_service import ModelNotReadyError, ModelService
from apps.api.app.services.rag_service import RagService


settings = get_settings()
model_service = ModelService(settings.model_metadata_path)
rag_service = RagService(
    qdrant_url=settings.qdrant_url,
    collection=settings.qdrant_collection,
    embedding_provider=settings.embedding_provider,
    embedding_model=settings.embedding_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    ollama_url=settings.ollama_url,
    chunks_path=settings.rag_chunks_path,
    knowledge_dir=settings.knowledge_dir,
)
llm_service = LlmService(
    provider=settings.llm_provider,
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    ollama_url=settings.ollama_url,
    ollama_model=settings.ollama_model,
)
insight_service = InsightService(
    model_service=model_service,
    rag_service=rag_service,
    llm_service=llm_service,
    clean_sdg16_path=settings.clean_sdg16_path,
    subnational_data_path=settings.subnational_data_path,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.reload()
    yield


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="Prediction, explainability, RAG and policy recommendation API.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "model_ready": model_service.ready,
        "rag_enabled": rag_service.enabled,
        "llm_enabled": llm_service.enabled,
    }


@app.get("/model/info")
def model_info() -> dict:
    return model_service.model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        score, missing = model_service.predict(request)
        artifact = model_service.artifact
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    warning = (
        f"Used training means for missing features: {', '.join(missing)}"
        if missing
        else None
    )
    return PredictionResponse(
        country=request.country,
        year=request.year,
        predicted_score=round(score, 4),
        model_version=artifact.model_version,
        warning=warning,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictionRequest) -> ExplainResponse:
    try:
        score, contributions, missing = model_service.explain(request)
        artifact = model_service.artifact
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExplainResponse(
        country=request.country,
        year=request.year,
        predicted_score=round(score, 4),
        model_version=artifact.model_version,
        baseline=artifact.intercept,
        contributions=contributions,
        warning=(
            f"Used training means for missing features: {', '.join(missing)}"
            if missing
            else None
        ),
    )


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    if not rag_service.enabled:
        raise HTTPException(
            status_code=503,
            detail="RAG search is not configured. Parse PDFs first or enable embeddings.",
        )
    try:
        return rag_service.search(request.query, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG unavailable: {exc}") from exc


@app.get("/insights/final", response_model=FinalInsightResponse)
def final_insight(
    country: str = "Vietnam",
    year: int | None = None,
    use_llm: bool = True,
) -> FinalInsightResponse:
    try:
        return insight_service.build_final_insight(
            country=country,
            year=year,
            use_llm=use_llm,
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Final insight unavailable: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    prediction = None
    explanation_text = "Chưa có dữ liệu feature để chạy mô hình."

    if request.features:
        prediction_request = PredictionRequest(
            country=request.country,
            year=request.year,
            features=request.features,
        )
        try:
            score, contributions, missing = model_service.explain(prediction_request)
            prediction = PredictionResponse(
                country=request.country,
                year=request.year,
                predicted_score=round(score, 4),
                model_version=model_service.artifact.model_version,
                warning=f"Missing features: {missing}" if missing else None,
            )
            top = contributions[:5]
            explanation_text = "\n".join(
                f"- {item.feature}: contribution={item.contribution:.4f}, "
                f"value={item.value:.4f}"
                for item in top
            )
        except ModelNotReadyError:
            explanation_text = "Model chưa được huấn luyện."

    evidence: list[SearchHit] = []
    if rag_service.enabled:
        try:
            evidence = rag_service.search(request.question, limit=5)
        except Exception:
            evidence = []

    context = "\n\n".join(
        (
            f"[Nguồn {i + 1}] "
            f"source={hit.metadata.get('source', 'unknown')}; "
            f"group={hit.metadata.get('group', 'unknown')}; "
            f"page={hit.metadata.get('page_start', '?')}-{hit.metadata.get('page_end', '?')}\n"
            f"{hit.text}"
        )
        for i, hit in enumerate(evidence)
    )
    system_prompt = (
        "Bạn là trợ lý phân tích chính sách SDG16. Chỉ đưa ra kết luận dựa trên "
        "số liệu mô hình và tài liệu PDF/RAG được cung cấp. Khi khuyến nghị, phải "
        "nêu rõ nguồn PDF liên quan theo dạng [Nguồn 1: tên_file.pdf, trang x-y]. "
        "Phân biệt rõ dự đoán, bằng chứng và giả định; không bịa số liệu. "
        "Trả lời bằng tiếng Việt."
    )
    user_prompt = f"""
Câu hỏi: {request.question}
Quốc gia: {request.country}; năm: {request.year}

Kết quả mô hình:
{explanation_text}

Tài liệu truy xuất:
{context or "Không có tài liệu RAG."}

Hãy trả lời ngắn gọn và đề xuất tối đa 3 ưu tiên chính sách có căn cứ.
Yêu cầu: mỗi khuyến nghị phải gắn với ít nhất một nguồn PDF/RAG nếu có evidence.
Nếu evidence không đủ mạnh, nói rõ hạn chế thay vì suy diễn.
""".strip()

    return AskResponse(
        answer=llm_service.answer(system_prompt, user_prompt),
        prediction=prediction,
        evidence=evidence,
        provider=llm_service.provider,
    )
