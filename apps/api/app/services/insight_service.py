import csv
from pathlib import Path

from apps.api.app.schemas import (
    FinalInsightResponse,
    IndicatorInsight,
    PredictionRequest,
    ProvinceInsight,
    ScenarioForecast,
    SearchHit,
)
from apps.api.app.services.llm_service import LlmService
from apps.api.app.services.model_service import ModelService
from apps.api.app.services.rag_service import RagService


FEATURE_LABELS = {
    "n_sdg16_admin": "Hành chính minh bạch",
    "n_sdg16_clabor": "Lao động trẻ em",
    "n_sdg16_cpi": "Chống tham nhũng / CPI",
    "n_sdg16_crime": "Tội phạm chung",
    "n_sdg16_crimepov": "Tội phạm và nghèo đói",
    "n_sdg16_detain": "Tạm giam trước xét xử",
    "n_sdg16_exprop": "Chống tịch thu tài sản",
    "n_sdg16_homicide": "Tỷ lệ giết người",
    "n_sdg16_homicides": "Tử vong do bạo lực",
    "n_sdg16_justice": "Tiếp cận tư pháp",
    "n_sdg16_power": "Quyền lực chia sẻ",
    "n_sdg16_prs": "Ổn định chính trị",
    "n_sdg16_rsf": "Tự do báo chí / trách nhiệm giải trình",
    "n_sdg16_safe": "Cảm giác an toàn",
    "n_sdg16_security": "An ninh công cộng",
    "n_sdg16_u5reg": "Đăng ký khai sinh",
    "n_sdg16_weaponsexp": "Xuất khẩu vũ khí",
}


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


class InsightService:
    def __init__(
        self,
        model_service: ModelService,
        rag_service: RagService,
        llm_service: LlmService,
        clean_sdg16_path: Path,
        shap_output_dir: Path | None = None,
    ):
        self.model_service = model_service
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.clean_sdg16_path = clean_sdg16_path
        self.shap_output_dir = shap_output_dir or Path("shap_output")

    def build_final_insight(
        self,
        country: str = "Vietnam",
        year: int | None = None,
        use_llm: bool = True,
    ) -> FinalInsightResponse:
        features, observed_score, data_year = self._latest_country_features(country, year)
        if year is None:
            year = data_year

        prediction_request = PredictionRequest(
            country=country,
            year=year,
            features=features,
        )
        current_score, contributions, _ = self.model_service.explain(prediction_request)
        artifact = self.model_service.artifact

        indicator_rows = [
            IndicatorInsight(
                feature=item.feature,
                label=feature_label(item.feature),
                value=item.value,
                coefficient=item.coefficient,
                contribution=item.contribution,
                direction=item.direction,
            )
            for item in contributions
        ]
        shap_rows = self._load_shap_gap_rows()
        explainability_source = "linear_metadata_contribution_fallback_for_shap"
        if shap_rows:
            indicator_rows = shap_rows
            explainability_source = "shap_gap_analysis_csv"

        weakest = sorted(indicator_rows, key=lambda item: item.contribution)[:5]
        strongest = sorted(indicator_rows, key=lambda item: item.contribution, reverse=True)[:5]
        forecasts = self._scenario_forecasts(country, year, features, current_score, weakest)

        evidence = self._retrieve_policy_evidence(country, weakest)
        province = ProvinceInsight(
            data_status="missing_provincial_dataset",
            message=(
                "Chưa có dataset PAPI/PCI cấp tỉnh trong project, nên hệ thống chưa kết luận "
                "tỉnh nào tệ nhất. Khi thêm dữ liệu tỉnh, phase drill-down có thể map top "
                "chỉ số yếu sang province-level FE."
            ),
        )

        recommendation = self._recommendation(
            country=country,
            year=year,
            current_score=current_score,
            observed_score=observed_score,
            weakest=weakest,
            forecasts=forecasts,
            evidence=evidence,
            province=province,
            use_llm=use_llm,
        )

        return FinalInsightResponse(
            country=country,
            year=year,
            current_score=round(current_score, 4),
            observed_score=round(observed_score, 4) if observed_score is not None else None,
            model_version=artifact.model_version,
            explainability_source=explainability_source,
            weakest_indicators=weakest,
            strongest_indicators=strongest,
            forecasts=forecasts,
            province=province,
            evidence=evidence,
            recommendation=recommendation,
            provider=self.llm_service.provider,
        )

    def _latest_country_features(
        self,
        country: str,
        year: int | None,
    ) -> tuple[dict[str, float], float | None, int]:
        artifact = self.model_service.artifact
        features = dict(artifact.feature_defaults)
        observed_score = None
        data_year = year or 2026

        if not self.clean_sdg16_path.exists():
            return features, observed_score, data_year

        rows = []
        with self.clean_sdg16_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Country", "").strip().lower() != country.lower():
                    continue
                try:
                    row_year = int(float(row.get("Year", "")))
                except ValueError:
                    continue
                if year is not None and row_year > year:
                    continue
                rows.append((row_year, row))

        if not rows:
            return features, observed_score, data_year

        data_year, latest = max(rows, key=lambda item: item[0])
        for feature in artifact.coefficients:
            raw = latest.get(feature)
            try:
                if raw not in {None, ""}:
                    features[feature] = float(raw)
            except ValueError:
                pass
        try:
            if latest.get("goal16") not in {None, ""}:
                observed_score = float(latest["goal16"])
        except ValueError:
            observed_score = None
        return features, observed_score, data_year

    def _load_shap_gap_rows(self) -> list[IndicatorInsight]:
        artifact = self.model_service.artifact
        candidates = [
            self.shap_output_dir / "gap_analysis.csv",
            Path("artifacts") / "shap" / "gap_analysis.csv",
            Path("shap_output") / "gap_analysis.csv",
        ]
        gap_path = next((path for path in candidates if path.exists()), None)
        if gap_path is None:
            return []

        rows: list[IndicatorInsight] = []
        with gap_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                feature = row.get("feature", "")
                if feature not in artifact.coefficients:
                    continue
                try:
                    value = float(row.get("vn_2022", row.get("value", 0)) or 0)
                    contribution = float(
                        row.get("shap_vn2022", row.get("shap", row.get("contribution", 0))) or 0
                    )
                except ValueError:
                    continue
                rows.append(
                    IndicatorInsight(
                        feature=feature,
                        label=row.get("label") or feature_label(feature),
                        value=value,
                        coefficient=artifact.coefficients.get(feature, 0.0),
                        contribution=contribution,
                        direction="positive" if contribution >= 0 else "negative",
                    )
                )
        return rows

    def _scenario_forecasts(
        self,
        country: str,
        year: int,
        features: dict[str, float],
        current_score: float,
        weakest: list[IndicatorInsight],
    ) -> list[ScenarioForecast]:
        scenarios = {
            "pessimistic": -5.0,
            "base": 0.0,
            "optimistic": 10.0,
        }
        outputs = []
        target_features = [item.feature for item in weakest[:3]]
        for name, delta in scenarios.items():
            scenario_features = dict(features)
            assumptions = {}
            for item in weakest[:3]:
                feature = item.feature
                current = scenario_features.get(feature, 0.0)
                direction = 1.0 if item.coefficient >= 0 else -1.0
                scenario_features[feature] = clamp_score(current + delta * direction)
                assumptions[feature] = scenario_features[feature]
            score, _ = self.model_service.predict(
                PredictionRequest(
                    country=country,
                    year=year,
                    features=scenario_features,
                )
            )
            outputs.append(
                ScenarioForecast(
                    scenario=name,
                    predicted_score=round(score, 4),
                    delta_vs_current=round(score - current_score, 4),
                    assumptions=assumptions,
                )
            )
        return outputs

    def _retrieve_policy_evidence(
        self,
        country: str,
        weakest: list[IndicatorInsight],
    ) -> list[SearchHit]:
        labels = " ".join(item.label for item in weakest[:5])
        features = " ".join(item.feature for item in weakest[:5])
        query = (
            f"{country} policy recommendation competitiveness governance FDI "
            f"anti corruption justice security digital government {labels} {features}"
        )
        if not self.rag_service.enabled:
            return []
        try:
            return self.rag_service.search(query, limit=6)
        except Exception:
            return []

    def _recommendation(
        self,
        country: str,
        year: int,
        current_score: float,
        observed_score: float | None,
        weakest: list[IndicatorInsight],
        forecasts: list[ScenarioForecast],
        evidence: list[SearchHit],
        province: ProvinceInsight,
        use_llm: bool,
    ) -> str:
        weakest_text = "\n".join(
            f"- {item.label} ({item.feature}): value={item.value:.2f}, "
            f"contribution={item.contribution:.3f}"
            for item in weakest
        )
        forecast_text = "\n".join(
            f"- {item.scenario}: score={item.predicted_score:.2f}, "
            f"delta={item.delta_vs_current:+.2f}"
            for item in forecasts
        )
        evidence_text = "\n\n".join(
            f"[Nguồn {index + 1}] {hit.metadata.get('source', 'unknown')}: {hit.text}"
            for index, hit in enumerate(evidence)
        )
        province_text = f"{province.data_status}: {province.message}"

        if not use_llm or not self.llm_service.enabled:
            return (
                f"Điểm mô hình hiện tại của {country} năm {year}: {current_score:.2f}. "
                f"Nhóm chỉ số cần ưu tiên gồm: "
                f"{', '.join(item.label for item in weakest[:3])}. "
                "Khuyến nghị sơ bộ: tập trung cải cách thể chế, minh bạch hóa thực thi, "
                "và gắn cải thiện chỉ số với chương trình chuyển đổi số/chất lượng dịch vụ công."
            )

        system_prompt = (
            "Bạn là trợ lý chính sách cho đề tài năng lực cạnh tranh quốc gia. "
            "Chỉ dùng số liệu mô hình, SHAP/contribution fallback và evidence RAG được cung cấp. "
            "Không bịa dữ liệu tỉnh nếu province dataset đang thiếu. Trả lời tiếng Việt, có cấu trúc."
        )
        user_prompt = f"""
Nhiệm vụ: tạo output cuối cùng cho dashboard/paper.

Quốc gia: {country}
Năm phân tích: {year}
Điểm mô hình hiện tại: {current_score:.2f}
Điểm quan sát nếu có: {observed_score if observed_score is not None else "không có"}

Top chỉ số kéo điểm xuống:
{weakest_text}

Dự báo kịch bản:
{forecast_text}

Drill-down tỉnh:
{province_text}

Evidence RAG:
{evidence_text or "Không có evidence RAG."}

Hãy trả về:
1. Điểm VN hiện tại và ý nghĩa.
2. Chỉ số yếu nhất cần ưu tiên.
3. Trạng thái tỉnh tệ nhất: nếu thiếu dataset thì nói rõ chưa kết luận.
4. Ba khuyến nghị chính sách cụ thể, có liên hệ evidence.
5. Một đoạn cảnh báo phương pháp: contribution hiện là fallback từ metadata nếu chưa có shap_output.
""".strip()
        return self.llm_service.answer(system_prompt, user_prompt)
