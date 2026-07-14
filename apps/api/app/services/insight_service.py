import csv
import json
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
        subnational_data_path: Path | None = None,
        shap_output_dir: Path | None = None,
    ):
        self.model_service = model_service
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.clean_sdg16_path = clean_sdg16_path
        self.subnational_data_path = subnational_data_path or Path(
            "data/subnational/sdg16_provinces.csv"
        )
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
        shap_current_score = self._load_shap_current_score(country)
        if shap_current_score is not None:
            current_score = shap_current_score
            explainability_source = f"{explainability_source}+xgboost_shap_summary"

        weakest = sorted(indicator_rows, key=lambda item: item.contribution)[:5]
        strongest = sorted(indicator_rows, key=lambda item: item.contribution, reverse=True)[:5]
        forecasts = self._scenario_forecasts(country, year, features, current_score, weakest)

        evidence = self._retrieve_policy_evidence(country, weakest)
        province = self._province_insight(weakest)

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
        panel_ols_diagnostics = self._panel_ols_diagnostics()
        leakage_report = self._leakage_report()
        gru_forecast_baseline = self._gru_forecast_baseline(year)

        return FinalInsightResponse(
            country=country,
            year=year,
            current_score=round(current_score, 4),
            observed_score=round(observed_score, 4) if observed_score is not None else None,
            model_version=(
                "xgboost_shap_runner+gru_forecast+subnational_drilldown"
                if shap_current_score is not None
                else artifact.model_version
            ),
            explainability_source=explainability_source,
            weakest_indicators=weakest,
            strongest_indicators=strongest,
            forecasts=forecasts,
            province=province,
            evidence=evidence,
            recommendation=recommendation,
            provider=self.llm_service.provider,
            panel_ols_diagnostics=panel_ols_diagnostics,
            leakage_report=leakage_report,
            gru_forecast_baseline=gru_forecast_baseline,
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
            self.shap_output_dir / "policy_priority_indicators.csv",
            Path("artifacts") / "shap" / "policy_priority_indicators.csv",
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
                if row.get("headline_eligible", "true").strip().lower() in {"false", "0", "no"}:
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
                        label=feature_label(feature),
                        value=value,
                        coefficient=artifact.coefficients.get(feature, 0.0),
                        contribution=contribution,
                        direction="positive" if contribution >= 0 else "negative",
                    )
                )
        return rows

    def _load_shap_current_score(self, country: str) -> float | None:
        if country.lower() not in {"vietnam", "viet nam", "việt nam"}:
            return None

        summary_candidates = [
            self.shap_output_dir / "shap_summary.json",
            Path("artifacts") / "shap" / "shap_summary.json",
        ]
        summary_path = next((path for path in summary_candidates if path.exists()), None)
        if summary_path is None:
            return None
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return float(payload["vietnam_latest_predicted"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _panel_ols_diagnostics(self) -> dict | None:
        path = Path("artifacts") / "panel_ols" / "panel_ols_results.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        both_fe = payload.get("models", {}).get("both_fe", {})
        diagnostics = payload.get("diagnostics", {}).get("both_fe", {})
        vif = payload.get("multicollinearity", {}).get("vif", {})
        top_vif = [
            {"feature": feature, "vif": round(float(value), 4)}
            for feature, value in list(vif.items())[:5]
        ]
        return {
            "model_type": payload.get("model_type"),
            "r2_overall": both_fe.get("r2_overall"),
            "r2_within": both_fe.get("r2_within"),
            "rmse": diagnostics.get("rmse"),
            "mae": diagnostics.get("mae"),
            "durbin_watson_panel_mean": diagnostics.get("durbin_watson_panel_mean"),
            "abs_residual_fitted_correlation": diagnostics.get("abs_residual_fitted_correlation"),
            "heteroskedasticity_flag": diagnostics.get("heteroskedasticity_flag"),
            "top_vif": top_vif,
        }

    def _leakage_report(self) -> dict | None:
        path = Path("artifacts") / "shap" / "shap_summary.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        report = payload.get("leakage_correlation_report")
        if not isinstance(report, dict):
            return None
        return {
            "leakage_status": report.get("leakage_status"),
            "feature_policy": report.get("feature_policy"),
            "circular_target_warning": report.get("circular_target_warning"),
            "recommended_framing": report.get("recommended_framing"),
            "exact_duplicate_features": report.get("exact_duplicate_features", []),
            "high_corr_features_abs_ge_0_98": report.get("high_corr_features_abs_ge_0_98", {}),
            "suspicious_numeric_columns_abs_ge_0_98": report.get(
                "suspicious_numeric_columns_abs_ge_0_98", {}
            ),
            "top_abs_correlations": report.get("top_abs_correlations", [])[:5],
        }

    def _gru_forecast_baseline(self, year: int) -> list[dict]:
        forecast_path = Path("artifacts") / "gru" / "vietnam_forecast_2024_2030.csv"
        if not forecast_path.exists():
            return []
        rows: list[dict] = []
        with forecast_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    forecast_year = int(float(row.get("year", "")))
                    score = float(row.get("predicted_goal16", ""))
                except ValueError:
                    continue
                if forecast_year <= year:
                    continue
                trend_norm = None
                try:
                    if row.get("feature_trend_norm") not in {None, ""}:
                        trend_norm = float(row["feature_trend_norm"])
                except ValueError:
                    trend_norm = None
                rows.append(
                    {
                        "year": forecast_year,
                        "predicted_goal16": round(score, 4),
                        "feature_trend_norm": round(trend_norm, 6)
                        if trend_norm is not None
                        else None,
                    }
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
        gru_outputs = self._gru_scenario_forecasts(year, current_score, weakest)
        if gru_outputs:
            return gru_outputs

        scenario_progress_per_year = {
            "pessimistic": 0.04,
            "base": 0.12,
            "optimistic": 0.22,
        }
        outputs = []
        benchmark_targets = self._benchmark_targets()
        forecast_start = max(year + 1, 2024)
        forecast_end = 2030

        for name, annual_progress in scenario_progress_per_year.items():
            for forecast_year in range(forecast_start, forecast_end + 1):
                years_ahead = forecast_year - year
                progress = min(1.0, max(0.0, annual_progress * years_ahead))
                scenario_features = dict(features)
                assumptions = {}
                for item in weakest[:3]:
                    feature = item.feature
                    current = float(features.get(feature, scenario_features.get(feature, 0.0)))
                    target = benchmark_targets.get(feature)
                    if target is None:
                        direction = 1.0 if item.coefficient >= 0 else -1.0
                        target = clamp_score(current + 20.0 * direction)
                    next_value = clamp_score(current + (target - current) * progress)
                    scenario_features[feature] = next_value
                    assumptions[feature] = round(next_value, 4)

                score, _ = self.model_service.predict(
                    PredictionRequest(
                        country=country,
                        year=forecast_year,
                        features=scenario_features,
                    )
                )
                outputs.append(
                    ScenarioForecast(
                        scenario=name,
                        year=forecast_year,
                        predicted_score=round(score, 4),
                        delta_vs_current=round(score - current_score, 4),
                        assumptions=assumptions,
                    )
                )
        return outputs

    def _gru_scenario_forecasts(
        self,
        year: int,
        current_score: float,
        weakest: list[IndicatorInsight],
    ) -> list[ScenarioForecast]:
        forecast_candidates = [
            Path("artifacts") / "gru" / "vietnam_forecast_2024_2030.csv",
        ]
        forecast_path = next((path for path in forecast_candidates if path.exists()), None)
        if forecast_path is None:
            return []

        baseline_rows: list[tuple[int, float]] = []
        with forecast_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    forecast_year = int(float(row.get("year", "")))
                    score = float(row.get("predicted_goal16", ""))
                except ValueError:
                    continue
                if forecast_year > year:
                    baseline_rows.append((forecast_year, score))

        if not baseline_rows:
            return []

        scenario_adjustment_per_year = {
            "pessimistic": -0.15,
            "base": 0.0,
            "optimistic": 0.35,
        }
        benchmark_targets = self._benchmark_targets()
        first_forecast_year = min(row[0] for row in baseline_rows)
        outputs: list[ScenarioForecast] = []

        for scenario, annual_adjustment in scenario_adjustment_per_year.items():
            for forecast_year, baseline_score in baseline_rows:
                years_ahead = forecast_year - first_forecast_year + 1
                predicted_score = clamp_score(baseline_score + annual_adjustment * years_ahead)
                assumptions = {
                    "gru_baseline": round(baseline_score, 4),
                }
                if scenario != "base":
                    for item in weakest[:3]:
                        target = benchmark_targets.get(item.feature)
                        if target is not None:
                            assumptions[item.feature] = round(target, 4)
                outputs.append(
                    ScenarioForecast(
                        scenario=scenario,
                        year=forecast_year,
                        predicted_score=round(predicted_score, 4),
                        delta_vs_current=round(predicted_score - current_score, 4),
                        assumptions=assumptions,
                    )
                )
        return outputs

    def _benchmark_targets(self) -> dict[str, float]:
        candidates = [
            self.shap_output_dir / "gap_analysis.csv",
            Path("artifacts") / "shap" / "gap_analysis.csv",
            Path("shap_output") / "gap_analysis.csv",
        ]
        gap_path = next((path for path in candidates if path.exists()), None)
        if gap_path is None:
            return {}

        targets: dict[str, float] = {}
        with gap_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                feature = row.get("feature")
                if not feature:
                    continue
                try:
                    target = float(row.get("top20_mean", "") or "")
                except ValueError:
                    continue
                targets[feature] = clamp_score(target)
        return targets

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

    def _province_insight(self, weakest: list[IndicatorInsight]) -> ProvinceInsight:
        data_path_candidates = [
            self.subnational_data_path,
            Path("data/subnational/sdg16_provinces.csv"),
            Path("data/subnational/sdg16_provinces_demo.csv"),
        ]
        data_path = next((path for path in data_path_candidates if path.exists()), None)
        if data_path is None:
            return ProvinceInsight(
                data_status="missing_provincial_dataset",
                message=(
                    "Chưa có dataset PAPI/PCI cấp tỉnh trong project, nên hệ thống chưa kết luận "
                    "tỉnh nào tệ nhất. Khi thêm dữ liệu tỉnh, phase drill-down có thể map top "
                    "chỉ số yếu sang province-level FE."
                ),
            )

        rows = []
        with data_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for raw in reader:
                row = {str(key).strip().lower(): value for key, value in raw.items()}
                province = row.get("province") or row.get("tinh") or row.get("province_name")
                if not province:
                    continue
                try:
                    year = int(float(row.get("year", 0)))
                except ValueError:
                    continue
                score = self._first_float(row, ["goal16", "papi_score", "pci_score"])
                if score is None:
                    continue
                rows.append((year, province, score, row))

        if not rows:
            return ProvinceInsight(
                data_status="invalid_provincial_dataset",
                message=(
                    f"Đã tìm thấy {data_path}, nhưng chưa đọc được cột province/year/goal16 "
                    "hoặc các cột điểm PAPI/PCI thay thế."
                ),
            )

        latest_year = max(year for year, _, _, _ in rows)
        latest_rows = [row for row in rows if row[0] == latest_year]
        _, weakest_province, weakest_score, weakest_row = min(
            latest_rows, key=lambda item: item[2]
        )

        dimension_map = {
            "n_sdg16_cpi": "bribery_people",
            "n_sdg16_admin": "admin_procedure",
            "n_sdg16_justice": "vertical_accountability",
            "n_sdg16_power": "transparency",
            "n_sdg16_security": "citizen_participation",
        }
        province_weak_dims = []
        for item in weakest[:5]:
            mapped = dimension_map.get(item.feature)
            if not mapped:
                continue
            value = self._first_float(weakest_row, [mapped])
            if value is not None:
                province_weak_dims.append(f"{mapped}={value:.2f}")

        is_demo = "demo" in data_path.name or str(weakest_row.get("source", "")).lower().startswith("demo")
        status = "demo_provincial_dataset" if is_demo else "ready"
        source_note = "Dataset hiện là demo/mock, cần thay bằng PAPI/PCI thật trước khi kết luận chính thức." if is_demo else "Dataset PAPI/PCI cấp tỉnh đã sẵn sàng."
        dims_text = (
            f" Các chiều yếu liên quan: {', '.join(province_weak_dims)}."
            if province_weak_dims
            else ""
        )
        return ProvinceInsight(
            data_status=status,
            message=(
                f"{source_note} Năm {latest_year}, tỉnh yếu nhất theo goal16/PAPI-PCI là "
                f"{weakest_province} với điểm {weakest_score:.2f}.{dims_text}"
            ),
            weakest_province=weakest_province,
            weakest_score=round(weakest_score, 4),
        )

    @staticmethod
    def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
        for key in keys:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                return float(raw)
            except ValueError:
                continue
        return None

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
            f"- {item.scenario} {item.year}: score={item.predicted_score:.2f}, "
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
                f"Drill-down cấp tỉnh: {province.message} "
                "Dự báo 2024-2030 được lấy từ GRU baseline và ba kịch bản pessimistic/base/optimistic. "
                "Khuyến nghị sơ bộ: tập trung cải cách thể chế, minh bạch hóa thực thi, "
                "nâng khả năng tiếp cận tư pháp, bảo vệ quyền tài sản và gắn cải thiện chỉ số "
                "với chương trình chuyển đổi số/chất lượng dịch vụ công."
            )

        system_prompt = (
            "Bạn là trợ lý chính sách cho đề tài năng lực cạnh tranh quốc gia. "
            "Chỉ dùng số liệu mô hình, SHAP/contribution fallback, GRU forecast, drill-down tỉnh "
            "và evidence RAG được cung cấp. "
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
5. Một đoạn cảnh báo phương pháp: SHAP/contribution giải thích hành vi mô hình, không tự chứng minh quan hệ nhân quả.
""".strip()
        return self.llm_service.answer(system_prompt, user_prompt)
