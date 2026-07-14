# Tài liệu luồng hoạt động SDG16 Intelligence Platform

Phiên bản cập nhật theo project hiện tại.

## 1. Mục tiêu hệ thống

Project xây dựng một pipeline phân tích SDG16 cho Việt Nam, gồm dữ liệu quốc gia, mô hình kinh tế lượng, XGBoost, giải thích đóng góp chỉ số, drill-down cấp tỉnh, dự báo GRU, RAG + LLM và giao diện Web/API.

Điểm cần ghi đúng trong báo cáo: XGBoost hiện không nên được mô tả là mô hình “dự đoán độc lập” hay “chứng minh nhân quả”. Vì `goal16` là điểm tổng hợp được xây dựng từ các chỉ số `n_sdg16_*`, XGBoost đang làm nhiệm vụ tái dựng điểm tổng hợp và phân rã đóng góp chỉ số.

## 2. Luồng tổng thể đúng với code

```text
data/clean/sdg16_spark.csv
  -> Phase 1: Panel OLS + Fixed Effects
  -> Phase 2: XGBoost composite-score reconstruction
  -> Phase 3: XGBoost tree-contribution explanation
  -> Phase 5: GRU forecast 2024-2030
  -> Phase 6: RAG + LLM policy recommendation
  -> Final output

data/subnational/sdg16_provinces.csv
  -> Phase 4: Subnational drill-down / province Panel FE
  -> Phase 6: RAG + LLM policy recommendation
  -> Final output
```

Output cuối cùng gồm:

- Điểm Việt Nam hiện tại.
- Các chỉ số kéo điểm Việt Nam xuống.
- Tỉnh yếu nhất theo dữ liệu cấp tỉnh hiện có.
- Dự báo 2024-2030.
- Khuyến nghị chính sách có evidence từ RAG.

## 3. Entrypoint chính

| Mục đích | File chạy |
|---|---|
| Chạy toàn bộ pipeline model + so sánh | `python scripts\optimize_and_compare_models.py` |
| Phase 1 Panel OLS | `python scripts\run_panel_ols.py` |
| Phase 2 optional XGBoost | `python scripts\train_xgboost.py` |
| Phase 3 XGBoost reconstruction + contribution | `python scripts\run_shap_sdg16_vietnam.py` |
| Phase 4 drill-down cấp tỉnh | `python scripts\run_subnational.py` |
| Phase 5 GRU forecast | `python scripts\run_gru_forecast.py` |
| Tổng hợp bảng so sánh model | `python scripts\evaluate_models.py` |

## 4. Phase 1 — Panel OLS + Fixed Effects

Phase 1 chính thức là Panel OLS + Fixed Effects, dùng `linearmodels.PanelOLS`.

Entrypoint:

```powershell
python scripts\run_panel_ols.py
```

Input:

```text
data/clean/sdg16_spark.csv
```

Mô hình:

```text
goal16_it = alpha + beta * sdg16_indicators_it + country_FE_i + year_FE_t + error_it
```

Artifact:

```text
artifacts/panel_ols/panel_ols_results.json
```

Kết quả hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| RMSE | 2.1276 |
| MAE | 1.6099 |
| R2 overall | 0.6263 |

Vai trò:

- Baseline kinh tế lượng chính.
- Có kiểm soát country fixed effects và year fixed effects.
- Có diagnostics như RMSE, MAE, Durbin-Watson, VIF.
- Không phải nguồn chính của `current_score` trong Final Insight; Final Insight ưu tiên artifact XGBoost/contribution.

## 5. Phase 2 — XGBoost composite-score reconstruction

Entrypoint:

```powershell
python scripts\train_xgboost.py
```

Input:

```text
data/clean/sdg16_spark.csv
```

Vai trò:

- Tái dựng/ước lượng điểm tổng hợp `goal16` từ các chỉ số thành phần `n_sdg16_*`.
- Là mô hình phi tuyến để so sánh với Panel OLS.
- Có pipeline optional để tuning/ablation.

Artifact:

```text
artifacts/xgboost/metadata.json
artifacts/xgboost/xgboost_model.pkl
artifacts/xgboost/scaler.pkl
```

Kết quả optional XGBoost hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| RMSE | 7.7773 |
| MAE | 4.6659 |
| R2 | 0.6974 |

Lưu ý: Phase 2 này là optional XGBoost pipeline. XGBoost đang chạy chính cho output cuối là runner ở Phase 3.

## 6. Phase 3 — XGBoost tree-contribution explanation

Entrypoint chính:

```powershell
python scripts\run_shap_sdg16_vietnam.py
```

File xử lý chính:

```text
src/sdg16_pipeline/phase4_shap_explanation/run_shap_sdg16_vietnam.py
```

Vai trò:

- Tune/train XGBoost chính.
- Dùng XGBoost `pred_contribs=True` để sinh tree contribution.
- Xuất top chỉ số kéo điểm Việt Nam xuống.
- Xuất leakage/correlation report.
- Là nguồn artifact chính cho Final Insight.

Artifact:

```text
artifacts/shap/shap_summary.json
artifacts/shap/gap_analysis.csv
artifacts/shap/global_shap_importance.csv
artifacts/shap/leakage_correlation_report.csv
```

Kết quả hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| RMSE | 1.8249 |
| MAE | 1.4019 |
| R2 | 0.9859 |

Cách diễn giải đúng:

- Đây là hiệu suất tái dựng điểm tổng hợp `goal16`.
- Không được viết là bằng chứng nhân quả.
- Không được viết là dự đoán độc lập về chất lượng quản trị.
- Contribution dùng để ưu tiên chỉ số cần chú ý, không phải tác động chính sách chắc chắn.

Top chỉ số kéo điểm Việt Nam xuống:

| Feature | Ý nghĩa | Contribution |
|---|---|---:|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2.3477 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1.9228 |
| `n_sdg16_exprop` | Bảo vệ quyền tài sản / chống tịch thu | -1.3962 |
| `n_sdg16_clabor` | Lao động trẻ em | -0.5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0.3568 |

## 7. Phase 4 — Drill-down cấp tỉnh

Entrypoint:

```powershell
python scripts\run_subnational.py
```

Input:

```text
data/subnational/sdg16_provinces.csv
```

Vai trò:

- Map chỉ số yếu cấp quốc gia sang proxy PAPI/PCI cấp tỉnh.
- Tính ranking tỉnh theo `goal16`.
- Chạy Panel FE cấp tỉnh.
- Trả về tỉnh yếu nhất cho Final Insight.

Artifact:

```text
artifacts/subnational/subnational_results.json
```

Leakage đã được giảm bằng cách dùng biến trễ:

```text
papi_score_lag1
pci_transparency_lag1
grdp_index_lag1
```

Kết quả Panel FE tỉnh hiện tại có R2 thấp, nên chỉ nên dùng như lớp drill-down hỗ trợ, không nên xem là ranking chính thức nếu chưa validate dữ liệu PAPI/PCI.

## 8. Phase 5 — GRU forecasting 2024-2030

Entrypoint:

```powershell
python scripts\run_gru_forecast.py
```

Input:

```text
data/clean/sdg16_spark.csv
```

Artifact:

```text
artifacts/gru/gru_summary.json
artifacts/gru/vietnam_forecast_2024_2030.csv
artifacts/gru/gru_model_state.pt
```

Kết quả model:

| Chỉ số | Giá trị |
|---|---:|
| RMSE | 2.4411 |
| MAE | 1.9119 |
| R2 | 0.9747 |

Dự báo baseline Việt Nam hiện tại:

| Năm | Predicted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.8785 |
| 2026 | 64.1630 |
| 2027 | 64.4077 |
| 2028 | 64.6922 |
| 2029 | 64.9520 |
| 2030 | 65.1743 |

Điểm cần sửa trong báo cáo: không viết forecast “phẳng quanh 63.64” nữa. Artifact hiện tại cho thấy baseline tăng nhẹ đến 65.1743 năm 2030.

## 9. Phase 6 — RAG + LLM policy recommendation

Luồng RAG gồm:

```text
data/knowledge/text/
  -> chunking / indexing
  -> retrieval evidence
  -> LLM recommendation
```

Provider LLM hiện có thể cấu hình qua `.env`, ví dụ Ollama hoặc Gemini qua openai-compatible endpoint. Báo cáo không nên ghi cứng GPT-4/Claude nếu code không cố định model đó.

Final Insight dùng:

- current score từ XGBoost/contribution artifact.
- weakest indicators từ `gap_analysis.csv`.
- GRU forecast baseline.
- province insight từ subnational artifact.
- evidence từ RAG.
- LLM để viết khuyến nghị cuối.

## 10. Web/API serving

Luồng serving:

```text
Vue frontend
  -> Nginx/Spring Boot proxy
  -> FastAPI ML service
  -> artifacts + RAG + LLM
  -> final insight JSON
```

Các endpoint quan trọng:

| Mục đích | Endpoint |
|---|---|
| Health/model status | `/health`, `/model` |
| Explain input features | `/explain` |
| Search RAG evidence | `/search` |
| Ask LLM | `/ask` |
| Final insight | `/ml/insights/final` |

UI hiện hiển thị thêm:

- Panel OLS diagnostics.
- XGBoost leakage/circular target warning.
- GRU forecast baseline mới.
- Khuyến nghị cuối.

## 11. Artifact đầu ra quan trọng

| Phase | Artifact |
|---|---|
| Panel OLS | `artifacts/panel_ols/panel_ols_results.json` |
| XGBoost optional | `artifacts/xgboost/metadata.json` |
| XGBoost contribution | `artifacts/shap/shap_summary.json` |
| Leakage report | `artifacts/shap/leakage_correlation_report.csv` |
| GRU forecast | `artifacts/gru/vietnam_forecast_2024_2030.csv` |
| Subnational | `artifacts/subnational/subnational_results.json` |
| Model comparison | `artifacts/model_comparison/model_comparison.csv` |

## 12. Checklist báo cáo cho đúng project

- Phase 1 phải là Panel OLS + Fixed Effects.
- XGBoost chính nằm ở `scripts/run_shap_sdg16_vietnam.py`.
- XGBoost phải được gọi là composite-score reconstruction, không claim causal prediction.
- GRU forecast phải dùng bảng 2024-2030 tăng từ 63.6456 lên 65.1743.
- Subnational chỉ là drill-down hỗ trợ nếu dữ liệu tỉnh chưa được validate.
- RAG + LLM là prototype recommendation engine, chưa claim expert validation nếu chưa có module đánh giá chuyên gia.
