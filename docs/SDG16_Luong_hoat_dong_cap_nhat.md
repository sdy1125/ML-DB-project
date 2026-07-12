# TÀI LIỆU LUỒNG HOẠT ĐỘNG SDG16 INTELLIGENCE PLATFORM

**Phiên bản cập nhật theo luồng project hiện tại**  
**Mục tiêu:** mô tả đầy đủ luồng dữ liệu, học máy, SHAP, GRU, drill-down cấp tỉnh, RAG, LLM, Web/API và Docker Compose.

---

## Mục lục

1. Tổng quan hệ thống  
2. Cấu trúc phase hiện tại  
3. Luồng dữ liệu đầu vào  
4. Phase 1 — Panel OLS + Fixed Effects  
5. Phase 2 — XGBoost prediction  
6. Phase 3 — SHAP explainability  
7. Phase 4 — Drill-down cấp tỉnh  
8. Phase 5 — GRU forecasting 2024–2030  
9. Phase 6 — RAG + LLM policy recommendation  
10. Final Insight API  
11. Luồng Web/API serving  
12. Docker Compose và cách chạy  
13. Artifact đầu ra quan trọng  
14. Checklist kiểm tra luồng hoàn chỉnh

---

## 1. Tổng quan hệ thống

SDG16 Intelligence Platform là hệ thống phân tích chỉ số SDG16 nhằm hỗ trợ đánh giá năng lực thể chế, dự báo xu hướng và sinh khuyến nghị chính sách cho Việt Nam. Project hiện được tổ chức theo hai nhóm luồng chính:

- **Offline / Batch pipeline:** chuẩn bị dữ liệu, huấn luyện mô hình, sinh artifact, tạo SHAP, dự báo GRU, phân tích tỉnh và index tài liệu RAG.
- **Online / Serving pipeline:** giao diện Vue gửi request qua Nginx/Spring Boot/FastAPI; FastAPI đọc artifact đã sinh để trả về điểm hiện tại, chỉ số yếu nhất, tỉnh yếu nhất, dự báo 2030 và khuyến nghị chính sách.

Luồng cập nhật hiện tại đã khớp với sơ đồ mục tiêu:

```text
SDG16 country data
  -> Phase 1 Panel OLS + Fixed Effects
  -> Phase 2 XGBoost prediction
  -> Phase 3 SHAP: top weak indicators
  -> Phase 5 GRU forecast 2024–2030
  -> Phase 6 RAG + LLM
  -> Final output

SDG16 provincial data
  -> Phase 4 Drill-down / Panel tỉnh
  -> Phase 6 RAG + LLM
  -> Final output
```

Output cuối cùng gồm:

- Điểm Việt Nam hiện tại.
- Chỉ số yếu nhất kéo điểm xuống.
- Tỉnh yếu nhất theo dữ liệu cấp tỉnh.
- Dự báo 2024–2030 theo ba kịch bản.
- Khuyến nghị chính sách có evidence RAG.

---

## 2. Cấu trúc phase hiện tại

```text
src/sdg16_pipeline/
├── data_ingestion/
│   └── pdf_ingestion.py
├── phase1_panel_ols/
│   ├── run_pipeline.py
│   └── spark_jobs/
│       ├── prepare_data.py
│       └── train_linear.py
├── phase2_xgboost_ablation/
│   └── xgboost_pipeline.py
├── phase3_gru_forecasting/
│   └── gru_forecast.py
├── phase4_shap_explanation/
│   ├── shap_analyzer.py
│   └── run_shap_sdg16_vietnam.py
├── phase5_subnational_drilldown/
│   └── subnational_analyzer.py
├── phase6_rag_llm_policy/
│   ├── embeddings.py
│   └── index_documents.py
└── final_output/
    └── generate_figures.py
```

Các wrapper/script chính:

```text
scripts/optimize_and_compare_models.py   # chạy tổng model + comparison
scripts/run_shap_sdg16_vietnam.py        # chạy XGBoost + SHAP cho Việt Nam
scripts/run_gru_forecast.py              # chạy GRU forecast 2024–2030
scripts/train_xgboost.py                 # chạy optional XGBoost phase 2
scripts/run_subnational.py               # chạy drill-down cấp tỉnh
scripts/evaluate_models.py               # in bảng so sánh model
```

---

## 3. Luồng dữ liệu đầu vào

Project hiện dùng hai nhóm dữ liệu chính:

### 3.1. Dữ liệu quốc gia

```text
data/clean/sdg16_spark.csv
```

Vai trò:

- Là nguồn chính cho XGBoost, SHAP, GRU và final insight.
- Gồm dữ liệu SDR2024 đã làm sạch.
- Có các cột `country`, `year`, `goal16` và các feature dạng `n_sdg16_*`.

Artifact liên quan:

```text
artifacts/panel_ols/panel_ols_results.json
artifacts/linear_regression/metadata.json
artifacts/shap/shap_summary.json
artifacts/shap/gap_analysis.csv
artifacts/gru/vietnam_forecast_2024_2030.csv
```

### 3.2. Dữ liệu cấp tỉnh

```text
data/subnational/sdg16_provinces.csv
```

Vai trò:

- Đại diện cho dữ liệu drill-down cấp tỉnh.
- Có các cột như `province`, `year`, `goal16`, `papi_score`, `pci_transparency`, `grdp_index`.
- Được dùng để xác định tỉnh yếu nhất và các chiều PAPI/PCI liên quan.

Artifact liên quan:

```text
artifacts/subnational/subnational_results.json
figures/provincial_heatmap.png
```

### 3.3. Dữ liệu tri thức RAG

```text
data/knowledge/text/
data/knowledge/processed/chunks.jsonl
```

Vai trò:

- Chứa tài liệu PDF đã parse thành text/chunk.
- Chia thành 5 nhóm tri thức: IMD/WEF, Vietnam country reports, case studies, academic papers, Vietnam policies.
- Được dùng cho search evidence và sinh khuyến nghị chính sách.

---

## 4. Phase 1 — Panel OLS + Fixed Effects

Phase 1 chính thức của project hiện tại là **Panel OLS + Fixed Effects**. Mô hình này học quan hệ giữa điểm `goal16` và các chỉ số `n_sdg16_*` trên dữ liệu panel quốc gia - năm, đồng thời kiểm soát hiệu ứng cố định theo quốc gia và theo năm.

Entrypoint:

```powershell
python scripts\run_panel_ols.py
```

Mô hình:

```text
Goal16_it = alpha + beta * SDG16_indicators_it + country_FE_i + year_FE_t + error_it
```

Output chính:

```text
artifacts/panel_ols/panel_ols_results.json
```

Artifact Panel OLS gồm:

- `model_type = panel_ols_fixed_effects`
- `formula`
- `features`
- `models.entity_fe`
- `models.both_fe`
- `models.pooled`
- `comparison`
- `significant_factors_p05`

Kết quả hiện tại:

```text
Entity + Time FE R² within  = 0.6297
Entity + Time FE R² overall = 0.6263
Pooled OLS R² overall       = 0.7122
Số feature SDG16            = 17
```

Vai trò hiện tại:

- Là baseline kinh tế lượng chính của Phase 1.
- Học trọng số/quan hệ tuyến tính có kiểm soát country FE và year FE.
- Là cơ sở để so sánh với XGBoost/GRU và giải thích định lượng trong paper.
- Không phải nguồn chính cho `current_score` của final insight; final insight ưu tiên XGBoost/SHAP.

Legacy artifact vẫn còn:

```text
artifacts/linear_regression/metadata.json
```

File này được giữ lại như fallback online cũ nếu thiếu SHAP/XGBoost artifact, không còn là Phase 1 chính.

---

## 5. Phase 2 — XGBoost prediction

Entrypoint:

```powershell
python scripts/train_xgboost.py
```

Nguồn dữ liệu:

```text
data/clean/sdg16_spark.csv
```

Vai trò:

- Dự đoán điểm `goal16`.
- Là mô hình phi tuyến để so sánh với Linear Regression.
- Có pipeline optional để tuning và ablation.

Artifact:

```text
artifacts/xgboost/metadata.json
artifacts/xgboost/xgboost_model.pkl
artifacts/xgboost/scaler.pkl
```

Kết quả so sánh hiện tại:

```text
Optional XGBoost Tuned Pipeline:
RMSE = 7.7773
MAE  = 4.6659
R²   = 0.6974
```

Trong project hiện tại, mô hình XGBoost tốt nhất cho output cuối nằm ở runner SHAP/XGBoost của Phase 3.

---

## 6. Phase 3 — SHAP explainability

Entrypoint:

```powershell
python scripts/run_shap_sdg16_vietnam.py
```

Vai trò:

- Huấn luyện/tune XGBoost tốt nhất cho SDG16.
- Sinh SHAP-style contribution bằng XGBoost `pred_contribs`.
- Xác định top chỉ số kéo điểm Việt Nam xuống.
- Tạo artifact chính cho final insight.

Artifact chính:

```text
artifacts/shap/shap_summary.json
artifacts/shap/gap_analysis.csv
artifacts/shap/global_shap_importance.csv
artifacts/shap/xgboost_tuning_results.csv
```

Kết quả model hiện tại:

```text
XGBoost SHAP Runner:
RMSE = 1.8249
MAE  = 1.4019
R²   = 0.9859
```

Điểm Việt Nam hiện tại theo XGBoost/SHAP:

```text
Observed score 2023  = 63.7269
Predicted score 2023 = 64.0963
```

Top chỉ số kéo điểm Việt Nam xuống:

| Feature | Ý nghĩa | Contribution |
|---|---|---:|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2.3477 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1.9228 |
| `n_sdg16_exprop` | Bảo vệ quyền tài sản / chống tịch thu tài sản | -1.3962 |
| `n_sdg16_clabor` | Lao động trẻ em | -0.5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0.3568 |

Trong final insight hiện tại:

```text
explainability_source = shap_gap_analysis_csv+xgboost_shap_summary
model_version = xgboost_shap_runner+gru_forecast+subnational_drilldown
```

---

## 7. Phase 4 — Drill-down cấp tỉnh

Entrypoint:

```powershell
python scripts/run_subnational.py
```

Nguồn dữ liệu:

```text
data/subnational/sdg16_provinces.csv
```

Vai trò:

- Map các chỉ số yếu cấp quốc gia sang chiều PAPI/PCI cấp tỉnh.
- Tính ranking tỉnh theo `goal16`.
- Chạy phân tích Panel FE cấp tỉnh bằng `linearmodels.PanelOLS`.
- Xuất heatmap và kết quả drill-down.

Artifact:

```text
artifacts/subnational/subnational_results.json
figures/provincial_heatmap.png
```

Mapping hiện tại:

| SDG16 feature | Cột cấp tỉnh |
|---|---|
| `n_sdg16_cpi` | `bribery_people` |
| `n_sdg16_admin` | `admin_procedure` |
| `n_sdg16_justice` | `vertical_accountability` |
| `n_sdg16_power` | `transparency` |
| `n_sdg16_security` | `citizen_participation` |

Kết quả serving hiện tại:

```text
province.data_status = ready
weakest_province     = Hải Dương
weakest_score        = 0.3772
```

Kết quả Panel FE tỉnh hiện tại:

```text
estimator  = linearmodels_panel_ols_entity_fe
nobs       = 315
R² overall = 0.0032
```

Nghĩa là hệ thống hiện đã có thể trả lời phần “tỉnh nào tệ nhất” trong output cuối, đồng thời có phân tích Panel FE tỉnh để phục vụ nhánh drill-down trong sơ đồ.

---

## 8. Phase 5 — GRU forecasting 2024–2030

Entrypoint:

```powershell
python scripts/run_gru_forecast.py
```

Nguồn dữ liệu:

```text
data/clean/sdg16_spark.csv
```

Vai trò:

- Dự báo `goal16` theo chuỗi thời gian.
- Dùng sequence length = 5 năm.
- Sinh forecast cho Việt Nam giai đoạn 2024–2030.

Artifact:

```text
artifacts/gru/gru_summary.json
artifacts/gru/vietnam_forecast_2024_2030.csv
artifacts/gru/gru_model_state.pt
```

Kết quả model:

```text
GRU Sequence Forecaster:
RMSE = 2.4411
MAE  = 1.9119
R²   = 0.9747
```

Dự báo baseline cho Việt Nam:

| Năm | Predicted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.6389 |
| 2026 | 63.6521 |
| 2027 | 63.6432 |
| 2028 | 63.6419 |
| 2029 | 63.6419 |
| 2030 | 63.6419 |

Trong final insight, hệ thống tạo thêm 3 kịch bản:

```text
pessimistic
base
optimistic
```

Tổng số forecast trả về:

```text
3 kịch bản × 7 năm = 21 dòng forecast
```

---

## 9. Phase 6 — RAG + LLM policy recommendation

Luồng RAG gồm hai phần:

### 9.1. Offline parse/index

Parse PDF:

```powershell
docker compose --profile jobs run --rm pdf-parser
```

Index documents:

```powershell
docker compose --profile jobs run --rm rag-indexer
```

Nguồn text/chunk:

```text
data/knowledge/text/
data/knowledge/processed/chunks.jsonl
```

### 9.2. Online recommendation

Final insight sẽ tạo query từ:

- Quốc gia.
- Top chỉ số yếu từ SHAP.
- Nhóm policy keyword: governance, justice, anti-corruption, digital government.

Sau đó:

```text
RAG search -> evidence -> LLM prompt -> recommendation
```

LLM provider hỗ trợ:

```text
disabled
openai
openai-compatible
ollama
```

Khi `LLM_PROVIDER=disabled`, hệ thống trả fallback recommendation, không gọi model ngoài.

---

## 10. Final Insight API

Endpoint chính:

```http
GET /insights/final?country=Vietnam&use_llm=true
```

Qua Spring Boot:

```http
GET /api/v1/insights/final?country=Vietnam&useLlm=true
```

Output schema chính:

```text
country
year
current_score
observed_score
model_version
explainability_source
weakest_indicators
strongest_indicators
forecasts
province
evidence
recommendation
provider
```

Luồng xử lý hiện tại:

```text
1. Đọc feature mới nhất của Việt Nam từ data/clean/sdg16_spark.csv.
2. Chạy ModelService linear để có fallback.
3. Nếu có artifacts/shap/shap_summary.json:
   - dùng predicted score từ XGBoost/SHAP làm current_score.
4. Nếu có artifacts/shap/gap_analysis.csv:
   - dùng SHAP gap rows làm weakest_indicators.
5. Nếu có artifacts/gru/vietnam_forecast_2024_2030.csv:
   - dùng GRU forecast làm baseline 2024–2030.
   - sinh thêm 3 kịch bản pessimistic/base/optimistic.
6. Nếu có data/subnational/sdg16_provinces.csv:
   - xác định tỉnh yếu nhất.
7. Search RAG evidence theo top chỉ số yếu.
8. Gọi LLM nếu bật, hoặc trả fallback text nếu disabled.
```

Kết quả kiểm tra hiện tại:

```text
model_version = xgboost_shap_runner+gru_forecast+subnational_drilldown
source        = shap_gap_analysis_csv+xgboost_shap_summary
year          = 2023
current_score = 64.0963
observed_score = 63.7269
top weak      = n_sdg16_rsf, n_sdg16_justice, n_sdg16_exprop
province      = Hải Dương, score = 0.3772
forecast_count = 21
```

---

## 11. Luồng Web/API serving

Project hiện có stack web/API:

```text
Vue frontend
  -> Nginx
  -> Spring Boot backend
  -> FastAPI ml-api
```

Hoặc frontend có thể gọi thẳng FastAPI qua prefix `/ml`.

Route chính:

```text
/api/      -> Spring Boot backend
/ml/       -> FastAPI ml-api
/actuator/ -> Spring Boot actuator
```

FastAPI endpoint:

```text
GET  /health
GET  /model/info
POST /predict
POST /explain
POST /search
POST /ask
GET  /insights/final
```

Spring Boot public endpoint:

```text
GET  /api/v1/health
GET  /api/v1/model
POST /api/v1/predictions
POST /api/v1/explanations
POST /api/v1/knowledge/search
POST /api/v1/assistant/questions
GET  /api/v1/insights/final
```

Vai trò:

- Vue hiển thị mô phỏng điểm, giải thích, recommendation và final insight.
- Spring Boot đóng vai trò backend/proxy API public.
- FastAPI là ML/RAG service chính.

---

## 12. Docker Compose và cách chạy

Chạy API/web:

```powershell
$env:BACKEND_PORT='8081'
docker compose up -d --build ml-api backend frontend
```

Kiểm tra FastAPI:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Kiểm tra final insight:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/insights/final?country=Vietnam&use_llm=false"
```

Nếu frontend chạy qua Nginx:

```text
http://localhost:3000
```

---

## 13. Artifact đầu ra quan trọng

### 13.1. Model comparison

```text
artifacts/model_comparison/model_comparison.csv
artifacts/model_comparison/model_comparison.json
artifacts/model_comparison/best_model.json
```

Best model hiện tại:

```text
XGBoost SHAP Runner
```

### 13.2. SHAP

```text
artifacts/shap/shap_summary.json
artifacts/shap/gap_analysis.csv
artifacts/shap/global_shap_importance.csv
artifacts/shap/xgboost_tuning_results.csv
```

### 13.3. GRU

```text
artifacts/gru/gru_summary.json
artifacts/gru/vietnam_forecast_2024_2030.csv
```

### 13.4. Subnational

```text
artifacts/subnational/subnational_results.json
figures/provincial_heatmap.png
```

### 13.5. RAG

```text
data/knowledge/text/
data/knowledge/processed/chunks.jsonl
data/knowledge/processed/parse_report.json
```

---

## 14. Checklist kiểm tra luồng hoàn chỉnh

Chạy toàn bộ model pipeline:

```powershell
python scripts\optimize_and_compare_models.py
```

Pipeline này hiện chạy:

```text
1. Run Phase-1 Panel OLS + Fixed Effects
2. Tune/evaluate XGBoost + SHAP runner
3. Tune/evaluate GRU sequence forecaster
4. Train optional Phase-2 XGBoost pipeline
5. Run Phase-5 subnational drill-down
6. Print final model comparison
```

Chạy test:

```powershell
python -m pytest tests -q
```

Kết quả mong đợi:

```text
7 passed
```

Kiểm tra Phase 1 riêng:

```powershell
python scripts\run_panel_ols.py
```

Kết quả mong đợi:

```text
Entity+Time FE R² within: 0.6297
Entity+Time FE R² overall: 0.6263
Artifacts written to artifacts/panel_ols/panel_ols_results.json
```

Kiểm tra Phase 4 riêng:

```powershell
python scripts\run_subnational.py
```

Kết quả mong đợi:

```text
Panel observations: 315
Mapped dimensions: ['CPI', 'Admin', 'Justice', 'Power', 'Security']
estimator: linearmodels_panel_ols_entity_fe
```

Kiểm tra final insight offline:

```text
current_score lấy từ XGBoost/SHAP
weakest_indicators lấy từ SHAP gap analysis
forecasts lấy từ GRU 2024–2030 và sinh 3 kịch bản
province trả tỉnh yếu nhất từ sdg16_provinces.csv
recommendation dùng RAG/LLM hoặc fallback
```

---

## 15. Kết luận luồng hiện tại

Project hiện đã khớp với luồng mục tiêu ở mức hệ thống:

```text
Phase 1 Panel OLS + Fixed Effects
-> Phase 2 XGBoost
-> Phase 3 SHAP
-> Phase 4 Drill-down tỉnh
-> Phase 5 GRU forecast
-> Phase 6 RAG + LLM
-> Final output
```

Phase 1 hiện đã dùng đúng Panel OLS + Fixed Effects. Spark Linear Regression chỉ còn là legacy fallback metadata, không phải luồng chính của Phase 1.
