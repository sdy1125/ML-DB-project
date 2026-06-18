# Ma trận phân công đề xuất

Mô hình phù hợp cho nhóm 5–7 người. Nếu nhóm ít hơn, có thể gộp Data với ML
và gộp Backend với DevOps.

| Workstream | Owner | Deliverable chính | Phụ thuộc |
|---|---|---|---|
| Research/Product | Thành viên 1 | research question, data dictionary, acceptance criteria, paper outline | tất cả |
| Data Engineering | Thành viên 2 | ingestion, source manifest, Spark cleaning, Parquet, quality report | Research |
| Econometrics | Thành viên 3 | Panel OLS/FE/RE, tests, coefficient table, robustness | Clean data |
| ML/Forecast | Thành viên 4 | Ridge/Lasso/XGBoost, GRU/LSTM, backtest, scenario | Clean data |
| XAI/RAG | Thành viên 5 | SHAP, document parsing, embedding, Qdrant evaluation | Models, reports |
| Backend/LLM | Thành viên 6 | Spring Boot BFF, FastAPI ML, prompt policy, API tests | Model contract, RAG |
| UI/DevOps/QA | Thành viên 7 | Vue, Nginx, Compose, CI/CD, monitoring, E2E test | APIs |

## Sprint 1 — Nền móng

- Research: chốt target và indicator dictionary.
- Data: đưa raw CSV thật vào pipeline, xử lý duplicate/source priority.
- Backend: chạy health API và thống nhất request/response schema.
- DevOps: xác nhận Compose chạy trên máy của mọi thành viên.

Gate: `raw CSV → clean Parquet` chạy lặp lại được.

## Sprint 2 — Baseline có thể kiểm chứng

- Econometrics: pooled OLS và fixed effects.
- ML: Spark Linear/Ridge/Lasso.
- Backend: load artifact, predict, explain.
- UI: dashboard nhập feature và hiển thị contribution.

Gate: artifact thật chạy end-to-end trên UI, có test metric và model version.

## Sprint 3 — Model nâng cao và XAI

- Data: feature lag/rolling/delta, leakage tests.
- ML: XGBoost/LightGBM, GRU/LSTM backtesting.
- XAI: SHAP và stability analysis.
- Research: viết methodology và ablation.

Gate: bảng so sánh model trên cùng test set.

## Sprint 4 — RAG và khuyến nghị

- XAI/RAG: parse/index tài liệu, retrieval benchmark.
- Backend/LLM: structured prompt, citation, guardrails.
- Research: expert review form và rubric.
- UI: chat có evidence và scenario.

Gate: mọi khuyến nghị có model evidence hoặc document citation.

## Sprint 5 — Hoàn thiện và paper

- QA: reproducibility run từ raw data.
- DevOps: production profile, secrets, logging, backup.
- Research: results, limitations, threats to validity.
- Cả nhóm: demo, poster, slide và artifact release.

## Quy tắc giao tiếp giữa các nhóm

Data bàn giao:

- Parquet path/version;
- schema và feature list;
- quality report;
- train/validation/test cutoff.

ML bàn giao:

- `metadata.json` theo contract hiện tại;
- metrics;
- model card;
- limitations và valid input range.

RAG bàn giao:

- collection name/version;
- embedding model/dimension;
- retrieval metrics;
- source catalog.

Backend bàn giao:

- OpenAPI;
- environment variables;
- error codes;
- smoke-test command.
