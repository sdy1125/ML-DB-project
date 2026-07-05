# Workflow triển khai đầy đủ

## Phase 0 — Chốt bài toán và data contract

Đầu ra:

- Chọn target chính thức: SDSN `goal16` hay IMD/WEF score.
- Chốt phạm vi quốc gia, năm và định nghĩa 17 indicator.
- Data dictionary ghi đơn vị, chiều tốt/xấu, missing semantics, nguồn và license.
- Tiêu chí nghiên cứu: dự đoán, reverse-engineer hệ số, hay causal inference.

Definition of Done:

- Một schema duy nhất được Data, ML và Backend ký xác nhận.
- Không trộn SDG16 với IMD trong cùng một target mà chưa có mapping học thuật.

## Phase 1 — Data ingestion và quality

Công việc:

1. Tải ba bộ Backdated SDSN và lưu nguyên bản.
2. Tạo manifest gồm URL, checksum, ngày tải, version.
3. Chuẩn hóa tên quốc gia bằng ISO3; chuẩn hóa năm và column name.
4. Xử lý duplicate `(country, year)` theo source priority.
5. Sinh báo cáo missing ratio, range, outlier và schema drift.
6. Ghi Parquet partition theo năm hoặc source nếu dữ liệu tăng lớn.

Definition of Done:

- Pipeline idempotent: chạy lại cho cùng input tạo cùng output.
- Có quality report và test cho 5 indicator giao nhau.
- Không còn target null trong training dataset.

## Phase 2 — Baseline econometrics và ML

Nhánh A:

- Pooled OLS.
- Country fixed effects + year fixed effects.
- Hausman test FE/RE.
- Robust/clustered standard errors.
- Breusch–Pagan, Durbin–Watson và multicollinearity/VIF.

Nhánh B:

- Spark Linear, Ridge, Lasso, Elastic Net.
- XGBoost/LightGBM với lag-1, lag-2, rolling mean và delta.
- Temporal/country-aware cross validation.

Definition of Done:

- Cùng một frozen test set.
- Báo cáo RMSE, MAE, R² và uncertainty.
- Hệ số/importance ổn định qua seed, time window và imputation strategy.
- Có ablation study thay vì chỉ chọn model tốt nhất.

## Phase 3 — Forecast và scenario

Công việc:

- So sánh naive, linear trend, XGBoost autoregressive, GRU và LSTM.
- Backtesting rolling-origin trước khi forecast 2025–2030.
- Xây pessimistic/base/optimistic bằng assumptions có version.
- Scenario engine thay đổi từng indicator trong range hợp lệ.

Lưu ý: “Tăng R&D 1% làm score tăng 2.3” chỉ được gọi là tác động nhân quả khi
thiết kế nghiên cứu hỗ trợ causal claim. Nếu không, phải ghi là model-based
scenario/association.

## Phase 4 — Explainability

Công việc:

- Linear coefficient/contribution.
- Tree SHAP cho XGBoost/LightGBM.
- Deep SHAP hoặc Integrated Gradients cho GRU.
- Waterfall Việt Nam, beeswarm toàn cầu, ASEAN comparison.
- Cross-check dấu và ranking giữa FE coefficients và SHAP.

Definition of Done:

- Mọi chart ghi model version, dataset version và evaluation split.
- Có stability analysis cho top-k indicator.

## Phase 5 — RAG và policy engine

Công việc:

1. Parse PDF bằng pypdf mặc định; chỉ bật Unstructured/OCR khi cần xử lý PDF scan ảnh.
2. Chunk và embedding multilingual.
3. Index Qdrant.
4. Tạo prompt schema nhận dữ liệu ML dạng JSON.
5. Bắt buộc citation và không cho LLM tự tính score.
6. Tạo bộ câu hỏi đánh giá retrieval và factuality.
7. Expert review bởi 2–3 chuyên gia.

Definition of Done:

- Retrieval có Recall@k/MRR trên bộ câu hỏi chuẩn.
- Khuyến nghị liên kết được với contribution và nguồn tài liệu.
- Không xuất con số không có trong structured model result.

## Phase 6 — Product và deployment

Công việc:

- Hoàn thiện FastAPI ML contract và Spring Boot public API.
- Vue dashboard: country/year, forecast, explain, scenario và chat.
- Authentication, audit log, observability.
- CI: lint, unit test, container build, integration smoke test.
- CD lên VM/cloud và backup Qdrant/object storage.

## Nhịp làm việc đề xuất

- Mỗi feature đi qua: issue → branch → pull request → review → test → merge.
- Dataset và model phải version hóa độc lập với source code.
- Demo cuối mỗi tuần bằng artifact thật, không hard-code output.
- Mỗi experiment ghi config, metric, seed, commit và dataset version.
