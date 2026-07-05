# Kiến trúc hệ thống

## Luồng online và offline

```text
                         OFFLINE / BATCH
SDSN CSV ──> raw lake ──> Spark prepare ──> Parquet
                                             │
                                             v
                                      Spark MLlib train
                                             │
                              ┌──────────────┴──────────────┐
                              v                             v
                       Spark Pipeline                 metadata.json
                       (batch scoring)          (coefficients/scaler/metrics)

Reports/PDF ──> PDF parser ──> chunks ──> embedding ──> Qdrant

                            ONLINE / SERVING
User ──> Vue/Nginx ──> Spring Boot BFF ──> FastAPI ML ──┬─> prediction
                                                       ├─> explainability
                                                       ├─> Qdrant retrieval
                                                       └─> LLM adapter
```

## Ranh giới trách nhiệm

### Data layer

`data/raw` là immutable input; `data/clean` chỉ do Spark pipeline sinh ra.
Schema, nguồn, license và thời điểm tải phải được ghi trong data catalog.
MinIO được để dưới profile riêng vì dataset hiện tại khoảng 10 nghìn dòng,
chưa bắt buộc phải vận hành HDFS cluster.

### Web application layer

Vue là SPA giao tiếp duy nhất với Spring Boot qua `/api/v1`. Nginx phục vụ
static assets và reverse-proxy API, do đó production không cần cấu hình CORS.
Spring Boot chịu trách nhiệm validation, public API contract, error mapping và
là vị trí để bổ sung authentication, authorization, audit log hoặc database.

FastAPI trở thành dịch vụ ML nội bộ; frontend không gọi trực tiếp service này.

### ML layer

Artifact online gồm `metadata.json`, đủ để backend tái tạo dự đoán linear mà
không phải nhúng Java/Spark vào API. Spark PipelineModel vẫn được giữ để batch
scoring và audit.

Baseline hiện thực:

1. Temporal split.
2. Median imputation fit trên train.
3. StandardScaler.
4. Linear Regression.
5. RMSE, MAE, R² trên validation và test.
6. Export hệ số, mean, standard deviation và split metadata.

Các model Panel Fixed Effects, XGBoost/LightGBM và GRU thuộc phase tiếp theo.
Chúng cần dùng chung data contract và model registry metadata để API không phụ
thuộc vào framework.

### Explainability layer

Với linear model, contribution cục bộ được tính:

`contribution_i = coefficient_i × standardized_value_i`

Đây là phép phân rã chính xác của linear prediction và phù hợp hơn việc kéo
SHAP vào baseline chỉ để có tên gọi. SHAP sẽ được thêm cho tree/deep models.

### RAG/LLM layer

LLM chỉ nhận:

- predicted score;
- top contribution;
- gap/scenario đã được code tính;
- đoạn tài liệu truy xuất từ Qdrant.

Prompt yêu cầu phân biệt số liệu, suy luận và giả định. Câu trả lời phải lưu
model version, prompt version và nguồn retrieval nếu dùng cho nghiên cứu.

## Data contract

Một observation biểu diễn một `country-year`.

```text
country: string
year: integer
goal16: double
n_sdg16_*: double | null
```

Khóa logic: `(country, year)`. Nếu nhiều nguồn có cùng khóa, phase Data phải
xác định source priority; không được âm thầm `dropDuplicates` trong bản
production mà thiếu lineage.

## Bảo mật và vận hành

- Không commit API key; dùng `.env` hoặc secret manager.
- API key chỉ nằm ở FastAPI ML service, không truyền xuống Spring Boot/Vue/browser.
- Artifact và dữ liệu clean mount read-only vào API.
- Production cần thêm authentication, rate limit, request log, model registry,
  CI/CD và object-storage backup.
