# SDG16 Intelligence Platform

Bộ khung triển khai đồ án phân tích và dự đoán `Goal16`, được xây dựng từ yêu
cầu trong `1.docx`:

`SDSN data → Data Lake → PySpark → Spark MLlib → Explainability → RAG → LLM → FastAPI → Spring Boot → Vue`

Mục tiêu trước mắt là tạo một đường chạy xuyên suốt, có thể chia việc độc lập
cho nhóm. Baseline hiện tại dùng Spark MLlib Linear Regression để dự đoán
`goal16` và reverse-engineer hệ số của các chỉ số `n_sdg16_*`.

## 1. Kiến trúc

| Khối | Công nghệ | Trách nhiệm |
|---|---|---|
| Data lake | thư mục volume / MinIO profile | lưu raw, clean và artifact |
| Processing | PySpark, Spark SQL, Parquet | hợp nhất, chuẩn hóa, lọc feature |
| ML baseline | Spark MLlib | imputation, scaling, Linear/Ridge/Lasso |
| Explainability | coefficient contribution | giải thích chiều và mức tác động |
| RAG | Qdrant + embedding adapter | truy xuất báo cáo/chính sách |
| LLM | OpenAI-compatible hoặc Ollama | diễn giải và sinh khuyến nghị |
| ML service | FastAPI | inference, explainability, RAG và LLM nội bộ |
| Web backend | Spring Boot | public API, validation, error handling và BFF |
| Frontend | Vue 3 + Nginx | dashboard web responsive và reverse proxy |
| Deployment | Docker Compose profiles | chạy từng lớp theo tài nguyên máy |

Chi tiết thiết kế: [docs/architecture.md](docs/architecture.md).

## 2. Chạy nhanh

Yêu cầu: Docker Desktop với Docker Compose v2.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Sau khi khởi động:

- Web Vue: <http://localhost:3000>
- Spring Boot API: <http://localhost:8080/api/v1/health>
- Spring Boot health: <http://localhost:8080/actuator/health>
- FastAPI ML docs: <http://localhost:8000/docs>
- Qdrant: <http://localhost:6333/dashboard>

API vẫn khởi động khi chưa có model, nhưng `/predict` và `/explain` sẽ trả
`503 Model not ready`. Đây là trạng thái có chủ đích để nhóm Data/ML và nhóm
Backend có thể làm song song.

### Chạy frontend/backend khi phát triển

Spring Boot:

```powershell
cd apps/backend
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-21.0.9.10-hotspot"
mvn spring-boot:run
```

Vue:

```powershell
cd apps/frontend
npm install
npm run dev
```

Vite chạy tại <http://localhost:5173> và proxy `/api` tới Spring Boot port
`8080`. Khi chạy Docker, Nginx phục vụ bản build tại port `3000`.

Public API do Spring Boot cung cấp:

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/v1/health` | trạng thái ML/RAG/LLM |
| GET | `/api/v1/model` | metadata và metric model |
| POST | `/api/v1/predictions` | dự đoán Goal16 |
| POST | `/api/v1/explanations` | dự đoán và contribution |
| POST | `/api/v1/knowledge/search` | tìm tài liệu RAG |
| POST | `/api/v1/assistant/questions` | tạo câu trả lời chính sách |

## 3. Chuẩn bị dữ liệu và train baseline

Đặt các file SDSN CSV vào `data/raw/`. Tối thiểu dữ liệu cần có:

- `country`
- `year`
- `goal16`
- các cột số có prefix `n_sdg16_`

Tên cột được chuẩn hóa thành chữ thường và dấu gạch dưới. Một số alias như
`Country Name`, `Goal 16`, `SDG16 Score` được tự động ánh xạ.

Khi nhiều nguồn có cùng `(country, year)`, pipeline ưu tiên filename chứa các
token trong `data.source_priority` của `configs/project.yaml`. Hãy đổi danh sách
này cho khớp tên file thật trước khi chạy.

Chạy toàn bộ pipeline bằng Spark local trong container:

```powershell
docker compose --profile jobs run --rm pipeline python -m pipelines.run_pipeline all
docker compose restart ml-api backend
```

Kết quả:

- `data/clean/sdg16.parquet`: dataset sạch;
- `artifacts/linear_regression/spark_pipeline`: Spark PipelineModel;
- `artifacts/linear_regression/metadata.json`: hệ số, scaler, metrics và version.

Muốn chạy Spark cluster:

```powershell
docker compose --profile spark up -d spark-master spark-worker
$env:SPARK_MASTER_URL="spark://spark-master:7077"
docker compose --profile jobs run --rm pipeline python -m pipelines.run_pipeline all
```

## 4. Bật RAG và LLM

Mặc định RAG/LLM bị tắt để stack lõi chạy được mà không cần API key hoặc GPU.

### Scan PDF local thành dữ liệu RAG

Bộ PDF hiện nằm trong:

```text
rag/tailieuLLM-20260626T154614Z-3-001/tailieuLLM
```

Parser sẽ tự nhận diện `Nhóm 1`–`Nhóm 5` để gắn metadata cho RAG. Chạy:

```powershell
docker compose --profile jobs run --rm pdf-parser
```

Output:

- `data/knowledge/processed/chunks.jsonl`
- `data/knowledge/processed/parse_report.json`
- `data/knowledge/text/*.txt`

Chi tiết: [docs/knowledge-ingestion.md](docs/knowledge-ingestion.md).

### OpenAI-compatible

Điền `.env`:

```dotenv
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=<embedding-model>
LLM_PROVIDER=openai-compatible
LLM_MODEL=<chat-model>
LLM_API_KEY=<secret>
LLM_BASE_URL=<provider-base-url>
```

Sau khi đã parse PDF hoặc đưa văn bản `.txt` vào `data/knowledge/text`, chạy:

```powershell
docker compose --profile jobs run --rm rag-indexer
docker compose restart ml-api backend
```

### Ollama local

```powershell
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

Sau đó đặt `LLM_PROVIDER=ollama`. Để embedding qua Ollama, đặt thêm
`EMBEDDING_PROVIDER=ollama` và chọn một embedding model đã pull.

Với demo hiện tại dùng Ollama `gemma3:4b` trên máy host và chưa cần embedding model,
đọc hướng dẫn nhanh tại [docs/rag-ollama-demo.md](docs/rag-ollama-demo.md).

## 5. Các profile Docker

| Lệnh | Thành phần |
|---|---|
| `docker compose up` | Vue + Spring Boot + FastAPI ML + Qdrant |
| `--profile jobs` | Spark pipeline, PDF parser và RAG indexer chạy theo job |
| `--profile spark` | Spark master/worker |
| `--profile data-lake` | MinIO |
| `--profile local-llm` | Ollama |

## 6. Các quyết định dữ liệu quan trọng

- Không thay missing value bằng `0` một cách mặc định. Với SDG, `0` có thể là
  một quan sát thật và việc lấp 0 dễ làm sai hệ số. Baseline dùng median học
  chỉ trên train split.
- Chia train/validation/test theo thời gian, không random split, để tránh
  leakage từ tương lai.
- API chỉ dùng artifact đã version hóa; LLM không trực tiếp tính điểm.
- Hệ số linear là baseline giải thích được, chưa nên gọi là “trọng số chính
  thức” của SDSN nếu chưa kiểm định độ ổn định, fixed effects và robustness.

## 7. Lộ trình triển khai

Workflow chi tiết và tiêu chí hoàn thành nằm tại
[docs/workflow.md](docs/workflow.md). Ma trận phân công nằm tại
[docs/team-allocation.md](docs/team-allocation.md).
