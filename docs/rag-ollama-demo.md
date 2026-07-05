# RAG demo với Ollama `gemma3:4b`

Mục tiêu của bước này là chạy được luồng hỏi đáp chính sách trước khi phần ML thật hoàn thiện:

```text
PDF đã scan → chunks.jsonl → local RAG search → mock ML explain → Ollama gemma3:4b → câu trả lời chính sách
```

## 1. Điều kiện cần

Ollama trên máy host đã có model:

```powershell
ollama pull gemma3:4b
ollama list
```

Nếu chạy API bằng Docker, container gọi Ollama qua:

```dotenv
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gemma3:4b
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=disabled
```

`EMBEDDING_PROVIDER=disabled` là cố ý trong demo này. API sẽ dùng local lexical search trên `data/knowledge/processed/chunks.jsonl`, chưa cần Qdrant embedding.

## 2. Dữ liệu RAG

Đảm bảo đã có:

```text
data/knowledge/processed/chunks.jsonl
data/knowledge/text/group_*.txt
```

Nếu cần parse lại PDF:

```powershell
docker compose --profile jobs run --rm pdf-parser
```

## 3. Mock ML data

Trong lúc mô hình học máy thật chưa xong, API dùng artifact mock:

```text
artifacts/linear_regression/metadata.json
```

Các feature demo:

- `r_and_d_expenditure_pct_gdp`
- `governance_effectiveness`
- `education_index`
- `infrastructure_quality`
- `trade_openness`
- `fdi_inflow_pct_gdp`
- `digital_government_score`
- `patent_applications_per_million`

Sau này nhóm ML train xong chỉ cần ghi đè file metadata thật vào cùng đường dẫn.

## 4. Chạy stack

```powershell
docker compose up --build
```

Kiểm tra FastAPI:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Kỳ vọng:

```json
{
  "model_ready": true,
  "rag_enabled": true,
  "llm_enabled": true
}
```

## 5. Test hỏi RAG trực tiếp

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/search `
  -ContentType "application/json" `
  -Body '{"query":"Vietnam R&D digital government governance competitiveness","limit":3}'
```

## 6. Test assistant RAG + mock ML + Ollama

```powershell
$body = @{
  question = "Việt Nam nên ưu tiên chính sách nào để cải thiện năng lực cạnh tranh và thu hút FDI chất lượng cao?"
  country = "Vietnam"
  year = 2026
  features = @{
    r_and_d_expenditure_pct_gdp = 0.7
    governance_effectiveness = 58
    education_index = 0.74
    infrastructure_quality = 60
    trade_openness = 185
    fdi_inflow_pct_gdp = 4.2
    digital_government_score = 0.69
    patent_applications_per_million = 14
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/ask `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Nếu đi qua Spring Boot:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8080/api/v1/assistant/questions `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```
