# Final insight workflow

Luồng này nối kết quả Machine Learning hiện có với RAG/LLM để tạo output cuối:

```text
metadata.json
  ↓
current Vietnam features từ data/clean/sdg16_spark.csv
  ↓
predict + contribution fallback
  ↓
top chỉ số kéo điểm xuống
  ↓
RAG search trong chunks.jsonl
  ↓
Ollama/LLM sinh khuyến nghị chính sách
```

## Artifact đang dùng

```text
artifacts/linear_regression/metadata.json
```

File này chứa:

- `features`: 17 chỉ số SDG16.
- `intercept`, `coefficients`.
- `feature_defaults`, `feature_means`, `feature_stds`.
- `metrics`, `split`, `model_version`.

Hiện endpoint dùng contribution từ linear metadata làm explainability fallback. Khi có output thật từ `shap_sdg16_vietnam.py`, có thể map thêm:

```text
shap_output/global_shap_importance.csv
shap_output/gap_analysis.csv
shap_output/counterfactual.csv
shap_output/shap_values_vn.npy
```

## Endpoint FastAPI

```powershell
Invoke-RestMethod "http://localhost:8000/insights/final?country=Vietnam&use_llm=false" |
  ConvertTo-Json -Depth 10
```

Bật LLM/Ollama:

```powershell
Invoke-RestMethod "http://localhost:8000/insights/final?country=Vietnam&use_llm=true" |
  ConvertTo-Json -Depth 10
```

## Endpoint Spring Boot

```powershell
Invoke-RestMethod "http://localhost:8081/api/v1/insights/final?country=Vietnam&useLlm=true" |
  ConvertTo-Json -Depth 10
```

## Output

Endpoint trả về:

- `current_score`: điểm mô hình hiện tại của Việt Nam.
- `observed_score`: điểm quan sát trong dataset nếu có.
- `weakest_indicators`: chỉ số kéo điểm xuống mạnh nhất.
- `strongest_indicators`: chỉ số kéo điểm lên mạnh nhất.
- `forecasts`: pessimistic/base/optimistic.
- `province`: trạng thái drill-down tỉnh.
- `evidence`: nguồn RAG.
- `recommendation`: khuyến nghị chính sách sinh bởi LLM hoặc fallback text.

## Lưu ý về drill-down tỉnh

Project hiện chưa có dataset PAPI/PCI cấp tỉnh. Vì vậy hệ thống chưa kết luận “tỉnh nào tệ nhất” để tránh bịa dữ liệu. Khi thêm dataset tỉnh, bước tiếp theo là tạo service map top chỉ số SDG16 yếu sang feature PAPI/PCI và chạy panel FE cấp tỉnh.
