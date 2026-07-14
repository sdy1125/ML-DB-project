# Tài liệu luồng SDG16 — bản Panel OLS

File này được rút gọn để tránh lệch với luồng mới. Bản đầy đủ đang nằm tại:

```text
docs/SDG16_Luong_hoat_dong_cap_nhat.md
```

## Luồng đúng hiện tại

```text
data/clean/sdg16_spark.csv
  -> Phase 1: Panel OLS + Fixed Effects
  -> Phase 2: XGBoost composite-score reconstruction
  -> Phase 3: XGBoost tree-contribution analysis
  -> Phase 5: GRU forecast 2024-2030
  -> Phase 6: RAG + LLM

data/subnational/sdg16_provinces.csv
  -> Phase 4: Subnational drill-down / province Panel FE
  -> Phase 6: RAG + LLM
```

## Phase 1 bắt buộc là Panel OLS

Entrypoint:

```powershell
python scripts\run_panel_ols.py
```

Artifact:

```text
artifacts/panel_ols/panel_ols_results.json
```

Kết quả:

| Chỉ số | Giá trị |
|---|---:|
| RMSE | 2.1276 |
| MAE | 1.6099 |
| R2 overall | 0.6263 |

## Lưu ý reviewer

`goal16` là điểm tổng hợp từ các chỉ số `n_sdg16_*`, nên XGBoost phải được gọi là composite-score reconstruction. Không claim nhân quả, không claim dự đoán độc lập.

GRU forecast đúng hiện tại:

| Năm | Predicted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.8785 |
| 2026 | 64.1630 |
| 2027 | 64.4077 |
| 2028 | 64.6922 |
| 2029 | 64.9520 |
| 2030 | 65.1743 |
