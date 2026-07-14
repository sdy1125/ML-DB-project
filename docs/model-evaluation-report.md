# Báo cáo đánh giá và lựa chọn mô hình SDG16

## 1. Framing bắt buộc

`goal16` là điểm tổng hợp được xây dựng từ các chỉ số thành phần `n_sdg16_*`. Vì vậy, XGBoost trong project hiện tại phải được mô tả là mô hình **tái dựng điểm tổng hợp và phân rã đóng góp chỉ số**, không phải mô hình chứng minh nhân quả hay dự đoán độc lập về chất lượng quản trị.

## 2. So sánh hiệu suất mô hình

| Model | Split | RMSE | MAE | R2 | Vai trò |
|---|---:|---:|---:|---:|---|
| XGBoost SHAP Reconstruction Runner | Test | 1.8249 | 1.4019 | 0.9859 | Tái dựng điểm tổng hợp và giải thích contribution |
| Panel OLS + Fixed Effects | Full panel | 2.1276 | 1.6099 | 0.6263 | Baseline kinh tế lượng Phase 1 |
| GRU Sequence Forecaster | Test | 2.4411 | 1.9119 | 0.9747 | Dự báo chuỗi thời gian 2024-2030 |
| Optional XGBoost Tuned Pipeline | Overall | 7.7773 | 4.6659 | 0.6974 | Runner phụ/robustness |
| Spark Linear Regression | Test | 12.2139 | 9.2173 | 0.3509 | Legacy fallback |

## 3. Leakage/circular-target check

| Kiểm tra | Kết quả |
|---|---|
| Exact duplicate feature với `goal16` | Không có |
| Feature có `abs(corr) >= 0.98` với `goal16` | Không có |
| Circular composite-target issue | Có |

Diễn giải đúng: không phát hiện duplicate leakage trực tiếp, nhưng vẫn tồn tại circular composite-target issue vì `goal16` được xây từ các component indicators.

## 4. Data-quality aware contribution

Raw diagnostic contribution:

| Feature | Ý nghĩa | Contribution | Trạng thái |
|---|---|---:|---|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2.3477 | Flagged: VN = 0, benchmark khác 0 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1.9228 | Eligible |
| `n_sdg16_exprop` | Bảo vệ quyền tài sản / chống tịch thu | -1.3962 | Flagged: VN = 0, benchmark khác 0 |
| `n_sdg16_clabor` | Lao động trẻ em | -0.5346 | Eligible |
| `n_sdg16_admin` | Hành chính minh bạch | -0.3568 | Eligible |

Headline policy priorities:

| Feature | Ý nghĩa | Contribution |
|---|---|---:|
| `n_sdg16_justice` | Tiếp cận tư pháp | -1.9228 |
| `n_sdg16_clabor` | Lao động trẻ em | -0.5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0.3568 |

## 5. GRU forecast mới

| Năm | Predicted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.8785 |
| 2026 | 64.1630 |
| 2027 | 64.4077 |
| 2028 | 64.6922 |
| 2029 | 64.9520 |
| 2030 | 65.1743 |

Không dùng lại câu “forecast phẳng quanh 63.64”. Artifact mới cho thấy baseline tăng nhẹ đến 65.1743 năm 2030.

## 6. Kết luận sử dụng trong báo cáo

Project hiện tại nên được mô tả như một hệ thống diagnostic policy intelligence. XGBoost tái dựng `goal16` tốt nhất, Panel OLS là baseline kinh tế lượng, GRU cung cấp forecast 2024-2030, drill-down cấp tỉnh hỗ trợ định vị địa phương, và RAG + LLM sinh khuyến nghị dựa trên evidence. Các kết quả contribution dùng để ưu tiên phân tích, không phải bằng chứng nhân quả; RSF và expropriation hiện là diagnostic flagged rows, không phải headline policy claims.
