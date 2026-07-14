# Ứng dụng học máy giải thích được và RAG trong phân tích SDG16 cho Việt Nam

**Tác giả:** ........................................................  
**Đơn vị:** ........................................................  
**Tác giả liên hệ:** ........................................................

## Tóm tắt

Nghiên cứu này đề xuất một khung phân tích SDG16 cho Việt Nam kết hợp Panel OLS, XGBoost, giải thích đóng góp chỉ số, GRU và truy xuất tăng cường sinh văn bản. Dữ liệu được xây dựng từ SDR2024, gồm 4.392 quan sát của 183 quốc gia, biến mục tiêu `goal16` và 17 chỉ số thành phần dạng `n_sdg16_*`. Do `goal16` là điểm tổng hợp được xây dựng từ các chỉ số thành phần, XGBoost trong nghiên cứu này được diễn giải là mô hình tái dựng điểm tổng hợp và phân rã đóng góp, không phải mô hình chứng minh nhân quả. Kết quả cho thấy XGBoost SHAP Reconstruction Runner đạt RMSE = 1,8249, MAE = 1,4019 và R2 = 0,9859 trên tập kiểm tra. Panel OLS với hiệu ứng cố định đạt R2 overall = 0,6263, đóng vai trò baseline kinh tế lượng. GRU đạt RMSE = 2,4411 và R2 = 0,9747, đồng thời dự báo điểm Việt Nam tăng từ 63,6456 năm 2024 lên 65,1743 năm 2030. Phân tích đóng góp thô cho thấy RSF và expropriation có đóng góp âm lớn nhưng bị gắn cờ do giá trị Việt Nam bằng 0 trong khi benchmark khác 0; vì vậy headline priorities chỉ gồm tiếp cận tư pháp, lao động trẻ em và minh bạch hành chính. Kết quả định lượng được kết nối với RAG + LLM nhằm sinh khuyến nghị chính sách dựa trên bằng chứng.

**Từ khóa:** SDG16; XGBoost; Panel OLS; GRU; RAG; khuyến nghị chính sách.

## 1. Đặt vấn đề

SDG16 nhấn mạnh hòa bình, công lý, thể chế hiệu quả, minh bạch và trách nhiệm giải trình. Tuy nhiên, điểm tổng hợp SDG16 thường chỉ cho biết vị trí tương đối của một quốc gia, chưa chỉ rõ chỉ số nào đang kéo điểm xuống và khuyến nghị chính sách nào có bằng chứng hỗ trợ.

Nghiên cứu này xây dựng workflow theo đúng project hiện tại: Panel OLS làm baseline kinh tế lượng, XGBoost tái dựng điểm tổng hợp, contribution analysis xác định chỉ số yếu, GRU dự báo 2024-2030, drill-down cấp tỉnh ánh xạ sang PAPI/PCI và RAG + LLM sinh khuyến nghị chính sách.

## 2. Phương pháp

### 2.1. Dữ liệu

| Thành phần | Mô tả |
|---|---|
| Nguồn | SDR2024 |
| Số quan sát | 4.392 |
| Số quốc gia | 183 |
| Input | 17 chỉ số `n_sdg16_*` |
| Target | `goal16` |

### 2.2. Luồng mô hình

```text
sdg16_spark.csv
  -> Panel OLS + Fixed Effects
  -> XGBoost composite-score reconstruction
  -> XGBoost tree-contribution analysis
  -> GRU forecast 2024-2030
  -> RAG + LLM recommendation

sdg16_provinces.csv
  -> Subnational drill-down / Panel FE tỉnh
  -> RAG + LLM recommendation
```

### 2.3. Lưu ý phương pháp

Vì `goal16` là điểm tổng hợp từ các biến đầu vào, R2 cao của XGBoost phản ánh khả năng tái dựng công thức/quan hệ tổng hợp của chỉ số, không phản ánh quan hệ nhân quả. Các contribution dùng để ưu tiên phân tích và định hướng khuyến nghị, không thay thế đánh giá chuyên gia.

## 3. Kết quả

### 3.1. So sánh mô hình

| Mô hình | Split | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| XGBoost SHAP Reconstruction Runner | Test | 1,8249 | 1,4019 | 0,9859 |
| Panel OLS + Fixed Effects | Full panel | 2,1276 | 1,6099 | 0,6263 |
| GRU Sequence Forecaster | Test | 2,4411 | 1,9119 | 0,9747 |
| Optional XGBoost Tuned Pipeline | Overall | 7,7773 | 4,6659 | 0,6974 |
| Spark Linear Regression | Test | 12,2139 | 9,2173 | 0,3509 |

### 3.2. Raw diagnostic và headline priorities

Raw diagnostic contribution:

| Chỉ số | Diễn giải | Contribution | Trạng thái dữ liệu |
|---|---|---:|---|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2,3477 | Flagged: giá trị VN = 0, benchmark khác 0 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1,9228 | Eligible |
| `n_sdg16_exprop` | Bảo vệ quyền tài sản / chống tịch thu | -1,3962 | Flagged: giá trị VN = 0, benchmark khác 0 |
| `n_sdg16_clabor` | Lao động trẻ em | -0,5346 | Eligible |
| `n_sdg16_admin` | Hành chính minh bạch | -0,3568 | Eligible |

Headline policy priorities sau khi loại feature bị flag:

| Chỉ số | Diễn giải | Contribution |
|---|---|---:|
| `n_sdg16_justice` | Tiếp cận tư pháp | -1,9228 |
| `n_sdg16_clabor` | Lao động trẻ em | -0,5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0,3568 |

### 3.3. Dự báo GRU cho Việt Nam

| Năm | Predicted Goal16 |
|---:|---:|
| 2024 | 63,6456 |
| 2025 | 63,8785 |
| 2026 | 64,1630 |
| 2027 | 64,4077 |
| 2028 | 64,6922 |
| 2029 | 64,9520 |
| 2030 | 65,1743 |

## 4. Bàn luận

XGBoost đạt kết quả tốt nhất vì bài toán hiện tại là tái dựng điểm tổng hợp từ các chỉ số thành phần. Đây là điểm mạnh cho mục tiêu phân rã đóng góp chỉ số, nhưng cũng là giới hạn lớn nếu muốn viết theo hướng dự đoán độc lập hay nhân quả.

RSF và expropriation không nên được dùng làm phát hiện nổi bật cho đến khi xác minh lại dữ liệu gốc, vì giá trị Việt Nam bằng 0 trong khi benchmark khác 0. Do đó, khuyến nghị headline nên tập trung vào tiếp cận tư pháp, lao động trẻ em và minh bạch hành chính.

Panel OLS vẫn cần được giữ trong bài vì đây là baseline kinh tế lượng chính. GRU đóng vai trò bổ trợ cho dự báo chuỗi thời gian. Subnational drill-down giúp chuyển nhóm chỉ số yếu cấp quốc gia sang proxy PAPI/PCI cấp tỉnh, nhưng kết quả tỉnh cần được xem là lớp hỗ trợ nếu dữ liệu chưa được validate chính thức.

RAG + LLM giúp chuyển kết quả mô hình thành khuyến nghị chính sách có dẫn chứng tài liệu. Tuy nhiên, layer này hiện là prototype và chưa có expert validation.

## 5. Kết luận

Project hiện tại phù hợp nhất để trình bày như một hệ thống diagnostic policy intelligence cho SDG16. XGBoost tái dựng `goal16` tốt nhất, Panel OLS là baseline kinh tế lượng, GRU cung cấp forecast 2024-2030, drill-down cấp tỉnh hỗ trợ định vị địa phương, và RAG + LLM sinh khuyến nghị dựa trên evidence. Khi viết bài, cần tránh claim nhân quả, cần nêu rõ giới hạn composite-target và không dùng feature bị flag làm headline policy claim.

## Abstract

This study proposes an SDG16 intelligence framework for Vietnam combining Panel OLS, XGBoost composite-score reconstruction, tree-contribution analysis, GRU forecasting, subnational drill-down, and retrieval-augmented generation. Using SDR2024 data with 4,392 observations from 183 countries and 17 SDG16 component indicators, the main XGBoost runner achieves RMSE = 1.8249, MAE = 1.4019, and R2 = 0.9859. Since `goal16` is constructed from the same component indicators, this result is interpreted as composite-score reconstruction rather than causal or independent prediction. GRU forecasts Vietnam's baseline from 63.6456 in 2024 to 65.1743 in 2030. After excluding zero-valued flagged indicators from headline claims, the framework identifies access to justice, child labor, and administrative transparency as key policy-priority contributors for Vietnam and connects these outputs with a RAG + LLM recommendation layer.

**Keywords:** SDG16; XGBoost; Panel OLS; GRU; RAG; Vietnam.
