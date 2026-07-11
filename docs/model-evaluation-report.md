# Báo cáo đánh giá và lựa chọn mô hình dự đoán SDG16

## 1. Mục tiêu đánh giá

Mục tiêu của bước thực nghiệm là so sánh hiệu suất của các mô hình dự đoán điểm `Goal16` dựa trên bộ chỉ số SDG16, từ đó chọn mô hình phù hợp nhất để phục vụ các bước giải thích mô hình, dự báo nhiều năm và sinh khuyến nghị chính sách bằng RAG + LLM.

Các mô hình được đưa vào so sánh gồm:

1. Spark Linear Regression.
2. XGBoost SHAP Runner.
3. Optional XGBoost Tuned Pipeline.
4. GRU Sequence Forecaster.

Các thước đo đánh giá chính:

- RMSE: sai số căn bậc hai trung bình, càng thấp càng tốt.
- MAE: sai số tuyệt đối trung bình, càng thấp càng tốt.
- R²: mức độ giải thích phương sai của mô hình, càng cao càng tốt.

## 2. Dữ liệu và thiết lập thực nghiệm

Bộ dữ liệu chính được sử dụng là dữ liệu `SDR2024`, gồm:

- 4.392 dòng dữ liệu sau khi lọc nguồn `SDR2024`.
- 183 quốc gia.
- 17 biến đầu vào dạng `n_sdg16_*`.
- Biến mục tiêu: `goal16`.

Với mô hình XGBoost SHAP Runner, dữ liệu được chia theo thời gian:

- Train: các quan sát đến năm 2018.
- Validation: giai đoạn 2019–2021.
- Test: từ năm 2022 trở đi.

Với mô hình GRU, dữ liệu được chuyển thành chuỗi 5 năm liên tiếp cho từng quốc gia. Mỗi mẫu đầu vào là lịch sử 5 năm của 17 chỉ số SDG16, đầu ra là điểm `Goal16` của năm tiếp theo.

## 3. Kết quả so sánh hiệu suất

| Mô hình | Tập đánh giá | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| XGBoost SHAP Runner | Test | 1.8249 | 1.4019 | 0.9859 |
| GRU Sequence Forecaster | Test | 2.4411 | 1.9119 | 0.9747 |
| Optional XGBoost Tuned Pipeline | Overall | 7.7773 | 4.6659 | 0.6974 |
| Spark Linear Regression | Test | 12.2139 | 9.2173 | 0.3509 |

Kết quả cho thấy XGBoost SHAP Runner là mô hình có hiệu suất tốt nhất trong các mô hình được thử nghiệm. Mô hình này đạt RMSE thấp nhất là `1.8249`, MAE là `1.4019` và R² đạt `0.9859`. Điều này cho thấy mô hình giải thích được phần lớn biến thiên của điểm `Goal16` trên tập kiểm tra.

GRU Sequence Forecaster đứng thứ hai với RMSE `2.4411` và R² `0.9747`. Kết quả này cho thấy mô hình chuỗi thời gian có khả năng dự báo khá tốt, nhưng vẫn kém XGBoost trong bài toán hiện tại.

Spark Linear Regression có hiệu suất thấp nhất, với RMSE `12.2139` và R² `0.3509`. Tuy nhiên, mô hình tuyến tính vẫn có vai trò quan trọng như baseline và hỗ trợ reverse-engineer trọng số tuyến tính của các chỉ số.

## 4. Tối ưu hóa XGBoost SHAP Runner

Mô hình XGBoost SHAP Runner được tinh chỉnh bằng một lưới tham số nhỏ, có tính tái lập. Tổng cộng có 28 cấu hình được thử nghiệm.

Cấu hình tốt nhất:

```text
max_depth = 4
learning_rate = 0.09
n_estimators = 850
subsample = 0.82
colsample_bytree = 0.82
reg_alpha = 0.5
reg_lambda = 1.0
tree_method = hist
objective = reg:squarederror
```

Hiệu suất sau tối ưu:

| Tập đánh giá | RMSE | MAE | R² |
|---|---:|---:|---:|
| Validation | 1.1886 | 0.8502 | 0.9937 |
| Test | 1.8249 | 1.4019 | 0.9859 |

Sau tối ưu, XGBoost SHAP Runner được chọn làm mô hình chính cho hệ thống vì đạt sai số thấp nhất và vẫn hỗ trợ giải thích bằng SHAP thông qua `pred_contribs` của XGBoost.

## 5. Phân tích SHAP cho Việt Nam

Với mô hình XGBoost tốt nhất, hệ thống dùng SHAP contribution để xác định các chỉ số đang kéo điểm của Việt Nam xuống.

Các chỉ số có contribution âm lớn nhất đối với Việt Nam:

| Chỉ số | Ý nghĩa | SHAP contribution |
|---|---|---:|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2.3477 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1.9228 |
| `n_sdg16_exprop` | Chống tịch thu tài sản / bảo vệ quyền tài sản | -1.3962 |
| `n_sdg16_clabor` | Lao động trẻ em | -0.5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0.3568 |

Kết quả này cho thấy các nhóm vấn đề liên quan đến quản trị, tiếp cận tư pháp, bảo vệ quyền tài sản và minh bạch hành chính là những điểm nghẽn cần ưu tiên nếu muốn cải thiện điểm `Goal16`.

Ở mức độ quan trọng toàn cục, các biến có mean absolute SHAP cao nhất gồm:

| Chỉ số | Ý nghĩa | Mean absolute SHAP |
|---|---|---:|
| `n_sdg16_cpi` | Chống tham nhũng / CPI | 5.6254 |
| `n_sdg16_u5reg` | Đăng ký khai sinh | 3.1668 |
| `n_sdg16_detain` | Tạm giam trước xét xử | 2.1033 |
| `n_sdg16_homicides` | Tử vong do bạo lực | 1.9804 |
| `n_sdg16_clabor` | Lao động trẻ em | 1.9643 |

Điều này cho thấy mô hình không chỉ phụ thuộc vào một nhóm chỉ số đơn lẻ, mà kết hợp cả yếu tố quản trị, pháp quyền, an ninh xã hội và bảo vệ quyền con người.

## 6. Kết quả GRU Forecast

Mô hình GRU được dùng để kiểm tra khả năng dự báo chuỗi thời gian. Cấu hình tốt nhất sau thử nghiệm:

```text
hidden_size = 32
num_layers = 1
dropout = 0.0
learning_rate = 0.01
sequence_length = 5
```

Hiệu suất:

| Tập đánh giá | RMSE | MAE | R² |
|---|---:|---:|---:|
| Validation | 1.6999 | 1.2350 | 0.9871 |
| Test | 2.4411 | 1.9119 | 0.9747 |

Dự báo điểm `Goal16` của Việt Nam giai đoạn 2024–2030:

| Năm | Dự báo Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.6389 |
| 2026 | 63.6521 |
| 2027 | 63.6432 |
| 2028 | 63.6419 |
| 2029 | 63.6419 |
| 2030 | 63.6419 |

GRU cho thấy điểm số của Việt Nam có xu hướng ổn định quanh mức 63.64 nếu giả định cấu trúc chỉ số đầu vào không thay đổi đáng kể. Vì vậy, để tạo cải thiện rõ rệt đến năm 2030, cần can thiệp chính sách vào các nhóm chỉ số yếu đã được SHAP xác định.

## 7. Lựa chọn mô hình cuối cùng

Dựa trên kết quả so sánh, mô hình được chọn cho hệ thống là:

```text
XGBoost SHAP Runner
```

Lý do lựa chọn:

1. Có RMSE thấp nhất trên tập test.
2. Có R² cao nhất, đạt khoảng `0.9859`.
3. Có thể sinh SHAP contribution để giải thích vì sao điểm của Việt Nam tăng hoặc giảm.
4. Phù hợp với mục tiêu của đồ án: không chỉ dự đoán điểm, mà còn xác định chỉ số yếu và sinh khuyến nghị chính sách.

GRU được giữ lại như mô hình dự báo chuỗi thời gian bổ trợ. Spark Linear Regression được giữ làm baseline và phục vụ giải thích tuyến tính/reverse-engineering trọng số. Optional XGBoost Tuned Pipeline hiện có hiệu suất thấp hơn XGBoost SHAP Runner, nên chưa được chọn làm mô hình chính.

## 8. Ý nghĩa đối với hệ thống RAG + LLM

Kết quả mô hình học máy được dùng như đầu vào định lượng cho hệ thống sinh khuyến nghị:

- XGBoost SHAP Runner cung cấp điểm dự đoán hiện tại và các chỉ số kéo điểm Việt Nam xuống.
- GRU cung cấp dự báo điểm `Goal16` cho các năm 2024–2030.
- Subnational analyzer cung cấp drill-down cấp tỉnh khi có dữ liệu PAPI/PCI.
- RAG cung cấp bằng chứng chính sách từ các tài liệu PDF, báo cáo quốc tế và văn bản Việt Nam.
- LLM kết hợp các bằng chứng trên để sinh khuyến nghị chính sách có căn cứ.

Luồng cuối cùng của hệ thống có thể hiểu như sau:

```text
SDG16 data
  -> XGBoost/GRU/Linear
  -> SHAP + forecast + province drill-down
  -> RAG retrieval từ tài liệu chính sách
  -> LLM sinh khuyến nghị cuối cùng
```

## 9. Kết luận

Thực nghiệm cho thấy các mô hình phi tuyến cho hiệu suất vượt trội so với mô hình tuyến tính. XGBoost SHAP Runner là mô hình tốt nhất hiện tại, vừa có độ chính xác cao vừa có khả năng giải thích bằng SHAP. GRU có hiệu suất tốt cho bài toán dự báo chuỗi thời gian nhưng chưa vượt XGBoost trong bài toán dự đoán điểm `Goal16` hiện tại.

Đối với Việt Nam, các điểm nghẽn chính được mô hình xác định gồm tự do báo chí/trách nhiệm giải trình, tiếp cận tư pháp, bảo vệ quyền tài sản, lao động trẻ em và hành chính minh bạch. Đây là các nhóm chỉ số nên được ưu tiên trong bước sinh khuyến nghị chính sách của hệ thống RAG + LLM.

Lệnh dùng để tái tạo kết quả:

```powershell
python scripts\optimize_and_compare_models.py
```

Các file kết quả chính:

- `artifacts/model_comparison/model_comparison.csv`
- `artifacts/model_comparison/best_model.json`
- `artifacts/shap/shap_summary.json`
- `artifacts/gru/gru_summary.json`
- `artifacts/gru/vietnam_forecast_2024_2030.csv`
