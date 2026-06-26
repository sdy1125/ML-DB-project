# Báo Cáo Chi Tiết Pipeline 3 Mô Hình

## 1. Mục Tiêu Chung

Pipeline triển khai 3 mô hình với 3 vai trò tách biệt:

1. **Model 1 - Elastic Net Regression**
   Diễn giải ảnh hưởng tương đối của các chỉ số SDG đầu vào đối với `goal16`.

2. **Model 2 - XGBoost Regression**
   Tăng độ chính xác dự đoán bằng cách học quan hệ phi tuyến và tương tác giữa các biến.

3. **Model 3 - ARIMAX/ARIMA Forecast**
   Dự báo điểm `goal16` của Việt Nam giai đoạn 2025-2030 theo 3 kịch bản.

Ba mô hình không được dùng để chọn "mô hình tốt nhất". Mỗi mô hình có một vai trò riêng: diễn giải, tăng accuracy, và forecast.

## 2. Dữ Liệu Đầu Vào

Pipeline tự động chọn file:

```text
data/clean/sdg16_spark.csv
```

Thông tin dữ liệu:

| Thành phần | Giá trị |
|---|---:|
| Số dòng dữ liệu gốc | 10,150 |
| Số cột dữ liệu gốc | 21 |
| Số dòng sau xử lý | 10,150 |
| Số cột sau xử lý | 35 |
| Cột quốc gia | `country` |
| Cột năm | `year` |
| Cột target | `goal16` |
| Train years | 2000-2018 |
| Test years | 2019-2023 |

Các biến đầu vào chính là các chỉ số dạng số có tiền tố `n_sdg16_`, ví dụ:

```text
n_sdg16_homicide
n_sdg16_crime
n_sdg16_safe
n_sdg16_security
n_sdg16_justice
n_sdg16_cpi
```

## 3. Tiền Xử Lý Và Feature Engineering

Pipeline thực hiện các bước chính:

1. Tự động tìm file `.csv`, `.xlsx`, `.xls` trong `data/clean`.
2. Chuẩn hóa tên cột về dạng `snake_case`.
3. Tự động nhận diện `country`, `year`, `target`.
4. Sort dữ liệu theo `country` và `year`.
5. Fill missing values theo từng quốc gia bằng forward-fill và backward-fill.
6. Tạo feature thời gian theo từng quốc gia:
   - `goal16_lag_1`
   - `goal16_lag_2`
   - `goal16_rolling_mean_3`
   - lag/delta cho một số biến numeric.
7. Chia train/test theo thời gian, không random split.
8. Loại `delta_goal16` khỏi tập feature chung để tránh leakage vì biến này dùng target hiện tại.

Lưu ý quan trọng: các feature lịch sử của target vẫn được tạo để phục vụ bài toán dự đoán của Model 2, nhưng **không được đưa vào Model 1**.

## 4. Output Tổng Quan

Các output chính được lưu trong:

```text
outputs/
```

| Thư mục | Nội dung |
|---|---|
| `outputs/models` | File mô hình đã train |
| `outputs/tables` | Bảng trọng số, feature importance, báo cáo |
| `outputs/figures` | Biểu đồ trực quan |
| `outputs/predictions` | File dự đoán và forecast |
| `outputs/logs` | Log quá trình chạy pipeline |

## 5. Model 1 - Elastic Net Regression

### 5.1. Model 1 Làm Gì?

Model 1 dùng Elastic Net để ước lượng hệ số tuyến tính có regularization trên các chỉ số SDG đã chuẩn hóa. Mục tiêu là trả lời:

```text
Chỉ số SDG nào liên quan mạnh hơn đến điểm goal16?
Quan hệ thống kê đó có chiều dương hay chiều âm?
```

### 5.2. Chỉnh Sửa Quan Trọng Theo Yêu Cầu Nghiên Cứu

Vì Model 1 có vai trò **diễn giải ảnh hưởng của các chỉ số SDG**, mô hình hiện tại đã loại các biến lịch sử của chính target trước khi huấn luyện:

```text
goal16_lag_1
goal16_lag_2
goal16_rolling_mean_3
```

Điều này rất quan trọng. Nếu đưa các biến này vào Model 1, hệ số dễ bị chi phối bởi quan hệ tự hồi quy của `goal16`, khiến bảng trọng số không còn phản ánh rõ vai trò của các chỉ số SDG đầu vào.

Sau chỉnh sửa, bảng trọng số Model 1 không còn chứa biến lịch sử của target:

```text
outputs/tables/model1_hidden_weights.csv
```

### 5.3. Đầu Vào Của Model 1

Model 1 sử dụng:

- Các biến numeric từ dữ liệu gốc.
- Các lag/delta của biến chỉ số SDG đầu vào.
- Không đưa country dummy vào mặc định để bảng trọng số tập trung vào numeric indicators.
- Không dùng `goal16_lag_1`, `goal16_lag_2`, `goal16_rolling_mean_3`.
- Không dùng `delta_goal16`.

Các biến được scale bằng `StandardScaler`, missing numeric được impute bằng median trong sklearn Pipeline.

### 5.4. Đầu Ra Của Model 1

```text
outputs/models/model1_elasticnet.pkl
outputs/predictions/model1_predictions.csv
outputs/tables/model1_hidden_weights.csv
outputs/figures/model1_top_hidden_weights.png
outputs/figures/model1_actual_vs_predicted.png
```

### 5.5. Cấu Trúc Bảng Hidden Weights

| Cột | Ý nghĩa |
|---|---|
| `feature` | Tên biến đầu vào |
| `coefficient` | Hệ số Elastic Net |
| `abs_coefficient` | Trị tuyệt đối của hệ số |
| `normalized_weight` | Trọng số tương đối sau chuẩn hóa |
| `direction` | Chiều tác động: positive, negative, zero |

```text
normalized_weight = abs_coefficient / tổng abs_coefficient
```

### 5.6. Kết Quả Model 1

Metrics trên test set sau khi loại target-history khỏi Model 1:

| Metric | Giá trị |
|---|---:|
| RMSE | 7.9003 |
| MAE | 5.5864 |
| R2 | 0.7188 |
| MAPE | 9.5368% |

Top hidden weights của Model 1:

| Feature | Direction | Normalized weight |
|---|---|---:|
| `n_sdg16_homicides` | positive | 0.1102 |
| `n_sdg16_detain` | positive | 0.1001 |
| `n_sdg16_cpi` | positive | 0.0954 |
| `n_sdg16_cpi_lag_1` | positive | 0.0952 |
| `n_sdg16_u5reg` | positive | 0.0856 |
| `n_sdg16_justice` | negative | 0.0656 |
| `n_sdg16_power` | positive | 0.0631 |
| `n_sdg16_exprop` | positive | 0.0548 |
| `n_sdg16_weaponsexp` | positive | 0.0459 |
| `n_sdg16_crime` | positive | 0.0379 |

### 5.7. Đánh Giá Model 1

R2 của Model 1 giảm từ mức rất cao trước đây xuống khoảng `0.7188`. Đây là kết quả hợp lý hơn cho một mô hình diễn giải chỉ số, vì mô hình không còn được hưởng lợi từ các biến tự hồi quy của `goal16`.

Điểm quan trọng nhất của Model 1 không phải là dự đoán chính xác nhất, mà là bảng trọng số có thể diễn giải. Vì vậy, việc loại target-history giúp kết quả phù hợp hơn với yêu cầu nghiên cứu.

### 5.8. Lưu Ý Khi Trình Bày Model 1

- Không gọi các trọng số này là "trọng số chính thức" của bộ chỉ số.
- Đây là trọng số ước lượng từ dữ liệu và mô hình.
- Hệ số âm/dương phản ánh quan hệ thống kê trong dữ liệu, không tự động chứng minh quan hệ nhân quả.
- Model 1 hiện không dùng `goal16_lag_1`, `goal16_lag_2`, `goal16_rolling_mean_3`.
- Khi muốn nói về ảnh hưởng của các chỉ số SDG, dùng `model1_hidden_weights.csv`.

## 6. Model 2 - XGBoost Regression

### 6.1. Model 2 Làm Gì?

Model 2 dùng XGBoost Regression để tăng độ chính xác dự đoán `goal16`. Khác với Model 1, Model 2 phục vụ nhiệm vụ dự đoán nên vẫn có thể dùng các biến lịch sử của target.

### 6.2. Đầu Vào Của Model 2

Model 2 sử dụng tập feature chung đã chuẩn bị:

- Các biến numeric gốc.
- Các lag feature.
- Các delta feature hợp lệ.
- Bao gồm target-history như `goal16_rolling_mean_3`, `goal16_lag_2`, vì đây là mô hình dự đoán.
- Train/test theo năm, không random split.

### 6.3. Đầu Ra Của Model 2

```text
outputs/models/model2_xgboost.pkl
outputs/predictions/model2_predictions.csv
outputs/tables/model2_feature_importance.csv
outputs/figures/model2_feature_importance.png
outputs/figures/model2_actual_vs_predicted.png
outputs/figures/model2_residual_plot.png
```

### 6.4. Kết Quả Model 2

| Metric | Giá trị |
|---|---:|
| RMSE | 2.5489 |
| MAE | 1.6510 |
| R2 | 0.9707 |
| MAPE | 2.7629% |

Top feature importance:

| Feature | Normalized importance |
|---|---:|
| `goal16_rolling_mean_3` | 0.5035 |
| `goal16_lag_2` | 0.3751 |
| `n_sdg16_u5reg` | 0.0195 |
| `n_sdg16_clabor` | 0.0191 |
| `n_sdg16_exprop` | 0.0094 |

### 6.5. Đánh Giá Model 2

Model 2 có R2 khoảng `0.9707`, cao hơn Model 1 vì Model 2 được thiết kế cho accuracy và được phép dùng target-history. Feature importance của XGBoost không nên diễn giải giống hệ số Elastic Net; nó cho biết biến nào hữu ích cho dự đoán, không cho biết trực tiếp chiều tác động.

## 7. Model 3 - ARIMAX/ARIMA Forecast

### 7.1. Model 3 Làm Gì?

Model 3 dự báo điểm `goal16` của Việt Nam giai đoạn:

```text
2025-2030
```

Model này không nhằm tìm trọng số ẩn và cũng không nhằm so sánh accuracy với Model 1/2.

### 7.2. Cách Model 3 Hoạt Động

Pipeline ưu tiên dùng ARIMAX nếu có biến ngoại sinh phù hợp. Các biến ngoại sinh được chọn theo top hidden weights từ Model 1 sau chỉnh sửa.

Trong lần chạy hiện tại:

```text
Model used: ARIMAX
Order: (1, 0, 1)
```

Biến ngoại sinh:

```text
n_sdg16_homicides
n_sdg16_detain
n_sdg16_cpi
n_sdg16_cpi_lag_1
n_sdg16_u5reg
```

### 7.3. Kết Quả Forecast Việt Nam 2025-2030

| Year | Pessimistic | Base | Optimistic |
|---:|---:|---:|---:|
| 2025 | 61.2452 | 64.4681 | 67.6910 |
| 2026 | 61.2505 | 64.4743 | 67.6981 |
| 2027 | 61.2707 | 64.4955 | 67.7202 |
| 2028 | 61.2882 | 64.5139 | 67.7396 |
| 2029 | 61.3062 | 64.5328 | 67.7595 |
| 2030 | 61.3241 | 64.5517 | 67.7792 |

### 7.4. Scenario Impact Theo Từng Biến +1%

Ngoài forecast 3 kịch bản, Model 3 hiện bổ sung bảng what-if sensitivity:

```text
outputs/tables/model3_policy_impact_plus1pct.csv
```

Cách tính:

1. Giữ nguyên từng bối cảnh `pessimistic`, `base`, `optimistic`.
2. Tăng riêng từng biến ngoại sinh thêm `+1%`.
3. Chạy lại forecast 2025-2030.
4. Tính `score_delta = intervention_score - baseline_score`.

Các cột chính trong bảng:

| Cột | Ý nghĩa |
|---|---|
| `year` | Năm forecast |
| `scenario` | Bối cảnh pessimistic/base/optimistic |
| `indicator` | Biến ngoại sinh được tăng riêng +1% |
| `baseline_score` | Điểm forecast gốc của bối cảnh đó |
| `intervention_score` | Điểm forecast sau khi tăng biến đó +1% |
| `score_delta` | Mức thay đổi điểm so với baseline |

Tóm tắt tác động trung bình của `+1%` theo giai đoạn 2025-2030:

| Scenario | Indicator | Avg score delta |
|---|---|---:|
| optimistic | `n_sdg16_homicides` | 2.3894 |
| base | `n_sdg16_homicides` | 2.2756 |
| pessimistic | `n_sdg16_homicides` | 2.1618 |
| optimistic | `n_sdg16_cpi` | 0.0671 |
| base | `n_sdg16_cpi` | 0.0639 |
| pessimistic | `n_sdg16_cpi` | 0.0607 |
| optimistic | `n_sdg16_cpi_lag_1` | 0.0544 |
| base | `n_sdg16_cpi_lag_1` | 0.0518 |
| pessimistic | `n_sdg16_cpi_lag_1` | 0.0492 |
| pessimistic | `n_sdg16_detain` | -0.0447 |

Ví dụ đọc kết quả: trong bối cảnh `base`, nếu `n_sdg16_cpi` tăng thêm `1%` so với đường dự phóng ban đầu, điểm `goal16` trung bình giai đoạn 2025-2030 tăng khoảng `0.0639` điểm theo mô hình.

Lưu ý: đây là phân tích độ nhạy theo mô hình, không tự động chứng minh quan hệ nhân quả. Nếu muốn dùng cho khuyến nghị chính sách, cần kiểm tra ý nghĩa thực tế của từng chỉ số và chiều đo của biến đó.

### 7.5. Đánh Giá Model 3

Forecast base tăng nhẹ từ khoảng `64.47` năm 2025 lên khoảng `64.55` năm 2030. Biên giữa pessimistic và optimistic rộng hơn trước vì nhóm biến ngoại sinh được chọn lại sau khi Model 1 không còn dùng target-history.

Khi trình bày, nên nhấn mạnh đây là forecast định lượng dựa trên xu hướng dữ liệu. Phần `+1%` giúp trả lời câu hỏi "nếu một biến ngoại sinh tăng thêm thì score đổi bao nhiêu", nhưng vẫn là what-if sensitivity, chưa phải mô phỏng chính sách sâu hay kết luận nhân quả.

## 8. Đánh Giá Chung Ba Mô Hình

| Model | Vai trò chính | Điểm mạnh | Output quan trọng nhất |
|---|---|---|---|
| Elastic Net | Diễn giải trọng số chỉ số SDG | Có hệ số, dễ diễn giải | `model1_hidden_weights.csv` |
| XGBoost | Tăng độ chính xác dự đoán | Học phi tuyến tốt | `model2_predictions.csv`, `model2_feature_importance.csv` |
| ARIMAX/ARIMA | Forecast Việt Nam 2025-2030 | Dự báo theo chuỗi thời gian | `model3_vietnam_forecast_2025_2030.csv` |

Logic trình bày:

1. **Elastic Net** giúp hiểu các chỉ số SDG nào liên quan mạnh đến `goal16`, không dùng target-history.
2. **XGBoost** kiểm tra khả năng dự đoán khi cho phép quan hệ phi tuyến và target-history.
3. **ARIMAX** dùng kết quả Model 1 và chuỗi Việt Nam để forecast giai đoạn 2025-2030.

Không nên nói XGBoost "thắng" Elastic Net, vì hai mô hình có nhiệm vụ khác nhau.

## 9. Các File Cần Mở Khi Trình Bày

### Báo cáo

```text
outputs/tables/model_summary_report.md
outputs/tables/three_model_detailed_report.md
```

### Model 1

```text
outputs/tables/model1_hidden_weights.csv
outputs/figures/model1_top_hidden_weights.png
outputs/figures/model1_actual_vs_predicted.png
```

### Model 2

```text
outputs/tables/model2_feature_importance.csv
outputs/figures/model2_feature_importance.png
outputs/figures/model2_actual_vs_predicted.png
outputs/figures/model2_residual_plot.png
```

### Model 3

```text
outputs/predictions/model3_vietnam_forecast_2025_2030.csv
outputs/tables/model3_policy_impact_plus1pct.csv
outputs/figures/model3_vietnam_forecast_2025_2030.png
```

## 10. Cách Chạy Lại Pipeline

Từ project root:

```powershell
python -m pipelines.modeling_pipeline
```

Nếu cần chỉ định rõ cột:

```powershell
python -m pipelines.modeling_pipeline --country-col Country --year-col Year --target-col goal16
```

## 11. Kết Luận Ngắn Gọn Để Trình Bày

Pipeline hiện đã đúng vai trò nghiên cứu của Model 1: Elastic Net dùng để diễn giải ảnh hưởng tương đối của các chỉ số SDG và không còn đưa các biến lịch sử của chính target vào mô hình. Model 2 vẫn phục vụ dự đoán nên có thể dùng target-history, còn Model 3 dùng top indicators mới từ Model 1 để forecast Việt Nam giai đoạn 2025-2030.
