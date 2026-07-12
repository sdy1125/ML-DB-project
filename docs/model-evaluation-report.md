# Báo cáo đánh giá và lựa chọn mô hình dự đoán SDG16

## 1. Mục tiêu đánh giá

Mục tiêu của bước thực nghiệm là so sánh hiệu suất của các mô hình dự đoán điểm `Goal16` dựa trên bộ chỉ số SDG16, từ đó chọn mô hình phù hợp nhất để phục vụ các bước giải thích mô hình, dự báo nhiều năm và sinh khuyến nghị chính sách bằng RAG + LLM.

Các mô hình được đưa vào so sánh gồm:

1. Panel OLS + Fixed Effects.
2. Spark Linear Regression legacy fallback.
3. XGBoost SHAP Runner.
4. Optional XGBoost Tuned Pipeline.
5. GRU Sequence Forecaster.

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
| Panel OLS + Fixed Effects | Full panel | N/A | N/A | 0.6263 |
| Spark Linear Regression | Test | 12.2139 | 9.2173 | 0.3509 |

Kết quả cho thấy XGBoost SHAP Runner là mô hình có hiệu suất tốt nhất trong các mô hình được thử nghiệm. Mô hình này đạt RMSE thấp nhất là `1.8249`, MAE là `1.4019` và R² đạt `0.9859`. Điều này cho thấy mô hình giải thích được phần lớn biến thiên của điểm `Goal16` trên tập kiểm tra.

GRU Sequence Forecaster đứng thứ hai với RMSE `2.4411` và R² `0.9747`. Kết quả này cho thấy mô hình chuỗi thời gian có khả năng dự báo khá tốt, nhưng vẫn kém XGBoost trong bài toán hiện tại.

Panel OLS + Fixed Effects là baseline kinh tế lượng chính của Phase 1, đạt R² overall `0.6263` khi kiểm soát hiệu ứng cố định theo quốc gia và theo năm. Spark Linear Regression hiện chỉ giữ vai trò legacy fallback metadata cho một số API cũ.

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

GRU được giữ lại như mô hình dự báo chuỗi thời gian bổ trợ. Panel OLS + Fixed Effects được giữ làm baseline kinh tế lượng và phục vụ giải thích tuyến tính/reverse-engineering trọng số. Optional XGBoost Tuned Pipeline hiện có hiệu suất thấp hơn XGBoost SHAP Runner, nên chưa được chọn làm mô hình chính.

## 8. Ý nghĩa đối với hệ thống RAG + LLM

Kết quả mô hình học máy được dùng như đầu vào định lượng cho hệ thống sinh khuyến nghị:

- XGBoost SHAP Runner cung cấp điểm dự đoán hiện tại và các chỉ số kéo điểm Việt Nam xuống.
- GRU cung cấp dự báo điểm `Goal16` cho các năm 2024–2030.
- Subnational analyzer cung cấp drill-down cấp tỉnh khi có dữ liệu PAPI/PCI.
- RAG cung cấp bằng chứng chính sách từ các tài liệu PDF, báo cáo quốc tế và văn bản Việt Nam.
- LLM kết hợp các bằng chứng trên để sinh khuyến nghị chính sách có căn cứ.

Luồng cuối cùng của hệ thống có thể hiểu như sau:

```text
sdg16.csv / data/clean/sdg16_spark.csv
  -> Phase 1: Panel OLS + Fixed Effects
  -> Phase 2: XGBoost dự đoán goal16
  -> Phase 3: SHAP xác định top chỉ số kéo điểm Việt Nam xuống
  -> Phase 5: GRU dự báo 2024–2030
  -> Phase 6: RAG + LLM sinh khuyến nghị chính sách
  -> Output cuối

SDG16_dataset.csv / data/subnational/sdg16_provinces.csv
  -> Phase 4: Drill-down map sang PAPI/PCI tỉnh
  -> Panel FE tỉnh
  -> Phase 6: RAG + LLM
  -> Output tỉnh yếu nhất
```

## 9. Đối chiếu luồng hoạt động với sơ đồ mục tiêu

Sau khi kiểm tra lại project, luồng hiện tại đã khớp với sơ đồ mục tiêu ở mức thực thi.

| Thành phần | Script/Artifact | Trạng thái | Kết quả chính |
|---|---|---|---|
| Phase 1 Panel OLS | `python scripts\run_panel_ols.py` | Đã chạy | Entity+Time FE R² overall = `0.6263` |
| Phase 2 XGBoost | `artifacts/xgboost/metadata.json` | Đã có | Optional XGBoost R² = `0.6974` |
| Phase 3 SHAP | `artifacts/shap/shap_summary.json` | Đã có | XGBoost SHAP Runner R² = `0.9859` |
| Phase 4 Drill-down | `python scripts\run_subnational.py` | Đã chạy | Tỉnh yếu nhất: Hải Dương, score = `0.3772` |
| Panel FE tỉnh | `artifacts/subnational/subnational_results.json` | Đã chạy | estimator = `linearmodels_panel_ols_entity_fe`, nobs = `315` |
| Phase 5 GRU | `artifacts/gru/vietnam_forecast_2024_2030.csv` | Đã có | Forecast 2024–2030 quanh `63.64` |
| Phase 6 RAG + LLM | `/insights/final` | Đã nối | Có recommendation fallback/LLM tùy cấu hình |

Kết quả final insight hiện tại:

```text
model_version  = xgboost_shap_runner+gru_forecast+subnational_drilldown
source         = shap_gap_analysis_csv+xgboost_shap_summary
year           = 2023
current_score  = 64.0963
observed_score = 63.7269
top weak       = n_sdg16_rsf, n_sdg16_justice, n_sdg16_exprop
province       = Hải Dương, score = 0.3772
forecast_count = 21
scenarios      = base, optimistic, pessimistic
forecast_years = 2024–2030
```

Như vậy, phần output cuối đã có đủ bốn nhóm thông tin trong sơ đồ: điểm Việt Nam hiện tại, chỉ số yếu nhất, tỉnh tệ nhất và dự báo 2030 theo ba kịch bản kèm khuyến nghị chính sách.

## 10. Kết luận

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

## 11. Cập nhật kiểm tra leakage, diagnostics và forecast ngày 2026-07-12

Mục này ghi lại 5 kiểm tra mới nhất theo yêu cầu rà soát luồng SDG16.

### 11.1 Panel OLS quốc gia

Phase 1 hiện dùng `linearmodels.PanelOLS` với country fixed effects và year fixed effects. Artifact chính là `artifacts/panel_ols/panel_ols_results.json`.

Kết quả Entity + Time FE:

| Chỉ tiêu | Giá trị |
|---|---:|
| R² overall | 0.6263 |
| R² within | 0.6297 |
| RMSE residual | 2.1276 |
| MAE residual | 1.6099 |
| Durbin-Watson panel mean | 2.2810 |
| Corr(|residual|, fitted) | -0.2238 |
| Heteroskedasticity flag | False |

Kiểm tra đa cộng tuyến cho thấy một số biến có VIF cao, đặc biệt `n_sdg16_homicide` khoảng 29.30 và `n_sdg16_crimepov` khoảng 19.57. Vì vậy Panel OLS nên được xem là baseline kinh tế lượng/diễn giải, không phải model dự đoán chính.

### 11.2 Panel cấp tỉnh đã sửa leakage

Panel FE cấp tỉnh trước đây dùng proxy PAPI/PCI cùng năm để dự đoán `goal16`, dễ gây leakage. Bản mới đã chuyển sang dùng biến trễ t-1:

```text
papi_score_lag1
pci_transparency_lag1
grdp_index_lag1
```

Kết quả sau khi sửa:

| Chỉ tiêu | Giá trị |
|---|---:|
| Estimator | linearmodels_panel_ols_entity_time_fe_lagged |
| Leakage control | uses_lagged_t_minus_1_features_only |
| Quan sát sau lag | 252 |
| Dòng bị loại do lag | 63 |
| R² overall | 0.0064 |

R² thấp hơn là hợp lý vì mô hình không còn dùng biến cùng năm có khả năng “nhìn trước” mục tiêu.

### 11.3 Breakdown PAPI/PCI cho Đắk Nông

Artifact `artifacts/subnational/subnational_results.json` đã có khóa `dak_nong_breakdown`. Năm mới nhất là 2023.

| Chiều | Giá trị Đắk Nông | Trung bình 63 tỉnh | Chênh lệch | Rank giảm dần |
|---|---:|---:|---:|---:|
| bribery_people | 5.4305 | 5.2642 | +0.1663 | 31 |
| admin_procedure | 3.2251 | 5.7474 | -2.5223 | 52 |
| vertical_accountability | 8.8367 | 5.7195 | +3.1172 | 9 |
| transparency | 5.0077 | 5.6081 | -0.6003 | 36 |
| citizen_participation | 5.6334 | 5.7103 | -0.0770 | 32 |
| papi_score | 44.3693 | 48.3502 | -3.9809 | 36 |
| pci_transparency | 67.7885 | 64.3429 | +3.4456 | 22 |
| grdp_index | 16540.9549 | 30981.9185 | -14440.9636 | 54 |
| goal16 | 0.7964 | 0.5777 | +0.2186 | 2 |

Điểm yếu nổi bật của Đắk Nông trong dữ liệu demo hiện tại là `admin_procedure`, `transparency`, `papi_score` và `grdp_index`.

### 11.4 GRU forecast đã hết phẳng

GRU forecast trước đây phẳng vì dùng lại vector feature cuối cùng cho mọi năm tương lai. Bản mới dùng recursive forecast với trend feature gần nhất của Việt Nam, damping theo thời gian và clamp theo phân vị 1%–99% của dữ liệu global.

| Năm | predicted_goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.8785 |
| 2026 | 64.1630 |
| 2027 | 64.4077 |
| 2028 | 64.6922 |
| 2029 | 64.9520 |
| 2030 | 65.1743 |

Forecast này vẫn là dự báo mô phỏng theo xu hướng feature, không phải dự báo chính thức của SDSN.

### 11.5 Correlation check cho XGBoost SHAP Runner

XGBoost SHAP Runner đã xuất thêm:

- `artifacts/shap/leakage_correlation_report.csv`
- `artifacts/shap/shap_summary.json` phần `leakage_correlation_report`

Kết quả kiểm tra:

| Kiểm tra | Kết quả |
|---|---|
| Exact duplicate feature với `goal16` | Không có |
| Feature có `abs(corr) >= 0.98` với `goal16` | Không có |
| Numeric column đáng ngờ có `abs(corr) >= 0.98` | Không có |
| Leakage status | no_exact_target_duplicate_detected |

Top tương quan cao nhất với `goal16` là `n_sdg16_cpi` khoảng 0.8206, tiếp theo là `n_sdg16_exprop` khoảng 0.6333 và `n_sdg16_detain` khoảng 0.5780. Đây là tương quan mạnh hợp lý giữa chỉ số thành phần và điểm tổng, chưa phải bằng chứng leakage.
