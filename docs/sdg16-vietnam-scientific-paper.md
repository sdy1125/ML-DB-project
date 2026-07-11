# ỨNG DỤNG HỌC MÁY GIẢI THÍCH VÀ RAG TRONG DỰ BÁO SDG16 CHO VIỆT NAM

**Nguyễn Văn A<sup>1</sup>, Trần Thị B<sup>1</sup>, Lê Văn C<sup>1</sup>**

<sup>1</sup>Khoa/Viện: ........................................................, Trường/Đơn vị: ........................................................  
**Tác giả liên hệ:** ........................................................; Email: ........................................................

## Tóm tắt

Nghiên cứu này đề xuất một khung phân tích kết hợp học máy giải thích được, dự báo chuỗi thời gian và truy xuất tăng cường sinh văn bản nhằm hỗ trợ đánh giá chỉ số SDG16 cho Việt Nam. Dữ liệu nghiên cứu được xây dựng từ bộ SDR2024, gồm 4.392 quan sát của 183 quốc gia với 17 chỉ số đầu vào thuộc nhóm SDG16 và biến mục tiêu là điểm `goal16`. Bốn mô hình được so sánh gồm Spark Linear Regression, XGBoost có giải thích đóng góp biến, XGBoost tinh chỉnh bổ sung và GRU dự báo chuỗi thời gian. Kết quả thực nghiệm cho thấy XGBoost có giải thích đạt hiệu suất tốt nhất trên tập kiểm tra với RMSE = 1,8249, MAE = 1,4019 và R² = 0,9859. Mô hình GRU đạt RMSE = 2,4411 và R² = 0,9747, phù hợp cho dự báo giai đoạn 2024–2030. Phân tích đóng góp biến cho Việt Nam cho thấy trách nhiệm giải trình, tiếp cận tư pháp, bảo vệ quyền tài sản, lao động trẻ em và minh bạch hành chính là các điểm nghẽn chính. Kết quả nghiên cứu cung cấp cơ sở định lượng để kết hợp với RAG và LLM trong sinh khuyến nghị chính sách có căn cứ.

**Từ khóa:** dự báo SDG16; giải thích mô hình; GRU; khuyến nghị chính sách; RAG; XGBoost.

## 1. Đặt vấn đề

Mục tiêu phát triển bền vững số 16 (SDG16) nhấn mạnh hòa bình, công lý, thể chế hiệu quả, minh bạch, trách nhiệm giải trình và khả năng tiếp cận pháp lý của người dân. Đây là nhóm mục tiêu có vai trò nền tảng vì chất lượng thể chế không chỉ ảnh hưởng đến quản trị công mà còn liên quan đến năng lực cạnh tranh, thu hút đầu tư và phát triển bền vững dài hạn. Tuy nhiên, việc đánh giá SDG16 thường gặp khó khăn do các chỉ số thành phần có bản chất đa chiều, có quan hệ phi tuyến và chịu ảnh hưởng bởi bối cảnh thể chế của từng quốc gia.

Các phương pháp theo dõi truyền thống thường dừng ở bảng xếp hạng, thống kê mô tả hoặc so sánh điểm số giữa các quốc gia. Cách tiếp cận này hữu ích cho việc nhận diện vị trí tương đối nhưng chưa trả lời đầy đủ các câu hỏi quan trọng trong hoạch định chính sách: chỉ số nào đang kéo điểm quốc gia xuống, điểm số có thể thay đổi như thế nào trong tương lai, và khuyến nghị chính sách nào có bằng chứng hỗ trợ từ tài liệu thực tế. Vì vậy, cần một khung phân tích có khả năng kết hợp dự đoán, giải thích, dự báo và truy xuất bằng chứng.

Nghiên cứu này hướng đến xây dựng một quy trình phân tích SDG16 cho Việt Nam dựa trên học máy giải thích được và truy xuất tăng cường sinh văn bản. Cụ thể, nghiên cứu so sánh nhiều mô hình dự đoán điểm `goal16`, lựa chọn mô hình có hiệu suất tốt nhất, phân tích các chỉ số ảnh hưởng bất lợi đến Việt Nam, dự báo xu hướng đến năm 2030 và đề xuất cách kết nối kết quả định lượng với hệ thống RAG + LLM để sinh khuyến nghị chính sách. Phần còn lại của bài báo gồm: mục 2 trình bày phương pháp nghiên cứu; mục 3 trình bày kết quả thực nghiệm; mục 4 bàn luận ý nghĩa của kết quả; mục 5 nêu kết luận và hướng phát triển.

## 2. Phương pháp nghiên cứu

### 2.1. Dữ liệu nghiên cứu

Dữ liệu chính được sử dụng trong nghiên cứu là bộ SDR2024 sau khi lọc các quan sát phù hợp cho bài toán SDG16. Bộ dữ liệu gồm 4.392 quan sát thuộc 183 quốc gia. Biến đầu vào là 17 chỉ số đã chuẩn hóa thuộc nhóm `n_sdg16_*`; biến mục tiêu là điểm tổng hợp `goal16`.

**Bảng 1: Mô tả dữ liệu nghiên cứu**

| Thành phần | Mô tả |
|---|---|
| Nguồn dữ liệu | SDR2024 |
| Số quan sát | 4.392 |
| Số quốc gia | 183 |
| Biến đầu vào | 17 chỉ số chuẩn hóa dạng `n_sdg16_*` |
| Biến mục tiêu | `goal16` |
| Số quan sát huấn luyện | 3.477 |
| Số quan sát kiểm định | 549 |
| Số quan sát kiểm tra | 366 |

Các chỉ số đầu vào phản ánh nhiều khía cạnh của SDG16 như phòng chống tham nhũng, đăng ký khai sinh, tạm giam trước xét xử, tử vong do bạo lực, lao động trẻ em, tiếp cận tư pháp, minh bạch hành chính và trách nhiệm giải trình.

### 2.2. Thiết kế mô hình

Nghiên cứu so sánh bốn nhóm mô hình. Spark Linear Regression được dùng làm mô hình nền nhằm kiểm tra mức độ giải thích tuyến tính của các chỉ số. XGBoost có giải thích đóng góp biến được dùng làm mô hình phi tuyến chính vì có khả năng học quan hệ phức tạp giữa các chỉ số. Một pipeline XGBoost tinh chỉnh bổ sung được dùng để đối chiếu hiệu suất. GRU Sequence Forecaster được dùng cho nhiệm vụ dự báo chuỗi thời gian.

Với mô hình XGBoost chính, dữ liệu được chia theo thời gian để hạn chế rò rỉ thông tin: các quan sát đến năm 2018 được dùng để huấn luyện, giai đoạn 2019–2021 dùng để kiểm định và các quan sát từ năm 2022 trở đi dùng để kiểm tra. Với GRU, mỗi mẫu đầu vào gồm chuỗi 5 năm liên tiếp của các chỉ số SDG16, đầu ra là điểm `goal16` của năm kế tiếp.

### 2.3. Chỉ số đánh giá

Hiệu suất mô hình được đánh giá bằng ba chỉ số hồi quy: RMSE, MAE và R². RMSE phản ánh sai số bình phương trung bình sau khi lấy căn bậc hai; MAE phản ánh sai số tuyệt đối trung bình; R² phản ánh tỷ lệ phương sai của biến mục tiêu được mô hình giải thích. Trong nghiên cứu này, mô hình tốt hơn là mô hình có RMSE và MAE thấp hơn, đồng thời có R² cao hơn.

### 2.4. Giải thích mô hình và tích hợp RAG

Sau khi chọn mô hình tốt nhất, nghiên cứu sử dụng đóng góp biến từ XGBoost để xác định các chỉ số kéo điểm SDG16 của Việt Nam xuống. Các đóng góp âm được xem là tín hiệu ưu tiên chính sách vì chúng làm giảm dự đoán điểm tổng hợp. Kết quả này không được xem là bằng chứng nhân quả tuyệt đối, mà là căn cứ giải thích hành vi của mô hình và định hướng truy xuất tài liệu.

Ở tầng ứng dụng, kết quả học máy được kết hợp với hệ thống RAG + LLM. Hệ thống RAG truy xuất các đoạn tài liệu chính sách, báo cáo quốc tế, văn bản pháp lý và nghiên cứu liên quan. LLM sau đó sử dụng đồng thời kết quả dự đoán, chỉ số yếu, dự báo tương lai và bằng chứng truy xuất để sinh khuyến nghị chính sách.

## 3. Kết quả

### 3.1. So sánh hiệu suất mô hình

Kết quả so sánh hiệu suất của bốn mô hình được trình bày ở Bảng 2.

**Bảng 2: So sánh hiệu suất các mô hình dự đoán SDG16**

| Mô hình | Tập đánh giá | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| XGBoost SHAP Runner | Test | 1,8249 | 1,4019 | 0,9859 |
| GRU Sequence Forecaster | Test | 2,4411 | 1,9119 | 0,9747 |
| Optional XGBoost Tuned Pipeline | Overall | 7,7773 | 4,6659 | 0,6974 |
| Spark Linear Regression | Test | 12,2139 | 9,2173 | 0,3509 |

Mô hình XGBoost SHAP Runner đạt RMSE = 1,8249, MAE = 1,4019 và R² = 0,9859 trên tập kiểm tra. Mô hình GRU Sequence Forecaster đạt RMSE = 2,4411, MAE = 1,9119 và R² = 0,9747. Optional XGBoost Tuned Pipeline đạt RMSE = 7,7773, MAE = 4,6659 và R² = 0,6974. Spark Linear Regression đạt RMSE = 12,2139, MAE = 9,2173 và R² = 0,3509.

### 3.2. Cấu hình XGBoost tốt nhất

Mô hình XGBoost chính được tinh chỉnh qua 28 cấu hình. Cấu hình được chọn gồm `max_depth = 4`, `learning_rate = 0,09`, `n_estimators = 850`, `subsample = 0,82`, `colsample_bytree = 0,82`, `reg_alpha = 0,5`, `reg_lambda = 1,0`, `tree_method = hist` và `objective = reg:squarederror`. Trên tập kiểm định, mô hình đạt RMSE = 1,1886, MAE = 0,8502 và R² = 0,9937.

### 3.3. Các chỉ số kéo điểm Việt Nam xuống

Bảng 3 trình bày năm chỉ số có đóng góp âm lớn nhất đối với dự đoán điểm SDG16 của Việt Nam.

**Bảng 3: Các chỉ số kéo điểm SDG16 của Việt Nam xuống theo đóng góp biến**

| Chỉ số | Diễn giải | Đóng góp |
|---|---|---:|
| `n_sdg16_rsf` | Tự do báo chí / trách nhiệm giải trình | -2,3477 |
| `n_sdg16_justice` | Tiếp cận tư pháp | -1,9228 |
| `n_sdg16_exprop` | Bảo vệ quyền tài sản / chống tịch thu tài sản | -1,3962 |
| `n_sdg16_clabor` | Lao động trẻ em | -0,5346 |
| `n_sdg16_admin` | Hành chính minh bạch | -0,3568 |

Ngoài ra, ở mức độ quan trọng toàn cục, các chỉ số có đóng góp tuyệt đối trung bình cao gồm `n_sdg16_cpi`, `n_sdg16_u5reg`, `n_sdg16_detain`, `n_sdg16_homicides` và `n_sdg16_clabor`.

### 3.4. Dự báo điểm SDG16 của Việt Nam giai đoạn 2024–2030

Bảng 4 trình bày kết quả dự báo của mô hình GRU cho Việt Nam trong giai đoạn 2024–2030.

**Bảng 4: Dự báo điểm Goal16 của Việt Nam giai đoạn 2024–2030**

| Năm | Điểm Goal16 dự báo |
|---:|---:|
| 2024 | 63,6456 |
| 2025 | 63,6389 |
| 2026 | 63,6521 |
| 2027 | 63,6432 |
| 2028 | 63,6419 |
| 2029 | 63,6419 |
| 2030 | 63,6419 |

Kết quả dự báo cho thấy điểm `goal16` của Việt Nam trong giai đoạn 2024–2030 dao động quanh mức 63,64 trong điều kiện mô hình học từ xu hướng lịch sử của bộ chỉ số đầu vào.

## 4. Bàn luận

Kết quả thực nghiệm cho thấy mô hình XGBoost có giải thích đóng góp biến đạt hiệu suất tốt nhất trong nhóm mô hình được so sánh. Điều này phù hợp với đặc điểm của dữ liệu SDG16, vì các chỉ số thể chế thường có quan hệ phi tuyến và có thể tương tác với nhau. Trong khi Linear Regression chỉ mô hình hóa quan hệ tuyến tính, XGBoost có khả năng chia tách không gian dữ liệu theo nhiều ngưỡng khác nhau, nhờ đó nắm bắt tốt hơn sự khác biệt giữa các quốc gia và giữa các giai đoạn.

Mô hình GRU không vượt XGBoost về sai số dự đoán trên tập kiểm tra, nhưng có vai trò riêng trong bài toán dự báo. Do GRU xử lý chuỗi quan sát theo thời gian, mô hình này phù hợp để ước lượng xu hướng điểm SDG16 trong các năm tiếp theo. Kết quả dự báo cho Việt Nam cho thấy nếu cấu trúc các chỉ số đầu vào không thay đổi đáng kể, điểm SDG16 có thể duy trì quanh mức 63,64 đến năm 2030. Điều này gợi ý rằng cải thiện điểm số cần đến các can thiệp chính sách có mục tiêu, thay vì chỉ kỳ vọng vào xu hướng tự nhiên.

Phân tích đóng góp biến cho thấy các điểm nghẽn chính của Việt Nam nằm ở nhóm trách nhiệm giải trình, tiếp cận tư pháp, bảo vệ quyền tài sản, lao động trẻ em và minh bạch hành chính. Đây đều là những khía cạnh có liên quan trực tiếp đến chất lượng thể chế. Kết quả này có ý nghĩa thực tiễn vì nó chuyển bài toán từ “Việt Nam đạt bao nhiêu điểm” sang “Việt Nam nên ưu tiên cải thiện chỉ số nào”. Khi kết hợp với RAG, các chỉ số yếu này có thể trở thành truy vấn định hướng để tìm kiếm bằng chứng chính sách từ báo cáo, luật, nghị quyết và tài liệu quốc tế.

Một điểm mạnh của khung nghiên cứu là sự kết hợp giữa dự đoán và giải thích. Mô hình XGBoost cung cấp độ chính xác cao, trong khi đóng góp biến giúp diễn giải kết quả theo từng chỉ số. GRU bổ sung góc nhìn tương lai, còn RAG + LLM giúp chuyển hóa kết quả định lượng thành khuyến nghị có căn cứ tài liệu. Tuy nhiên, kết quả đóng góp biến không nên được diễn giải như quan hệ nhân quả tuyệt đối. Các chỉ số SDG16 chịu ảnh hưởng bởi phương pháp đo lường, nguồn dữ liệu và bối cảnh quốc gia. Do đó, khuyến nghị chính sách cuối cùng cần kết hợp thêm thẩm định chuyên gia và phân tích định tính.

## 5. Kết luận

Nghiên cứu cho thấy XGBoost có giải thích đóng góp biến là mô hình phù hợp nhất trong thử nghiệm dự đoán SDG16, với RMSE = 1,8249 và R² = 0,9859; đối với Việt Nam, các ưu tiên cải thiện nên tập trung vào trách nhiệm giải trình, tiếp cận tư pháp, bảo vệ quyền tài sản, lao động trẻ em và minh bạch hành chính, đồng thời kết hợp RAG + LLM để sinh khuyến nghị chính sách dựa trên bằng chứng.

## Abstract

This study proposes an explainable machine learning and retrieval-augmented generation framework to support SDG16 performance prediction and policy recommendation for Vietnam. The empirical analysis uses the SDR2024 dataset, including 4,392 observations from 183 countries, 17 normalized SDG16 indicators, and `goal16` as the target variable. Four models are compared: Spark Linear Regression, an explainable XGBoost model, an additional tuned XGBoost pipeline, and a GRU sequence forecaster. The explainable XGBoost model achieves the best test performance, with RMSE = 1.8249, MAE = 1.4019, and R² = 0.9859. The GRU model obtains RMSE = 2.4411 and R² = 0.9747, making it useful for forecasting Vietnam's SDG16 trajectory from 2024 to 2030. Model contribution analysis indicates that press freedom/accountability, access to justice, protection against expropriation, child labor, and administrative transparency are the main negative contributors to Vietnam's predicted SDG16 score. The proposed framework connects these quantitative outputs with a RAG + LLM recommendation layer, enabling evidence-based policy generation from reports, legal documents, and policy texts. The findings demonstrate that combining prediction, explainability, forecasting, and document retrieval can transform SDG16 monitoring from static benchmarking into actionable policy intelligence.

**Keywords:** explainable AI; GRU; policy recommendation; RAG; SDG16; XGBoost.

## Tài liệu tham khảo

1. Sachs, J. D., Lafortune, G., Fuller, G., & Drumm, E. (2024). *Sustainable Development Report 2024*. SDSN and Dublin University Press.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
4. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of EMNLP 2014*, 1724–1734.
5. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.
6. World Bank. (2024). *Taking Stock: Vietnam Economic Growth Update*. World Bank.
7. International Monetary Fund. (2024). *Vietnam: 2024 Article IV Consultation*. IMF Country Report.
8. United Nations. (2024). *United Nations E-Government Survey 2024*. United Nations.
9. Government of Vietnam. (2020). *National Digital Transformation Program to 2025, orientation to 2030*. Decision No. 749/QĐ-TTg.
