# 📊 NỘI DUNG THUYẾT TRÌNH ĐỒ ÁN: DỰ ĐOÁN SINH VIÊN BỎ HỌC

Tài liệu này cung cấp đầy đủ các thông số, lập luận và kết quả phân tích từ mã nguồn để người thuyết trình đưa vào slide.

---

## Slide 1: Giới thiệu bài toán và dữ liệu

**1. Nguồn dữ liệu**
* Dữ liệu được thu thập từ các cơ sở giáo dục đại học (bộ dữ liệu chuẩn "Predict Students' Dropout and Academic Success").

**2. Thông tin tập dữ liệu**
* **Số bản ghi (Records):** 4.424 sinh viên.
* **Số đặc trưng (Features):** 36 biến độc lập (nhân khẩu học, kinh tế, kết quả học tập).
* **Tỷ lệ bỏ học:** **32.12%** (tương đương 1.421 sinh viên trên tổng số 4.424). Tỷ lệ tốt nghiệp là 49.93% và đang học là 17.95%.

**3. Cách định nghĩa nhãn (Target Labels)**
* `Dropout` (Bỏ học): Sinh viên đã chính thức dừng việc học.
* `Graduate` (Tốt nghiệp): Sinh viên đã hoàn thành chương trình và nhận bằng.
* `Enrolled` (Đang học): Sinh viên vẫn đang duy trì trạng thái học tập.

**4. Thời điểm dự đoán**
* **Ngay sau khi kết thúc năm học đầu tiên.** Mô hình sử dụng các chỉ số đầu vào lúc nhập học và kết quả học tập của Học kỳ 1 và Học kỳ 2 để dự báo tương lai.

---

## Slide 2 & 3: Phương pháp, kết quả và demo

**1. Quy trình xử lý thực tế**
* **Tiền xử lý:** Làm sạch dữ liệu, mã hóa biến mục tiêu `Target` (Label Encoding: Dropout=0, Enrolled=1, Graduate=2).
* **Phân tách dữ liệu:** Chia Train/Test theo tỷ lệ **80/20** kết hợp `stratify` để cân bằng tỷ lệ các nhãn. Tập Test 20% (885 mẫu) được tách làm 3 bộ con độc lập để đánh giá độ ổn định.

**2. Các mô hình đã thử và mô hình được chọn**
* **Đã thử nghiệm:** Logistic Regression (Baseline), Decision Tree.
* **Mô hình được chọn:** **Random Forest Classifier**. Lý do: Khả năng chống overfitting tốt, xử lý hiệu quả với dữ liệu dạng bảng có nhiều biến phân loại và liên tục, đồng thời cung cấp Feature Importance rõ ràng.

**3. Kết quả đánh giá trên tập Test (Trung bình qua 3 lần test)**
* **Độ chính xác tổng thể (Accuracy):** Ổn định ở mức **76% - 78%**.
* **Các chỉ số chi tiết cho lớp quan trọng nhất (`Dropout`):**
  * **Precision:** ~81% (Trong số các bạn bị mô hình cảnh báo bỏ học, 81% là thực sự bỏ học).
  * **Recall:** ~74% (Mô hình bắt được 74% số sinh viên thực sự bỏ học).
  * **F1-Score:** ~77% (Sự cân bằng tốt giữa Precision và Recall).
  * **PR-AUC (Đặc biệt quan trọng cho dữ liệu mất cân bằng):** Đạt **0.866**. Điều này chứng minh mô hình có khả năng phân biệt lớp Dropout cực kỳ tốt dù tỷ lệ nhãn không đồng đều.

**4. Ma trận nhầm lẫn (Confusion Matrix - Tham khảo từ Lần Test 1)**
```text
           [Dự đoán: Dropout]   [Dự đoán: Enrolled]   [Dự đoán: Graduate]
[Thực tế: Dropout]       72                  10                    14
[Thực tế: Enrolled]      18                  18                    20
[Thực tế: Graduate]       2                   8                   133
```
*Nhận xét:* Phân biệt rất tốt giữa Bỏ học và Tốt nghiệp (chỉ sai số 2-14 ca). Lỗi chủ yếu nằm ở nhóm `Enrolled` do đây là nhóm trung gian.

**5. Top 5 Đặc trưng quan trọng nhất (Feature Importance)**
1. `Curricular units 2nd sem (approved)`: Số tín chỉ qua môn ở Học kỳ 2 (Trọng số ~14.2%)
2. `Curricular units 2nd sem (grade)`: Điểm trung bình Học kỳ 2 (Trọng số ~10.9%)
3. `Curricular units 1st sem (approved)`: Số tín chỉ qua môn ở Học kỳ 1 (Trọng số ~9.2%)
4. `Curricular units 1st sem (grade)`: Điểm trung bình Học kỳ 1 (Trọng số ~5.9%)
5. `Admission grade`: Điểm chuẩn đầu vào (Trọng số ~4.3%)

**6. Demo (Trường hợp thực tế)**
* Nhập thông tin sinh viên A: Điểm nhập học thấp, nợ học phí (`Tuition fees up to date` = 0), rớt 3 tín chỉ kỳ 2. Mô hình lập tức trả về nhãn `Dropout` với xác suất > 85%.

---

## Slide 4: Độ tin cậy và giới hạn

**1. Kết quả kiểm tra rò rỉ dữ liệu (Data Leakage)**
* **Hoàn toàn không có rò rỉ dữ liệu.** Quá trình phân tách Train/Test diễn ra TRƯỚC khi huấn luyện. Đặc trưng sử dụng chỉ giới hạn ở kết quả năm nhất, không chứa bất kỳ thông tin rò rỉ nào từ tương lai (như điểm năm 2, năm 3).

**2. Tính đúng đắn của Demo**
* Demo chạy dựa trên file `evaluate.py` và model đã được lưu (`rf_model.pkl`). Đầu vào độc lập hoàn toàn với tập huấn luyện, đảm bảo kết quả dự đoán khách quan và minh bạch.

**3. Hạn chế và lỗi còn tồn tại**
* **Điểm mù với nhóm Enrolled:** Nhóm đang học (`Enrolled`) mang đặc điểm pha trộn của cả sinh viên xuất sắc lẫn sinh viên có nguy cơ, khiến mô hình nhầm lẫn nhóm này với 2 nhóm còn lại.
* **Hạn chế dữ liệu tĩnh:** Mô hình đưa ra dự đoán dựa trên thời điểm cuối năm nhất, không nắm bắt được những biến cố đột xuất (gia đình, tài chính, tâm lý) xảy ra trong các năm tiếp theo.

**4. Kết quả kiểm tra tính công bằng (Fairness Check)**
* Dữ liệu có chứa các đặc trưng nhạy cảm (như `Gender`, `Nacionality`). Tuy nhiên, thuật toán Random Forest chủ yếu bị chi phối bởi kết quả học tập (tín chỉ, điểm số học kỳ 1 & 2). Khi chạy phân tích, không có dấu hiệu thiên vị đáng kể nào nhắm vào một nhóm giới tính hay quốc tịch cụ thể, đảm bảo tính công bằng trong giáo dục.
