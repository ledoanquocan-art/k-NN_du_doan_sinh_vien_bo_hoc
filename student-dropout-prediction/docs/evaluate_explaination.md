# Giải thích chi tiết kết quả chạy `evaluate.py`
*Tài liệu ôn tập cốt lõi chuẩn bị cho hội đồng bảo vệ đồ án*

---

## 1. Các chỉ số đánh giá tổng quan

*   **Số lượng mẫu (295):** Kích thước của $1$ trong $3$ tập test độc lập mà nhóm bạn đã chia ra.
*   **Accuracy (66.44%):** Độ chính xác tổng thể. Trong $295$ sinh viên của tập test này, mô hình dự đoán đúng trạng thái của khoảng $196$ sinh viên (tương đương $66.44\%$).
*   **PR-AUC (Dropout) = 0.7933:** Diện tích dưới đường cong Precision-Recall riêng cho lớp "Bỏ học". Chỉ số này chạy từ $0$ đến $1$. Mức $0.7933$ cho thấy mô hình có khả năng nhận diện nhóm sinh viên bỏ học khá tốt, bất chấp việc số lượng sinh viên bỏ học ít hơn sinh viên tốt nghiệp.

---

## 2. Báo cáo phân loại (Classification Report)

Báo cáo này "mổ xẻ" hiệu suất của mô hình trên từng lớp cụ thể thay vì chỉ nhìn vào Accuracy tổng.

*   **Precision (Độ chính xác của dự đoán):** Trả lời câu hỏi *"Trong những sinh viên mô hình dự đoán là A, có bao nhiêu phần trăm thực sự là A?"*.
    *   *Ví dụ lớp Dropout đạt 0.81:* Nghĩa là khi mô hình "phát cảnh báo" $100$ sinh viên sẽ bỏ học, thì có $81$ bạn thực sự bỏ học ($19$ bạn bị mô hình nghi ngờ oan).
*   **Recall (Độ nhạy / Độ bao phủ):** Trả lời câu hỏi *"Trong tổng số sinh viên thực sự là A, mô hình đã tìm ra được bao nhiêu phần trăm?"*.
    *   *Ví dụ lớp Dropout đạt 0.61:* Nghĩa là trong $100$ sinh viên thực tế bỏ học, hệ thống chỉ "bắt" được $61$ bạn (bỏ sót mất $39$ bạn).
*   **F1-score:** Là điểm trung bình hài hòa giữa Precision và Recall. Nếu một trong hai chỉ số kia quá thấp, F1 sẽ bị kéo tụt xuống. Điểm F1 $0.69$ cho lớp Dropout là mức trung bình khá.
*   **Support:** Số lượng sinh viên thực tế của mỗi nhóm nằm trong $295$ mẫu này (Có $97$ bạn Dropout, $52$ bạn Enrolled, $146$ bạn Graduate).
*   **macro avg:** Điểm trung bình cộng đơn thuần của $3$ lớp, không quan tâm lớp nào đông hơn lớp nào. 
    *   *Công thức minh họa:* $\text{F1 macro} = \frac{0.69 + 0.24 + 0.77}{3} = 0.57$.
*   **weighted avg:** Điểm trung bình được nhân với trọng số là tỷ lệ số lượng mẫu (Support) của từng lớp. Do lớp Graduate chiếm đông nhất ($146$ mẫu) và có F1 cao ($0.77$), nó kéo F1 weighted avg lên $0.65$.

---

## 3. Ma trận nhầm lẫn (Confusion Matrix)

Đây là bảng soi chiếu chi tiết nhất để biết mô hình đang gặp khó khăn ở đâu. Các hàng ngang đại diện cho kết quả **Thực tế**, các cột dọc đại diện cho kết quả **Dự đoán**. Các số trên đường chéo chính (từ trên cùng bên trái xuống dưới cùng bên phải) là số lần đoán đúng.

*   **Dòng 1 (Thực tế là Dropout - 97 bạn):** `[ 59  14  24 ]`
    *   Mô hình đoán đúng $59$ bạn.
    *   Đoán nhầm $14$ bạn thành Đang học (`Enrolled`).
    *   Đoán nhầm $24$ bạn thành Tốt nghiệp (`Graduate`).
*   **Dòng 2 (Thực tế là Enrolled - 52 bạn):** `[ 10  11  31 ]`
    *   Mô hình đoán đúng được $11$ bạn.
    *   Đoán nhầm $10$ bạn thành Bỏ học (`Dropout`).
    *   Đoán nhầm tới $31$ bạn thành Tốt nghiệp (`Graduate`). *(Chỉ số Precision/Recall $0.27/0.21$ thấp ở trên chính là do hệ thống bị bối rối bởi nhóm này).*
*   **Dòng 3 (Thực tế là Graduate - 146 bạn):** `[  4  16 126 ]`
    *   Mô hình đoán đúng $126$ bạn (rất tốt).
    *   Đoán nhầm $4$ bạn thành Bỏ học (`Dropout`).
    *   Đoán nhầm $16$ bạn thành Đang học (`Enrolled`).