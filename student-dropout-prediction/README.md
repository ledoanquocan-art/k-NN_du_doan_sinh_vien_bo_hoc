# Student Dropout & Academic Success Prediction 🎓

Đồ án môn Trí tuệ nhân tạo: Xây dựng mô hình AI dự đoán tình trạng học tập của sinh viên (Dropout, Enrolled, Graduate).

## 👥 Đội ngũ phát triển (Nhóm X)
1. Lê Doãn Quốc An (Nhóm trưởng) - Thiết lập Pipeline & Review Code
2. Nông Đức Mạnh - Data Processing & Engineering
3. Trần Ngọc Tuấn - Exploratory Data Analysis (EDA)
4. Trịnh Phan Tuấn Anh - Model Development & Training
5. Nguyễn Anh Đức - Evaluation & Documentation


student-dropout-prediction/
│
├── data/                  # Chứa dữ liệu (Thường sẽ không push file dữ liệu lớn lên Git)
│   ├── raw/               # Chứa file gốc: data.csv, predict+students+dropout+and+academic+success.zip
│   └── processed/         # Chứa dữ liệu đã qua tiền xử lý, chuẩn hóa
│
├── notebooks/             # Thư mục cho Jupyter Notebook (.ipynb) - Phục vụ nghiên cứu
│   ├── 01_EDA.ipynb       # Khai phá dữ liệu, vẽ biểu đồ
│   └── 02_Modeling.ipynb  # Thử nghiệm các thuật toán, tinh chỉnh tham số
│
├── src/                   # Thư mục chứa mã nguồn chính (.py) - Phục vụ chạy thực tế
│   ├── data_prep.py       # Script làm sạch, mã hóa nhãn (Label Encoding)
│   ├── train.py           # Script huấn luyện (chia tập 80-20)
│   └── evaluate.py        # Script test mô hình (chia 3 phần test độc lập)
│
├── docs/                  # Báo cáo Word, slide thuyết trình, biểu đồ kết xuất
├── requirements.txt       # Chứa tên các thư viện cần cài đặt (pandas, scikit-learn, numpy)
├── .gitignore             # File cấu hình ẩn các file rác
└── README.md              # Tài liệu giới thiệu dự án


## 🚀 Cài đặt & Sử dụng
**1. Cài đặt thư viện**
```bash
pip install -r requirements.txt
**2. Chạy quy trình huấn luyện & Đánh giá (Chia test 3 phần độc lập)**
```bash
python src/train.py
python src/evaluate.py