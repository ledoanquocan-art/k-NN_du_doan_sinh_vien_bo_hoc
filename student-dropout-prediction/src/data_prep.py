import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Xác định đường dẫn thư mục gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_data():
    print("--- BƯỚC 1: XỬ LÝ DỮ LIỆU ---")
    raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'data.csv')
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    try:
        # Sử dụng dấu chấm phẩy làm separator
        df = pd.read_csv(raw_path, sep=';')
    except FileNotFoundError:
        print(f"Không tìm thấy file tại: {raw_path}. Vui lòng kiểm tra lại!")
        return
    
    # Mã hóa nhãn (Dropout: 0, Enrolled: 1, Graduate: 2)
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    
    # In thông tin ánh xạ nhãn để team nắm rõ
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Bảng mã hóa Target: {mapping}")
    
    # Lưu file đã xử lý
    out_path = os.path.join(processed_dir, 'cleaned_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Hoàn thành! Đã lưu file chuẩn hóa tại: {out_path}\n")

if __name__ == "__main__":
    prepare_data()