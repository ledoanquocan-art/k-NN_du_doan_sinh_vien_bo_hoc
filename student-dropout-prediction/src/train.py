import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    print("--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Đọc dữ liệu đã chuẩn hóa
    df = pd.read_csv(os.path.join(processed_dir, 'cleaned_data.csv'))
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Chia Train (80%) và Test (20%), sử dụng stratify để giữ nguyên tỷ lệ nhãn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tách 20% Test (xấp xỉ 885 dòng) ra file test_data.csv để T.Viên 5 xử lý
    test_data = X_test.copy()
    test_data['Target'] = y_test
    test_data.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
    
    # Cấu hình và Huấn luyện Random Forest
    print(f"Đang huấn luyện mô hình với {len(X_train)} mẫu (80% dữ liệu)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Lưu file mô hình
    joblib.dump(model, os.path.join(models_dir, 'rf_model.pkl'))
    print("Huấn luyện thành công! Mô hình đã lưu vào thư mục 'models'.\n")

if __name__ == "__main__":
    train_model()