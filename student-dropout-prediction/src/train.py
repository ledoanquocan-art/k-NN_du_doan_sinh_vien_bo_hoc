import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_knn_model():
    print("--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH k-NN ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Yêu cầu người dùng nhập hệ số k
    while True:
        try:
            k_input = input("Chọn hệ số k: ")
            k = int(k_input)
            if k <= 0:
                print("Hệ số k phải là số nguyên lớn hơn 0. Vui lòng thử lại.")
                continue
            break
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ!")
    
    # Đọc dữ liệu
    df = pd.read_csv(os.path.join(processed_dir, 'cleaned_data.csv'))
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Chia Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    test_data = X_test.copy()
    test_data['Target'] = y_test
    test_data.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
    
    # Chuẩn hóa dữ liệu
    print(f"Đang chuẩn hóa dữ liệu và train k-NN (với k={k}) trên {len(X_train)} mẫu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    # Khởi tạo và Train thuật toán k-NN với k vừa nhập
    model = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Lưu file mô hình kèm theo hệ số k trong tên file
    model_filename = f'knn_model_k{k}.pkl'
    joblib.dump(model, os.path.join(models_dir, model_filename))
    print(f"Huấn luyện thành công! Mô hình đã được lưu với tên '{model_filename}'.\n")

if __name__ == "__main__":
    train_knn_model()