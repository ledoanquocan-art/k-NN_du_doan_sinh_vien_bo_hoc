import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_nb_model():
    print("--- HUẤN LUYỆN MÔ HÌNH NAIVE BAYES ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Đọc dữ liệu
    df = pd.read_csv(os.path.join(processed_dir, 'cleaned_data.csv'))
    X = df.drop('Target', axis=1)
    y = df['Target']

    # 2. Chia Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_data = X_test.copy()
    test_data['Target'] = y_test
    test_data.to_csv(os.path.join(processed_dir, 'test_data_nb.csv'), index=False)

    # 3. Chuẩn hóa dữ liệu (Tùy chọn với Naive Bayes, nhưng giữ để đồng bộ)
    print(f"Đang chuẩn hóa và train Gaussian Naive Bayes trên {len(X_train)} mẫu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_nb.pkl'))

    # 4. Khởi tạo và Train thuật toán Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    # 5. Lưu mô hình
    model_filename = 'nb_model.pkl'
    joblib.dump(model, os.path.join(models_dir, model_filename))
    print(f"Huấn luyện thành công! Mô hình đã được lưu với tên '{model_filename}'.\n")

if __name__ == "__main__":
    train_nb_model()