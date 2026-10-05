import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_svm_model():
    print("--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH SVM ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Chọn kernel cho SVM
    valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    print(f"Các kernel hỗ trợ: {valid_kernels}")
    while True:
        kernel_input = input("Chọn kernel cho SVM (mặc định: rbf): ").strip().lower()
        if kernel_input == '':
            kernel_input = 'rbf'
        if kernel_input in valid_kernels:
            break
        print(f"Kernel không hợp lệ! Vui lòng chọn một trong: {valid_kernels}")

    # 2. Nhập tham số C (regularization)
    while True:
        try:
            c_input = input("Nhập tham số C (regularization, mặc định: 1.0): ").strip()
            C = float(c_input) if c_input != '' else 1.0
            if C <= 0:
                print("Tham số C phải là số dương. Vui lòng thử lại.")
                continue
            break
        except ValueError:
            print("Vui lòng nhập một số thực hợp lệ!")

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
    test_data.to_csv(os.path.join(processed_dir, 'test_data_svm.csv'), index=False)

    # Chuẩn hóa dữ liệu
    print(f"Đang chuẩn hóa dữ liệu và train SVM (kernel={kernel_input}, C={C}) trên {len(X_train)} mẫu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_svm.pkl'))

    # Khởi tạo và Train SVM
    model = SVC(kernel=kernel_input, C=C, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Lưu file mô hình kèm theo kernel và C trong tên file
    model_filename = f'svm_model_{kernel_input}_C{C}.pkl'
    joblib.dump(model, os.path.join(models_dir, model_filename))
    print(f"Huấn luyện thành công! Mô hình đã được lưu với tên '{model_filename}'.\n")

if __name__ == "__main__":
    train_svm_model()
