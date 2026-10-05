import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_svm_model():
    print("--- BƯỚC 3: ĐÁNH GIÁ SVM TRÊN 3 BỘ TEST ĐỘC LẬP ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')

    # 1. Tìm tất cả các file mô hình SVM đã lưu
    model_files = glob.glob(os.path.join(models_dir, 'svm_model_*.pkl'))

    if not model_files:
        print("Chưa có mô hình SVM nào được huấn luyện. Vui lòng chạy file train_svm.py trước!")
        return

    # Liệt kê các model cho người dùng
    print("\nDanh sách các mô hình SVM hiện có trong thư mục:")
    available_models = {}
    for idx, file in enumerate(model_files):
        filename = os.path.basename(file)
        # Bóc tách kernel và C từ tên file (Ví dụ: svm_model_rbf_C1.0.pkl)
        parts = filename.replace('svm_model_', '').replace('.pkl', '')
        available_models[str(idx + 1)] = (file, parts)
        print(f"  [{idx + 1}] {parts}")

    # 2. Yêu cầu chọn model để test
    while True:
        selected = input("\nChọn số thứ tự model muốn test: ").strip()
        if selected in available_models:
            break
        print(f"Lựa chọn không hợp lệ! Vui lòng nhập số từ 1 đến {len(available_models)}.")

    model_path, model_desc = available_models[selected]

    # Load Model đã chọn và Scaler
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(models_dir, 'scaler_svm.pkl'))

    print(f"\n=> Đang đánh giá mô hình SVM ({model_desc})...")

    # Đọc tập Test
    test_data = pd.read_csv(os.path.join(processed_dir, 'test_data_svm.csv'))
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # Chuẩn hóa tập Test
    X_test_scaled = scaler.transform(X_test)

    # Chia tập test thành 3 phần
    X_splits = np.array_split(X_test_scaled, 3)
    y_splits = np.array_split(y_test, 3)
    y_bin_splits = np.array_split(y_test_bin, 3)

    target_names = ['Dropout (0)', 'Enrolled (1)', 'Graduate (2)']

    for i in range(3):
        X_sub = X_splits[i]
        y_sub = y_splits[i]
        y_bin_sub = y_bin_splits[i]

        y_pred = model.predict(X_sub)
        y_proba = model.predict_proba(X_sub)

        acc = accuracy_score(y_sub, y_pred)
        pr_auc_dropout = average_precision_score(y_bin_sub[:, 0], y_proba[:, 0])

        print(f"\n================ [LẦN TEST {i+1}] ================")
        print(f"Số lượng mẫu: {len(y_sub)} | Accuracy: {acc*100:.2f}% | PR-AUC (Dropout): {pr_auc_dropout:.4f}")

        print("\n1. Báo cáo phân loại (Precision, Recall, F1):")
        print(classification_report(y_sub, y_pred, target_names=target_names))

        print("2. Ma trận nhầm lẫn (Confusion Matrix):")
        print(confusion_matrix(y_sub, y_pred))

if __name__ == "__main__":
    evaluate_svm_model()
