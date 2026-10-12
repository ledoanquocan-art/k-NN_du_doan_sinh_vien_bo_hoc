import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_nb_model():
    print("--- BƯỚC 3: ĐÁNH GIÁ NAIVE BAYES TRÊN 3 BỘ TEST ĐỘC LẬP ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')

    # 1. Load Model và Scaler
    model_path = os.path.join(models_dir, 'nb_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_nb.pkl')
    
    if not os.path.exists(model_path):
        print("Chưa có mô hình Naive Bayes! Vui lòng chạy train_naive_bayes.py trước.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Đọc tập Test
    test_data = pd.read_csv(os.path.join(processed_dir, 'test_data_nb.csv'))
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # 3. Chuẩn hóa tập Test
    X_test_scaled = scaler.transform(X_test)

    # 4. Chia tập test thành 3 phần
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
    evaluate_nb_model()