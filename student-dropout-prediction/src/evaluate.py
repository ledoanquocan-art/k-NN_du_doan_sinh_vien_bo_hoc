import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_model():
    print("--- BƯỚC 3: ĐÁNH GIÁ 3 LẦN TRÊN TẬP TEST ---")
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    
    # Load Model và Test Data
    model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
    test_data = pd.read_csv(os.path.join(processed_dir, 'test_data.csv'))
    
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']
    
    # Cắt DataFrame test thành 3 phần bằng nhau
    X_splits = np.array_split(X_test, 3)
    y_splits = np.array_split(y_test, 3)
    
    target_names = ['Dropout (0)', 'Enrolled (1)', 'Graduate (2)']
    
    for i in range(3):
        X_sub = X_splits[i]
        y_sub = y_splits[i]
        
        y_pred = model.predict(X_sub)
        acc = accuracy_score(y_sub, y_pred)
        
        print(f"\n================ [LẦN TEST {i+1}] ================")
        print(f"Số lượng mẫu kiểm tra: {len(y_sub)}")
        print(f"Độ chính xác tổng thể (Accuracy): {acc*100:.2f}%")
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_sub, y_pred, target_names=target_names))

if __name__ == "__main__":
    evaluate_model()