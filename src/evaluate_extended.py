import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, matthews_corrcoef, mean_absolute_error, 
                             mean_squared_error, precision_score, recall_score, f1_score)
from data_loader import load_data
from model import MultiTaskModel

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r"data"
MODEL_PATH = r"results/best_model.pth"
BATCH_SIZE = 128

def calculate_extended_metrics():
    print(f"Loading data and model on {DEVICE}...")
    # Load Test Data
    _, test_loader, input_dims = load_data(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
        BATCH_SIZE
    )
    
    model = MultiTaskModel(input_dims['cont'], input_dims['cat']).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model file not found. Please train first.")
        return

    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for x_cont, x_cat, y in test_loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            pred, _, _ = model(x_cont, x_cat)
            y_true.extend(y.cpu().numpy())
            y_scores.extend(pred.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)
    
    # 1. Basic Confusion Matrix Elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 2. Metrics
    sensitivity = tp / (tp + fn) # Recall
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    
    # Regression metrics on probabilities
    mae = mean_absolute_error(y_true, y_scores)
    mse = mean_squared_error(y_true, y_scores)
    rmse = np.sqrt(mse)
    
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("       EXTENDED RESEARCH METRICS REPORT       ")
    print("="*50)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Sensitivity (Re):{sensitivity:.4f}")
    print(f"Specificity:     {specificity:.4f}")
    print(f"MCC:             {mcc:.4f}")
    print(f"AUC:             {auc:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print("-" * 50)
    print(f"MAE:             {mae:.4f}")
    print(f"MSE:             {mse:.4f}")
    print(f"RMSE:            {rmse:.4f}")
    print("="*50)
    
    # Save to CSV for the user
    results = {
        "Metric": ["Accuracy", "Sensitivity", "Specificity", "MCC", "AUC", "F1-Score", "MAE", "MSE", "RMSE"],
        "Value": [accuracy, sensitivity, specificity, mcc, auc, f1, mae, mse, rmse]
    }
    pd.DataFrame(results).to_csv("results/extended_metrics.csv", index=False)
    print("Metrics saved to results/extended_metrics.csv")

if __name__ == "__main__":
    calculate_extended_metrics()
