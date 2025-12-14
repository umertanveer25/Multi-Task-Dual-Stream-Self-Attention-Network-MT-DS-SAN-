import torch
import torch.nn as nn
from data_loader import load_data
from model import MultiTaskModel
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r"data" 
MODEL_PATH = r"results/best_model.pth"
BATCH_SIZE = 128

def evaluate_metrics():
    print(f"Loading data and model on {DEVICE}...")
    train_loader, test_loader, input_dims = load_data(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
        BATCH_SIZE
    )
    
    model = MultiTaskModel(
        input_dim_cont=input_dims['cont'],
        input_dim_cat=input_dims['cat']
    ).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model file not found. Ensure training is complete.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_cont, x_cat, y in test_loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            pred, _, _ = model(x_cont, x_cat)
            predicted = (pred > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=['Normal', 'Attack'])
    
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)
    
    print("\n" + "="*40)
    print("       FINAL EVALUATION REPORT       ")
    print("="*40)
    print(f"accuracy:             {acc*100:.2f}%")
    print(f"detection_rate (DR):  {detection_rate*100:.2f}%")
    print(f"false_alarm_rate (FAR): {false_alarm_rate*100:.2f}%")
    print("-" * 40)
    print("Contrast vs Original KDD99/NSL-KDD Baselines:")
    print("Original KDD99 Best: ~92%")
    print("NSL-KDD Typical DL:  ~82% (Test+)")
    print("Target SOTA:         >90% (Test+)")
    print("-" * 40)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{cr}")

if __name__ == "__main__":
    evaluate_metrics()
