import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import os
from data_loader import load_data
from model import MultiTaskModel

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r"data"
MODEL_PATH = r"results/best_model.pth"
RESULTS_DIR = r"results"
BATCH_SIZE = 128
DPI = 600

def generate_high_res_plots():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Model & Data
    print("Loading resources for plotting...")
    train_loader, test_loader, input_dims = load_data(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
        BATCH_SIZE
    )
    
    model = MultiTaskModel(input_dims['cont'], input_dims['cat']).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Warning: Model not found. Plots will be random/untrained.")

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
    
    # --- COMBINED FIGURE ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'], ax=axes[0], annot_kws={"size": 14})
    axes[0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=16, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'combined_performance_600dpi.png')
    plt.savefig(save_path, dpi=DPI)
    print(f"Saved combined figure to {save_path}")
    plt.close()

    # --- TRAINING CURVE (Enhanced Mock based on logs) ---
    # Epoch 1/50 | Loss: 0.0720 | Train Acc: 0.9792
    # Epoch 2/50 | Loss: 0.0331 | Train Acc: 0.9888
    # Epoch 3/50 | Loss: 0.0285 | Train Acc: 0.9905
    # Epoch 4/50 | Loss: 0.0273 | Train Acc: 0.9908
    # Epoch 5/50 | Loss: 0.0236 | Train Acc: 0.9921
    # Epoch 6/50 | Loss: 0.0220 | Train Acc: 0.9925
    # Epoch 7/50 | Loss: 0.0212 | Train Acc: 0.9927
    # Epoch 8/50 | Loss: 0.0204 | Train Acc: 0.9934
    # Epoch 9/50 | Loss: 0.0200 | Train Acc: 0.9933
    epochs = np.arange(1, 10)
    loss_vals = [0.0720, 0.0331, 0.0285, 0.0273, 0.0236, 0.0220, 0.0212, 0.0204, 0.0200]
    acc_vals =  [0.9792, 0.9888, 0.9905, 0.9908, 0.9921, 0.9925, 0.9927, 0.9934, 0.9933]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(epochs, loss_vals, color=color, marker='o', linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)  
    ax2.plot(epochs, acc_vals, color=color, marker='s', linewidth=2, label='Training Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Dynamics (Acc vs Loss)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_dynamics_600dpi.png'), dpi=DPI)
    print("Saved training dynamics plot.")

if __name__ == "__main__":
    generate_high_res_plots()
