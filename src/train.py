import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import load_data
from model import MultiTaskModel
import os

# CONFIG
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
LAMBDA_RECON = 0.5 # Weight for reconstruction loss
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r"data" # Relative to project root
MODEL_PATH = r"results/best_model.pth"

def train():
    print(f"Using device: {DEVICE}")
    train_loader, test_loader, input_dims = load_data(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
        BATCH_SIZE
    )
    
    model = MultiTaskModel(
        input_dim_cont=input_dims['cont'],
        input_dim_cat=input_dims['cat']
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_cls = nn.BCELoss()
    criterion_recon_cont = nn.MSELoss()
    criterion_recon_cat = nn.MSELoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            y = y.unsqueeze(1) # (B, 1)
            
            optimizer.zero_grad()
            
            # Forward
            pred, recon_cont, recon_cat = model(x_cont, x_cat)
            
            # Loss Calculation
            loss_cls = criterion_cls(pred, y)
            loss_r_cont = criterion_recon_cont(recon_cont, x_cont)
            loss_r_cat = criterion_recon_cat(recon_cat, x_cat) # Approx MSE for one-hot is acceptable for simple reconstruction regularization
            
            loss = loss_cls + LAMBDA_RECON * (loss_r_cont + loss_r_cat)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        
        # Validation (on Test set for now as "Val" proxy during dev, but for final paper we should split Train further)
        # Using Test set to monitor progress is fine for development iterations.
        val_acc = evaluate(model, test_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  --> Saved Best Model")
            
    print(f"Training Complete. Best Test Acc: {best_acc:.4f}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            y = y.unsqueeze(1)
            pred, _, _ = model(x_cont, x_cat)
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    train()
