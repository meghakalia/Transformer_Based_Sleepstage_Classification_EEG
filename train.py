import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sleepTransformer import SleepTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import os
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from CNNBaseline import CNNBaseline

from focal_loss import FocalLoss

from dotenv import load_dotenv
load_dotenv()  # This works

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

save_model = "/Volumes/FF952/SleepStageClassification/transformer/12_05_2025/model"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model_type = "cnn"
focal_loss_use = True

# === 0. Init wandb ===
wandb.init(
    project="sleep-stage-transformer",
    name=f"{model_type}-all_patients",
    config={
        "epochs": 100,
        "batch_size": 8,
        "lr": 1e-3,
        "model": model_type,
        "loss": f"FL-{focal_loss_use}"
    }
)


# === 1. Load patient1 data ===
# df = pd.read_csv("sleep_all_patients.csv")
df = pd.read_csv("sleep_all_patients.csv")
# Infer input shape and reshape
X = df.drop(columns=["stage"]).values
print("Original X shape:", X.shape)
print("Total elements:", X.size)


# Load and clean features/labels
X = df.iloc[:, :-1].values  # All but last column is signal
X = X[:, :300]              # Retain only first 300 EEG values

# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # shape: (n_samples, 300)

if model_type == "cnn":
    X = X_scaled[:, np.newaxis, :]  # (B, 1, 300)
else:
    X = X_scaled.reshape(-1, 10, 30)  # (B, 10, 30) # for transformer


y = df.iloc[:, -1].values   # Last column = label

# Label names from index
label_names = ['W', 'N1', 'N2', 'N3', 'REM']
unique = np.unique(y)
label_names = [label_names[i] for i in unique]


# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp)

# Compute weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
focal_loss = FocalLoss(gamma=3.0, alpha=class_weights, reduction='mean')

# === 2. Dataset Class ===
class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = SleepDataset(X_train, y_train)
val_ds = SleepDataset(X_val, y_val)
test_ds = SleepDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=wandb.config.batch_size)
test_loader = DataLoader(test_ds, batch_size=wandb.config.batch_size)

# === 3. Transformer Model ===
if model_type == "cnn":
    model = CNNBaseline(num_classes=len(np.unique(y))).to(device)
else:
    model = SleepTransformer(input_dim=30, num_classes=len(np.unique(y))).to(device)

# model = CNNBaseline(num_classes=len(np.unique(y))).to(device)
# # model = SleepTransformer(num_classes=len(np.unique(y))).to(device)

# === 4. Train ===

if focal_loss_use:
    criterion = focal_loss
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

best_val_acc = 0 
for epoch in range(wandb.config.epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # probs = torch.softmax(preds, dim=1)
        # print(probs[0]) # print confidence

        _, pred_labels = torch.max(preds, 1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)

    acc = correct / total
    scheduler.step()

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
            _, pred_labels = torch.max(preds, 1)
            val_correct += (pred_labels == yb).sum().item()
            val_total += yb.size(0)
            val_preds.extend(pred_labels.cpu().numpy())
            val_labels.extend(yb.cpu().numpy())
    val_acc = val_correct / val_total

    # Save model if best val accuracy so far
    if epoch == 0 or val_acc > best_val_acc:
        torch.save(model.state_dict(), os.path.join(save_model, "best_model.pt"))
        best_val_acc = val_acc

    # Compute validation confusion matrix and log to wandb
    val_cm = confusion_matrix(val_labels, val_preds)
    fig_val, ax_val = plt.subplots(figsize=(6, 6))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax_val)
    ax_val.set_xlabel("Predicted")
    ax_val.set_ylabel("True")
    ax_val.set_title(f"Validation Confusion Matrix (Epoch {epoch+1})")
    wandb.log({
        "Train Loss": total_loss / len(train_loader),
        "Train Accuracy": acc,
        "Validation Loss": val_loss / len(val_loader),
        "Validation Accuracy": val_acc,
        "Validation Confusion Matrix": wandb.Image(fig_val)
    })
    plt.close(fig_val)


    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Accuracy: {acc:.4f} | Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

# === Load the best model before evaluation ===
model.load_state_dict(torch.load(os.path.join(save_model, "best_model.pt"), map_location=device))

# === 5. Evaluate ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

# === 6. Classification Report & Confusion Matrix ===
report = classification_report(all_labels, all_preds, target_names=label_names)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")

# === 7. Log to wandb ===
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close(fig)

# === 8. Save inference results to file ===
with open("inference_results.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    np.savetxt(f, cm, fmt='%d')
