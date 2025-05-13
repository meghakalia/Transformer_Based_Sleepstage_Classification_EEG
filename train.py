
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import dotenv
import os
import matplotlib.pyplot as plt

dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

# === 0. Init wandb ===
wandb.init(project="sleep-stage-transformer", name="patient1-transformer", config={
    "epochs": 5,
    "batch_size": 32,
    "lr": 1e-3,
    "model": "TransformerEncoder",
})

# === 1. Load patient1 data ===
df = pd.read_csv("sleep_patient1.csv")

# Infer input shape and reshape
X = df.drop(columns=["stage"]).values
X = X.reshape(-1, 10, 30)  # assuming 300 features per epoch → (samples, 10, 30)
y = df["stage"].values

# Label index → stage name (optional mapping if known)
label_names = ['W', 'N1', 'N2', 'N3', 'REM']

# Safe train-test split (no stratify if few classes present)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Dataset Class ===
class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = SleepDataset(X_train, y_train)
test_ds = SleepDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=wandb.config.batch_size)

# === 3. Transformer Model ===
class SleepTransformer(nn.Module):
    def __init__(self, input_dim=30, d_model=64, num_classes=5):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (B, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, B, d_model)
        out = self.transformer(x)
        out = out.mean(dim=0)
        return self.cls(out)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SleepTransformer(num_classes=len(np.unique(y))).to(device)

# === 4. Train ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

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

        _, pred_labels = torch.max(preds, 1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)

    acc = correct / total
    wandb.log({"Train Loss": total_loss / len(train_loader), "Train Accuracy": acc})
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Accuracy: {acc:.4f}")

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
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")

# === 7. Log to wandb ===
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close(fig)
