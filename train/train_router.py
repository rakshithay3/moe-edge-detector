"""Train the Router MLP on GAP vectors to select the correct expert.

Handles class imbalance via:
  1. Weighted random sampling (balanced batches)
  2. Class-weighted CrossEntropyLoss

Usage:
    python train/train_router.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from src.router import RouterMLP

# ── Hyperparameters ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 256
NUM_EXPERTS = 4
DROPOUT = 0.3
VAL_RATIO = 0.2
PATIENCE = 8
WEIGHT_DECAY = 1e-4

# ── Load precomputed GAP vectors ────────────────────────────────
X = np.load("data/gap_vectors_train.npy")
y = np.load("data/gap_labels_train.npy")

print(f"GAP vectors: {X.shape}")
print(f"Labels:      {y.shape}")

expert_counts = dict(zip(*np.unique(y, return_counts=True)))
print(f"Expert distribution: {expert_counts}")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ── Stratified split (train/validation) ──────────────────────────
indices = np.arange(len(y))
train_idx, val_idx = [], []
rng = np.random.default_rng(42)

for cls in np.unique(y):
    cls_idx = indices[y == cls]
    rng.shuffle(cls_idx)
    split = max(1, int(len(cls_idx) * (1.0 - VAL_RATIO)))
    train_idx.extend(cls_idx[:split].tolist())
    val_idx.extend(cls_idx[split:].tolist())

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

class_counts = np.bincount(y_train.numpy().astype(int), minlength=NUM_EXPERTS)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[y_train.numpy().astype(int)]

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train),
    replacement=True,
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ── Model ────────────────────────────────────────────────────────
router = RouterMLP(
    input_dim=X.shape[1],       # 960
    hidden_dim=HIDDEN_DIM,
    num_experts=NUM_EXPERTS,
    dropout=DROPOUT,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
router.to(device)

optimizer = torch.optim.AdamW(router.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Weighted loss — inverse frequency
loss_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
loss_weights = loss_weights / loss_weights.sum()
criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# ── Training loop ────────────────────────────────────────────────
best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    router.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = router(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, predicted = logits.max(1)
        train_correct += predicted.eq(yb).sum().item()
        train_total += yb.size(0)

    avg_loss = running_loss / train_total
    train_accuracy = 100.0 * train_correct / train_total

    # Validation phase
    router.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = router(xb)
            _, predicted = logits.max(1)
            val_correct += predicted.eq(yb).sum().item()
            val_total += yb.size(0)

    val_accuracy = 100.0 * val_correct / max(val_total, 1)
    scheduler.step(val_accuracy)

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        epochs_without_improvement = 0
        torch.save(router.state_dict(), "models/router.pt")
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS}  "
        f"Loss: {avg_loss:.4f}  "
        f"Train Acc: {train_accuracy:.1f}%  "
        f"Val Acc: {val_accuracy:.1f}%  "
        f"Best Val: {best_val_acc:.1f}%"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1} (patience={PATIENCE}).")
        break

print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
print("Saved → models/router.pt")
