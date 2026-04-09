"""Train the Router MLP on GAP vectors to select the correct expert.

Loads precomputed GAP vectors and expert labels, then trains a
small MLP (960 → 256 → 3) to classify which expert should handle
the input.

Usage:
    python train/train_router.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.router import RouterMLP

# ── Hyperparameters ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 256
NUM_EXPERTS = 3
DROPOUT = 0.3

# ── Load precomputed GAP vectors ────────────────────────────────
X = np.load("data/gap_vectors_train.npy")
y = np.load("data/gap_labels_train.npy")

print(f"GAP vectors: {X.shape}")
print(f"Labels:      {y.shape}")
print(f"Expert distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

optimizer = torch.optim.Adam(router.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ── Training loop ────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(EPOCHS):
    router.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = router(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(yb).sum().item()
        total += yb.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(router.state_dict(), "models/router.pt")

    print(f"Epoch {epoch+1:02d}/{EPOCHS}  Loss: {avg_loss:.4f}  "
          f"Acc: {accuracy:.1f}%  Best: {best_acc:.1f}%")

print(f"\nBest accuracy: {best_acc:.1f}%")
print("Saved → models/router.pt")
