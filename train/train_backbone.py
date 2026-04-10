"""Train the MobileNetV3-Small backbone on appliance classification.

Handles class imbalance via:
  1. Weighted random sampling (balanced batches)
  2. Class-weighted CrossEntropyLoss
  3. Data augmentation for minority classes

Usage:
    python train/train_backbone.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

# ── Augmented transforms for training ────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((340, 340)),
    transforms.RandomCrop((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/train", transform=train_transform)

print(f"Classes: {dataset.classes}")
print(f"Total samples: {len(dataset)}")

# ── Compute class weights for balanced training ──────────────────
targets = np.array(dataset.targets)
class_counts = np.bincount(targets)
print(f"Class distribution: {dict(zip(dataset.classes, class_counts))}")

# Inverse frequency weights
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # normalize
print(f"Class weights: {dict(zip(dataset.classes, class_weights.round(4)))}")

# Per-sample weights for WeightedRandomSampler
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

# ── Model ────────────────────────────────────────────────────────
model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
# Replace only the final linear layer (classifier[3]: Linear(1024→1000))
model.classifier[3] = nn.Linear(1024, len(dataset.classes))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Weighted loss as an additional safeguard
loss_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
loss_weights = loss_weights / loss_weights.sum()
criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))

# ── Training loop ────────────────────────────────────────────────
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = out.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}  Acc: {accuracy:.1f}%")

torch.save(model.state_dict(), "models/backbone.pt")
print("Saved → models/backbone.pt")
