"""Retrain the MobileNetV3-Small backbone with improvements.

Fixes:
  1. Added 'background' class so model can say "none of the above"
  2. Frozen feature extractor — only trains the classifier head
     (prevents overfitting on small datasets)
  3. Learning rate scheduler for better convergence
  4. Label smoothing for better calibration

Usage:
    python train/train_backbone_v2.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

BATCH_SIZE = 16
EPOCHS = 20          # more epochs since we're only training the head
LR = 3e-3            # higher LR is fine for small classifier head
LABEL_SMOOTHING = 0.1

# ── Augmented transforms for training ────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((340, 340)),
    transforms.RandomCrop((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Datasets ─────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/val", transform=val_transform)

print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ── Class distribution & balancing ───────────────────────────────
targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets)
print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")

# Inverse frequency weights
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
print(f"Class weights: {dict(zip(train_dataset.classes, class_weights.round(4)))}")

# Weighted sampler
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Model — FROZEN backbone, trainable classifier ────────────────
model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")

# Freeze ALL feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head (keep the structure but change output)
num_classes = len(train_dataset.classes)
model.classifier[3] = nn.Linear(1024, num_classes)

# Only classifier params are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

# Only optimize trainable parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Loss with label smoothing + class weights
loss_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
loss_weights = loss_weights / loss_weights.sum()
criterion = nn.CrossEntropyLoss(
    weight=loss_weights.to(device),
    label_smoothing=LABEL_SMOOTHING
)

# ── Training loop ────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # -- Train --
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
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

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    # -- Validate --
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, predicted = out.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100.0 * val_correct / val_total

    lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Loss: {train_loss:.4f}  "
          f"Train: {train_acc:.1f}%  "
          f"Val: {val_acc:.1f}%  "
          f"LR: {lr:.5f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/backbone.pt")
        print(f"  ↑ New best! Saved → models/backbone.pt")

    scheduler.step()

print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
print("Done!")
