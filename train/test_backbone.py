"""Test the trained MobileNetV3-Small backbone on the validation set.

Evaluates:
  1. Overall accuracy
  2. Per-class precision, recall, F1
  3. Confusion matrix

Usage:
    python train/test_backbone.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

MODEL_PATH = "models/backbone.pt"
VAL_DIR = "data/val"
BATCH_SIZE = 16

# ── Same preprocessing as training (no augmentation) ─────────────
val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = val_dataset.classes
num_classes = len(classes)

print(f"Classes: {classes}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Per-class: {dict(zip(classes, np.bincount(val_dataset.targets)))}")
print()

# ── Load model ───────────────────────────────────────────────────
model = torchvision.models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(1024, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ── Evaluation ───────────────────────────────────────────────────
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Overall accuracy ─────────────────────────────────────────────
overall_acc = (all_preds == all_labels).mean() * 100
print(f"Overall Accuracy: {overall_acc:.1f}%")
print()

# ── Confusion matrix ─────────────────────────────────────────────
cm = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true][pred] += 1

# Print confusion matrix
max_name_len = max(len(c) for c in classes)
header = " " * (max_name_len + 2) + "  ".join(f"{c:>{max_name_len}}" for c in classes)
print("Confusion Matrix (rows=true, cols=predicted):")
print(header)
for i, cls in enumerate(classes):
    row = "  ".join(f"{cm[i][j]:>{max_name_len}}" for j in range(num_classes))
    print(f"{cls:>{max_name_len}}  {row}")
print()

# ── Per-class metrics ────────────────────────────────────────────
print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 62)

for i, cls in enumerate(classes):
    tp = cm[i][i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    support = cm[i, :].sum()

    print(f"{cls:<20} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10d}")

# Macro averages
precisions, recalls, f1s = [], [], []
for i in range(num_classes):
    tp = cm[i][i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

print("-" * 62)
print(f"{'Macro Avg':<20} {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} {np.mean(f1s):>10.3f} {len(all_labels):>10d}")
print()

# ── Show misclassified samples ───────────────────────────────────
misclassified = np.where(all_preds != all_labels)[0]
if len(misclassified) > 0:
    print(f"Misclassified samples ({len(misclassified)}):")
    for idx in misclassified:
        path, _ = val_dataset.samples[idx]
        true_cls = classes[all_labels[idx]]
        pred_cls = classes[all_preds[idx]]
        print(f"  {path}  (true: {true_cls}, predicted: {pred_cls})")
else:
    print("No misclassified samples — perfect on validation set!")
