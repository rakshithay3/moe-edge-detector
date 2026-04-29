"""Train MobileNetV3-Small backbone — ZERO MISCLASSIFICATION target.

Two-phase training:
  Phase 1: Train only classifier head (frozen backbone) — fast convergence
  Phase 2: Unfreeze ALL layers with very low LR — fine-tune for perfection

No mixup, no label smoothing — we want maximum confidence and zero errors.
Strong class balancing for the small dishwasher class.

Usage:
    python train/train_backbone_v5.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4

# ── Phase 1: Head-only training ──────────────────────────────────
PHASE1_EPOCHS = 15
PHASE1_LR = 3e-3

# ── Phase 2: Full fine-tuning ────────────────────────────────────
PHASE2_EPOCHS = 60
PHASE2_LR_BACKBONE = 1e-5   # very low for pretrained features
PHASE2_LR_HEAD = 5e-4       # moderate for head
PATIENCE = 12

# ── Augmentation ─────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.RandomCrop((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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

num_classes = len(train_dataset.classes)
print(f"Classes ({num_classes}): {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ── Class balancing ──────────────────────────────────────────────
targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets, minlength=num_classes)
print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")

class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
class_weights = class_weights / class_weights.sum()

sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

# Also create a non-augmented train loader for final evaluation
train_eval_transform = val_transform  # same as val (no augmentation)
train_eval_dataset = datasets.ImageFolder("data/train", transform=train_eval_transform)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0)

# ── Model ────────────────────────────────────────────────────────
model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(1024, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

# ── Loss ─────────────────────────────────────────────────────────
loss_weights = torch.tensor(1.0 / np.maximum(class_counts, 1).astype(float),
                            dtype=torch.float32)
loss_weights = loss_weights / loss_weights.sum()
criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))


def evaluate(loader, desc="Val"):
    """Evaluate model on a loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, predicted = out.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_epoch(optimizer):
    """Train for one epoch."""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = out.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Train classifier head only (backbone frozen)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 1: Head-only training (backbone frozen)")
print(f"{'='*70}")

# Freeze entire backbone
for param in model.features.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

optimizer1 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)

best_val_acc = 0.0

print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Train%':>7}  {'Val%':>7}")
print("-" * 40)

for epoch in range(PHASE1_EPOCHS):
    loss, train_acc = train_epoch(optimizer1)
    val_acc = evaluate(val_loader)
    scheduler1.step()

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/backbone.pt")
        marker = " ★"

    print(f"{epoch+1:>5}  {loss:>8.4f}  {train_acc:>6.1f}%  {val_acc:>6.1f}%{marker}")

print(f"Phase 1 best: {best_val_acc:.1f}%")

# ══════════════════════════════════════════════════════════════════
# PHASE 2: Full fine-tuning (all layers unfrozen)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 2: Full fine-tuning (all layers unfrozen)")
print(f"{'='*70}")

# Reload best Phase 1 checkpoint
model.load_state_dict(torch.load("models/backbone.pt", map_location=device, weights_only=True))

# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# Differential LR
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "classifier" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer2 = torch.optim.AdamW([
    {"params": backbone_params, "lr": PHASE2_LR_BACKBONE},
    {"params": head_params, "lr": PHASE2_LR_HEAD},
], weight_decay=WEIGHT_DECAY)

scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS)

no_improve = 0

print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Train%':>7}  {'Val%':>7}  {'LR_bb':>10}  {'LR_hd':>10}")
print("-" * 65)

for epoch in range(PHASE2_EPOCHS):
    loss, train_acc = train_epoch(optimizer2)
    val_acc = evaluate(val_loader)
    scheduler2.step()

    lr_bb = optimizer2.param_groups[0]["lr"]
    lr_hd = optimizer2.param_groups[1]["lr"]

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/backbone.pt")
        marker = " ★ saved"
        no_improve = 0
    else:
        no_improve += 1

    print(f"{epoch+1:>5}  {loss:>8.4f}  {train_acc:>6.1f}%  {val_acc:>6.1f}%  "
          f"{lr_bb:>10.6f}  {lr_hd:>10.6f}{marker}")

    # Early check: perfect val
    if val_acc == 100.0:
        print(f"\n🎯 PERFECT validation accuracy at epoch {epoch+1}!")
        torch.save(model.state_dict(), "models/backbone.pt")
        break

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping — no improvement for {PATIENCE} epochs")
        break

# ══════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FINAL EVALUATION")
print(f"{'='*70}")

# Reload best model
model.load_state_dict(torch.load("models/backbone.pt", map_location=device, weights_only=True))
model.eval()

train_acc = evaluate(train_eval_loader, "Train")
val_acc = evaluate(val_loader, "Val")

print(f"Train accuracy (no aug): {train_acc:.1f}%")
print(f"Val accuracy:            {val_acc:.1f}%")
print(f"Best val accuracy:       {best_val_acc:.1f}%")
print(f"\nSaved → models/backbone.pt")
