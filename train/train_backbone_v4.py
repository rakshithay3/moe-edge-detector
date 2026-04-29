"""Train MobileNetV3-Small backbone for 8-class appliance classification.

Trains on all 8 classes found in data/train/:
  air_conditioner, air_purifier, background, dishwasher,
  microwave, refrigerator, tv, washing_machine

Features:
  1. Partially unfrozen backbone (last 3 feature blocks)
  2. Differential learning rate (low for pretrained, high for head)
  3. Strong augmentation to simulate webcam conditions
  4. Cosine warmup schedule
  5. Class-balanced sampling + weighted loss
  6. Mixup augmentation for better generalization
  7. Early stopping with patience

Target: 100% validation accuracy (zero misclassifications).

Usage:
    python train/train_backbone_v4.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

BATCH_SIZE = 16
EPOCHS = 80
LR_BACKBONE = 5e-5       # conservative LR for pretrained layers
LR_HEAD = 2e-3            # high LR for classifier head
LABEL_SMOOTHING = 0.0     # no smoothing → sharper predictions for 0 misclass
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2         # mixup regularization

# ── Augmentation ─────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.RandomCrop((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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

# ── Class distribution & balancing ───────────────────────────────
targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets, minlength=num_classes)
print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")

# Inverse frequency weights for balanced sampling
class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
class_weights = class_weights / class_weights.sum()
print(f"Class weights: {dict(zip(train_dataset.classes, class_weights.round(4)))}")

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

# ── Model — PARTIALLY unfrozen backbone ──────────────────────────
model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")

# Freeze early layers, unfreeze last 3 blocks + classifier
num_feature_blocks = len(model.features)
UNFREEZE_FROM = max(0, num_feature_blocks - 3)

for i, block in enumerate(model.features):
    if i < UNFREEZE_FROM:
        for param in block.parameters():
            param.requires_grad = False
    else:
        for param in block.parameters():
            param.requires_grad = True

# Replace classifier head for our num_classes
model.classifier[3] = nn.Linear(1024, num_classes)

frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = frozen + trainable
print(f"\nFrozen params:    {frozen:,}")
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
print(f"Unfrozen feature blocks: {UNFREEZE_FROM}..{num_feature_blocks-1}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

# ── Differential learning rate ───────────────────────────────────
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "classifier" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

print(f"Backbone trainable params: {sum(p.numel() for p in backbone_params):,}")
print(f"Head trainable params:     {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 3)

# Loss with class weights
loss_weights = torch.tensor(1.0 / np.maximum(class_counts, 1).astype(float),
                            dtype=torch.float32)
loss_weights = loss_weights / loss_weights.sum()
criterion = nn.CrossEntropyLoss(
    weight=loss_weights.to(device),
    label_smoothing=LABEL_SMOOTHING
)


# ── Mixup helper ─────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training loop ────────────────────────────────────────────────
best_val_acc = 0.0
patience = 15
no_improve = 0
WARMUP_EPOCHS = 3

print(f"\n{'='*70}")
print(f"{'Epoch':>5}  {'Loss':>8}  {'Train%':>7}  {'Val%':>7}  {'LR_bb':>10}  {'LR_hd':>10}")
print(f"{'='*70}")

for epoch in range(EPOCHS):
    # -- Warmup --
    if epoch < WARMUP_EPOCHS:
        warmup_factor = (epoch + 1) / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]:
                pg["lr"] = LR_BACKBONE * warmup_factor
            else:
                pg["lr"] = LR_HEAD * warmup_factor

    # -- Train --
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Apply mixup
        if MIXUP_ALPHA > 0 and epoch >= WARMUP_EPOCHS:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            out = model(mixed_imgs)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
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

    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    lr_bb = optimizer.param_groups[0]["lr"]
    lr_hd = optimizer.param_groups[1]["lr"]

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/backbone.pt")
        marker = " ★ saved"
        no_improve = 0
    else:
        no_improve += 1

    print(f"{epoch+1:>5}  {train_loss:>8.4f}  {train_acc:>6.1f}%  {val_acc:>6.1f}%  "
          f"{lr_bb:>10.6f}  {lr_hd:>10.6f}{marker}")

    if no_improve >= patience:
        print(f"\nEarly stopping — no improvement for {patience} epochs")
        break

    # If we hit 100% val accuracy, keep training a few more to confirm stability
    if val_acc == 100.0 and epoch >= 10:
        print(f"\n🎯 Perfect validation accuracy at epoch {epoch+1}!")
        # Save and break — we achieved the goal
        torch.save(model.state_dict(), "models/backbone.pt")
        break

print(f"\n{'='*70}")
print(f"Best validation accuracy: {best_val_acc:.1f}%")
print("Saved → models/backbone.pt")
