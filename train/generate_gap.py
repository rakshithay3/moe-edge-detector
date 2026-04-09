"""Generate GAP vectors from the trained backbone for router training.

Reads images from data/train/, extracts 960-d GAP vectors, and maps
each class to an expert group:
    Expert 0 → Display (tv)
    Expert 1 → Kitchen (refrigerator, microwave)
    Expert 2 → Climate (air_conditioner)

Usage:
    python train/generate_gap.py
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.backbone import load_backbone, extract_gap

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/train", transform=transform)
loader = DataLoader(dataset, batch_size=1)

print(f"Classes: {dataset.classes}")
print(f"Samples: {len(dataset)}")

model = load_backbone("models/backbone.pt")

X, y = [], []

for i, (img, label) in enumerate(loader):
    gap = extract_gap(model, img)
    X.append(gap.numpy().squeeze())

    # Map class → expert group
    class_name = dataset.classes[label.item()]

    if class_name == "tv":
        y.append(0)
    elif class_name in ["refrigerator", "microwave"]:
        y.append(1)
    elif class_name == "air_conditioner":
        y.append(2)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(dataset)} images...")

np.save("data/gap_vectors_train.npy", np.array(X))
np.save("data/gap_labels_train.npy", np.array(y))

print(f"\nSaved {len(X)} GAP vectors → data/gap_vectors_train.npy")
print(f"Saved {len(y)} labels     → data/gap_labels_train.npy")
print(f"Expert distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
