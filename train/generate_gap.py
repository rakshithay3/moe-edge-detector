"""Generate GAP vectors from the trained backbone for router training.

Reads images from data/train/, extracts 576-d GAP vectors, and maps
each class to an expert group:
    Expert 0 → kitchen (refrigerator, microwave, dishwasher)
    Expert 1 → display (tv)
    Expert 2 → climate (air_conditioner, air_purifier)
    Expert 3 → utility (washing_machine, robot_vacuum)

Usage:
    python train/generate_gap.py
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.backbone import load_backbone, extract_gap

EXPERT_GROUPS = {
    "kitchen": ["refrigerator", "microwave", "dishwasher"],
    "display": ["tv"],
    "climate": ["air_conditioner", "air_purifier"],
    "utility": ["washing_machine", "robot_vacuum"],
}

CLASS_TO_EXPERT = {
    class_name: expert_id
    for expert_id, (_, class_names) in enumerate(EXPERT_GROUPS.items())
    for class_name in class_names
}

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/train", transform=transform)
loader = DataLoader(dataset, batch_size=32, num_workers=2, pin_memory=torch.cuda.is_available())

print(f"Classes: {dataset.classes}")
print(f"Samples: {len(dataset)}")

model = load_backbone("models/backbone.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

X, y = [], []

for i, (img, label) in enumerate(loader):
    img = img.to(device, non_blocking=True)
    gap = extract_gap(model, img).cpu().numpy()
    batch_labels = label.numpy().tolist()
    batch_classes = [dataset.classes[idx] for idx in batch_labels]

    for j, class_name in enumerate(batch_classes):
        expert_id = CLASS_TO_EXPERT.get(class_name)
        if expert_id is None:
            print(f"  WARNING: Unknown class '{class_name}', skipping")
            continue

        X.append(gap[j])
        y.append(expert_id)

    processed = min((i + 1) * loader.batch_size, len(dataset))
    if (i + 1) % 10 == 0:
        print(f"  Processed {processed}/{len(dataset)} images...")

np.save("data/gap_vectors_train.npy", np.array(X))
np.save("data/gap_labels_train.npy", np.array(y))

print(f"\nSaved {len(X)} GAP vectors → data/gap_vectors_train.npy")
print(f"Saved {len(y)} labels     → data/gap_labels_train.npy")
print(f"Expert distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
