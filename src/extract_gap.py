"""Extract GAP vectors from trained backbone for router training."""

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


def extract_all_gap_vectors(data_dir="data/train", weights_path="models/backbone.pt",
                            output_vectors="data/gap_vectors_train.npy",
                            output_labels="data/gap_labels_train.npy"):
    """Extract GAP vectors from all training images and save to disk.

    Expert group mapping:
        0 → kitchen (refrigerator, microwave, dishwasher)
        1 → display (tv)
        2 → climate (air_conditioner, air_purifier)
        3 → utility (washing_machine, robot_vacuum)
    """
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1)

    model = load_backbone(weights_path)

    X, y = [], []

    for img, label in loader:
        gap = extract_gap(model, img)
        X.append(gap.numpy().squeeze())

        # Map class to expert group
        class_name = dataset.classes[label.item()]

        if class_name in ["refrigerator", "microwave", "dishwasher"]:
            y.append(0)           # Expert 0: kitchen
        elif class_name == "tv":
            y.append(1)           # Expert 1: display
        elif class_name in ["air_conditioner", "air_purifier"]:
            y.append(2)           # Expert 2: climate
        elif class_name in ["washing_machine", "robot_vacuum"]:
            y.append(3)           # Expert 3: utility
        else:
            X.pop()               # remove unmatched GAP vector
            continue

    np.save(output_vectors, np.array(X))
    np.save(output_labels, np.array(y))

    print(f"Saved {len(X)} GAP vectors → {output_vectors}")
    print(f"Saved {len(y)} labels     → {output_labels}")


if __name__ == "__main__":
    extract_all_gap_vectors()
