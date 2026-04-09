"""Train the MobileNetV3-Small backbone on appliance classification.

Finetunes a pretrained MobileNetV3-Small on the training images
organized under data/train/ with ImageFolder structure.

Usage:
    python train/train_backbone.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/train", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Classes: {dataset.classes}")
print(f"Samples: {len(dataset)}")

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(960, len(dataset.classes)))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:  {device}")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

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
