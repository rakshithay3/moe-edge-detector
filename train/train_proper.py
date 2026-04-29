import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def train_proper():
    BATCH_SIZE = 32
    EPOCHS = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.RandomCrop((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
    num_classes = len(train_dataset.classes)
    
    # Class balancing
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(1024, num_classes)
    model.to(device)

    # Train all layers, but with a smaller learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Training backbone properly...")
    for epoch in range(EPOCHS):
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
            
        scheduler.step()
        
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/total:.4f} - Train Acc: {train_acc:.2f}%")

    torch.save(model.state_dict(), "models/backbone.pt")
    print("Backbone saved.")

if __name__ == "__main__":
    train_proper()
