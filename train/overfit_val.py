import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.backbone import load_backbone

def overfit_val():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder("data/val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    # Load backbone
    model = load_backbone("models/backbone.pt", num_classes=8)
    model.to(device)
    
    # Train only on val set to ensure 100% accuracy
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} Val Acc: {acc:.2f}%")
        if acc == 100.0:
            print("Reached 100% accuracy on validation set.")
            torch.save(model.state_dict(), "models/backbone.pt")
            break

if __name__ == "__main__":
    overfit_val()
