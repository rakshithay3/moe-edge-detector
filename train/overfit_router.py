import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.router import RouterMLP

def overfit_router():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X = np.load("data/gap_vectors_train.npy")
    y = np.load("data/gap_labels_train.npy")
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)
    
    router = RouterMLP(input_dim=576, hidden_dim=256, num_experts=4, dropout=0.0)
    router.to(device)
    
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        router.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = router(xb)
            loss = criterion(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        router.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = router(xb)
                _, pred = logits.max(1)
                correct += pred.eq(yb).sum().item()
                total += yb.size(0)
                
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} Acc: {acc:.2f}%")
        
        if acc == 100.0:
            print("Router reached 100% accuracy!")
            torch.save(router.state_dict(), "models/router.pt")
            break

if __name__ == "__main__":
    overfit_router()
