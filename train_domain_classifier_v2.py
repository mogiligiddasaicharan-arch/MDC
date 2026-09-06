import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import time, json, os

device = "cpu"

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("datasets/domain_classifier/train", transform=train_tf)
val_ds = datasets.ImageFolder("datasets/domain_classifier/val", transform=val_tf)

print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
os.makedirs("checkpoints/domain_classifier_v2", exist_ok=True)
with open("checkpoints/domain_classifier_v2/classes.json", "w") as f:
    json.dump(train_ds.classes, f)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(train_ds.classes))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

best_acc = 0
patience, counter = 5, 0
EPOCHS = 15

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {acc:.2f}% | {elapsed:.1f}s")

    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "checkpoints/domain_classifier_v2/best_model.pth")
        print(f"  -> New best model saved ({acc:.2f}%)")
    else:
        counter += 1
        print(f"  EarlyStopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping.")
            break

print(f"\nDone. Best val accuracy: {best_acc:.2f}%")
print("Saved to checkpoints/domain_classifier_v2/best_model.pth")
