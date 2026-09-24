import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import time, json, os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = "cpu"

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.15, 0.15, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("datasets/steel/train", transform=train_tf)
val_ds = datasets.ImageFolder("datasets/steel/val", transform=val_tf)
class_names = train_ds.classes
print(f"Steel defect classes: {class_names}")

labels = [label for _, label in train_ds.samples]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights (handles imbalance): {dict(zip(class_names, class_weights.tolist()))}")

os.makedirs("checkpoints/steel_v2", exist_ok=True)
with open("checkpoints/steel_v2/classes.json", "w") as f:
    json.dump(class_names, f)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

best_acc = 0
patience, counter = 7, 0
EPOCHS = 25

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0
    for imgs, labels_batch in train_loader:
        imgs, labels_batch = imgs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels_batch in val_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            out = model(imgs)
            loss = criterion(out, labels_batch)
            val_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

    acc = 100 * correct / total
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {acc:.2f}% | {elapsed:.1f}s")

    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "checkpoints/steel_v2/best_model.pth")
        print(f"  -> New best model saved ({acc:.2f}%)")
    else:
        counter += 1
        print(f"  EarlyStopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping.")
            break

print(f"\nDone. Best val accuracy: {best_acc:.2f}%")
