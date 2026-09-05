import os
content = open('train_all_specialists_new.py', 'w', encoding='utf-8')
content.write('''"""Train all remaining specialist models - Auto-detects folder structure."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config
from models.msa_net import MSANet

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: " + str(self.counter) + " out of " + str(self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def find_data_structure(domain_path):
    train_dir = os.path.join(domain_path, "train")
    val_dir = os.path.join(domain_path, "val")
    if os.path.exists(train_dir):
        print("  Found train/val split structure")
        return train_dir, val_dir if os.path.exists(val_dir) else None
    has_class_folders = False
    for item in os.listdir(domain_path):
        item_path = os.path.join(domain_path, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    has_class_folders = True
                    break
            if has_class_folders:
                break
    if has_class_folders:
        print("  Found flat structure - will auto-split 80/20")
        return domain_path, None
    return None, None

def train_specialist(domain_name, data_dir, epochs=40, patience=7):
    print("\n" + "="*60)
    print("Training specialist: " + domain_name)
    print("="*60)
    domain_path = os.path.join(data_dir, domain_name)
    train_dir, val_dir = find_data_structure(domain_path)
    if train_dir is None:
        print("Skipping " + domain_name + ": no valid training data found")
        return
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    if val_dir is None:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if val_size > 0:
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print("Auto-split: " + str(train_size) + " train, " + str(val_size) + " val")
        else:
            train_dataset = full_dataset
            val_dataset = None
            print("Not enough data to split: " + str(len(full_dataset)) + " samples")
    else:
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(val_dir, transform=transform) if os.path.exists(val_dir) else None
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2) if val_dataset else None
    num_classes = len(full_dataset.classes)
    print("Classes: " + str(full_dataset.classes))
    print("Train samples: " + str(len(train_dataset)))
    if val_dataset:
        print("Val samples: " + str(len(val_dataset)))
    model = MSANet(num_classes=num_classes, base_channels=config.BASE_CHANNELS).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    early_stop = EarlyStopping(patience=patience, verbose=True)
    best_val_acc = 0.0
    checkpoint_dir = os.path.join("checkpoints", domain_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        if val_loader:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            scheduler.step(val_loss)
            print("Epoch " + str(epoch) + "/" + str(epochs) + " | Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)) + " | Val Acc: " + str(round(val_acc, 2)) + "%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": full_dataset.classes,
                    "val_acc": val_acc,
                }, os.path.join(checkpoint_dir, "best_model.pth"))
            early_stop(val_loss, model)
            if early_stop.early_stop:
                print("Early stopping at epoch " + str(epoch))
                break
        else:
            print("Epoch " + str(epoch) + "/" + str(epochs) + " | Train Loss: " + str(round(train_loss, 4)))
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": full_dataset.classes,
                }, os.path.join(checkpoint_dir, "best_model.pth"))
    print("Best val accuracy: " + str(round(best_val_acc, 2)) + "%")

if __name__ == "__main__":
    data_dir = "datasets"
    if not os.path.exists(data_dir):
        print("Data directory '" + data_dir + "' not found!")
        exit(1)
    domains = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item != "domain_classifier":
            domains.append(item)
    print("Found domains: " + str(domains))
    already_trained = []
    if os.path.exists("checkpoints"):
        already_trained = [d for d in os.listdir("checkpoints") 
                          if os.path.isdir(os.path.join("checkpoints", d)) 
                          and d != "domain_classifier"]
    print("Already trained: " + str(already_trained))
    for domain in domains:
        checkpoint_path = os.path.join("checkpoints", domain, "best_model.pth")
        if os.path.exists(checkpoint_path):
            print("Skipping " + domain + ": already trained")
            continue
        train_specialist(domain, data_dir)
    print("\nAll specialists trained!")
''')
content.close()
print("File created successfully!")
