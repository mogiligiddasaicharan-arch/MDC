# train_batch.py - Train specialists in batches with resume support
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time

import config
from models.msa_net import MSANet


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
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
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def find_data_structure(domain_path):
    train_dir = os.path.join(domain_path, "train")
    val_dir = os.path.join(domain_path, "val")

    if os.path.exists(train_dir):
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
        return domain_path, None

    return None, None


def count_images(domain_path):
    """Count total images in a domain folder."""
    count = 0
    for root, dirs, files in os.walk(domain_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                count += 1
    return count


def train_specialist(domain_name, data_dir, epochs=25, patience=5):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training specialist: {domain_name}")
    print(f"{sep}")

    device = config.DEVICE
    print(f"Device: {device}")

    domain_path = os.path.join(data_dir, domain_name)
    train_dir, val_dir = find_data_structure(domain_path)

    if train_dir is None:
        print(f"Skipping {domain_name}: no valid training data found")
        return False

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
            print(f"Auto-split: {train_size} train, {val_size} val")
        else:
            train_dataset = full_dataset
            val_dataset = None
            print(f"Not enough data to split: {len(full_dataset)} samples")
    else:
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(val_dir, transform=transform) if os.path.exists(val_dir) else None

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0) if val_dataset else None

    num_classes = len(full_dataset.classes)
    print(f"Classes: {full_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")

    model = MSANet(num_classes=num_classes, base_channels=config.BASE_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    early_stop = EarlyStopping(patience=patience, verbose=True)

    best_val_acc = 0.0
    checkpoint_dir = os.path.join("checkpoints", domain_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.2f}% | {epoch_time:.1f}s")

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
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | {epoch_time:.1f}s")
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": full_dataset.classes,
                }, os.path.join(checkpoint_dir, "best_model.pth"))

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes | Best val acc: {best_val_acc:.2f}%")
    return True


def get_pending_domains(data_dir="datasets"):
    """Get list of domains that still need training, sorted by image count (smallest first)."""
    if not os.path.exists(data_dir):
        return []

    domains = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item != "domain_classifier":
            ckpt = os.path.join("checkpoints", item, "best_model.pth")
            if not os.path.exists(ckpt):
                img_count = count_images(item_path)
                domains.append((item, img_count))

    # Sort by image count ascending (train smallest first for quick wins)
    domains.sort(key=lambda x: x[1])
    return domains


if __name__ == "__main__":
    data_dir = "datasets"

    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        sys.exit(1)

    pending = get_pending_domains(data_dir)

    if not pending:
        print("All specialists are already trained!")
        sys.exit(0)

    print("=" * 60)
    print("PENDING DOMAINS (sorted by size - smallest first)")
    print("=" * 60)
    total_images = 0
    for domain, count in pending:
        print(f"  {domain}: {count} images")
        total_images += count
    print(f"\nTotal remaining images: {total_images}")
    print(f"Total remaining domains: {len(pending)}")
    print("=" * 60)

    # Check if user wants to train all or a specific batch
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target.isdigit():
            # Train first N domains
            n = int(target)
            to_train = pending[:n]
            print(f"\nTraining first {n} domains...")
        else:
            # Train specific domain
            to_train = [(target, count_images(os.path.join(data_dir, target)))]
            print(f"\nTraining domain: {target}")
    else:
        to_train = pending
        print(f"\nTraining ALL {len(pending)} remaining domains...")

    print("\nPress Ctrl+C at any time to stop. Resume later by running again.")
    print("Already trained domains will be automatically skipped.\n")

    for domain, count in to_train:
        success = train_specialist(domain, data_dir, epochs=25, patience=5)
        if not success:
            print(f"Failed to train {domain}, continuing...")

    print("\n" + "=" * 60)
    print("Batch training complete!")
    print("=" * 60)
