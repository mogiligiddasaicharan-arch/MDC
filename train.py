"""Train MSA-Net on any manufacturing defect dataset.

Usage:
    python train.py --domain steel
    python train.py --domain steel --epochs 30 --batch_size 16 --lr 5e-5
"""

import os
import sys
import argparse
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data.dataset import get_dataloaders
from models.msa_net import MSANet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MSA-Net specialist model")
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain name (must match a folder under datasets/)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Number of epochs (default: {config.EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LR,
                        help=f"Learning rate (default: {config.LR})")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help=f"DataLoader workers (default: {config.NUM_WORKERS})")
    parser.add_argument("--base_channels", type=int, default=config.BASE_CHANNELS,
                        help=f"Base channels for MSA-Net (default: {config.BASE_CHANNELS})")
    args = parser.parse_args()

    data_dir = os.path.join("datasets", args.domain)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Dataset folder not found: {data_dir}")
        sys.exit(1)

    ckpt_dir = os.path.join("checkpoints", args.domain)
    out_dir = os.path.join("outputs", args.domain)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Training domain: {args.domain} | Data: {data_dir}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Base channels: {args.base_channels}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    with open(os.path.join(ckpt_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    model = MSANet(num_classes=num_classes, base_channels=args.base_channels).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_model_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs).")
            break

    plot_curves(history, os.path.join(out_dir, "training_curves.png"))
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
