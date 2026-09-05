"""
Evaluate a trained MSA-Net checkpoint on the test set.

Usage:
    python evaluate.py
"""

import os
import json
import torch
import numpy as np

import config
from data.dataset import get_dataloaders
from models.msa_net import MSANet
from utils.metrics import (
    compute_metrics, print_classification_report, plot_confusion_matrix
)


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {config.BEST_MODEL_PATH}. "
            f"Run train.py first."
        )

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    _, _, test_loader, loader_class_names = get_dataloaders()
    if loader_class_names != class_names:
        print("WARNING: class order in dataset differs from checkpoint. "
              "Using checkpoint's class order for label names; make sure "
              "your dataset folders haven't changed since training.")

    model = MSANet(num_classes=num_classes,
                    base_channels=config.BASE_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nCheckpoint from epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']:.4f}\n")

    metrics = compute_metrics(all_labels, all_preds, class_names)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}\n")

    print_classification_report(all_labels, all_preds, class_names)

    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    )

    # Save metrics as JSON for record-keeping
    metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
