"""
Dataset loading utilities for Manufacturing Defect Classification.

Expects folder structure:
dataset/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg

Uses torchvision.datasets.ImageFolder under the hood, which automatically
infers class labels from subfolder names (sorted alphabetically).
"""

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(split: str):
    """Return torchvision transform pipeline for a given split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)], p=0.2
            ),
            transforms.Grayscale(num_output_channels=3),  # safe even if already RGB-ish
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    else:  # val / test — no augmentation, just resize + normalize
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])


def get_dataloaders():
    """
    Builds train/val/test DataLoaders from config.DATA_DIR.
    Returns: (train_loader, val_loader, test_loader, class_names)
    """
    splits = {}
    class_names = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(config.DATA_DIR, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected folder '{split_dir}' not found. "
                f"Make sure your dataset is organized as "
                f"{config.DATA_DIR}/train|val|test/<class_name>/*.jpg"
            )

        dataset = datasets.ImageFolder(split_dir, transform=get_transforms(split))
        if class_names is None:
            class_names = dataset.classes
        splits[split] = dataset

    train_loader = DataLoader(
        splits["train"], batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        splits["val"], batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        splits["test"], batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick smoke test: run `python data/dataset.py` from project root
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
