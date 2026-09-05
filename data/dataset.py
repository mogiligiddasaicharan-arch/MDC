"""Dataset loading utilities for Manufacturing Defect Classification.

Expects folder structure:
    dataset/
    |-- train/<class_name>/*.jpg
    |-- val/<class_name>/*.jpg
    |-- test/<class_name>/*.jpg
"""

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(split: str, img_size: int = config.IMG_SIZE):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)], p=0.2
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])


def get_dataloaders(data_dir: str = None,
                    batch_size: int = None,
                    num_workers: int = None,
                    img_size: int = None):
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    img_size = img_size or config.IMG_SIZE

    splits = {}
    class_names = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected folder '{split_dir}' not found. "
                f"Make sure your dataset is organized as "
                f"{data_dir}/train|val|test/<class_name>/*.jpg"
            )

        dataset = datasets.ImageFolder(split_dir, transform=get_transforms(split, img_size))
        if class_names is None:
            class_names = dataset.classes
        splits[split] = dataset

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
