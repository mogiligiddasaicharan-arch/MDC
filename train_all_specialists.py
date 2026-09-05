"""
Train all remaining specialist models.

Expected project structure:

manufacturing_defect_classification/
│
├── config.py
├── train_all_specialists.py
├── models/
│   ├── __init__.py
│   └── msa_net.py
├── datasets/
│   ├── domain1/
│   │   ├── train/
│   │   │   ├── class1/
│   │   │   └── class2/
│   │   └── val/
│   │       ├── class1/
│   │       └── class2/
│   │
│   └── domain2/
│       ├── class1/
│       └── class2/
│
└── checkpoints/
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from models.msa_net import MSANet


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss > self.best_loss - self.delta:
            self.counter += 1

            if self.verbose:
                print(
                    f"  EarlyStopping: "
                    f"{self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0


# ============================================================
# FIND DATASET STRUCTURE
# ============================================================

def find_data_structure(domain_path):
    """
    Detect either:

    1. domain/train/class_x/*.jpg
       domain/val/class_x/*.jpg

    OR

    2. domain/class_x/*.jpg
       domain/class_y/*.jpg

    In the second case, an 80/20 train-validation split
    will automatically be created.
    """

    train_dir = os.path.join(domain_path, "train")
    val_dir = os.path.join(domain_path, "val")

    # --------------------------------------------------------
    # Structure 1: train/val
    # --------------------------------------------------------

    if os.path.isdir(train_dir):

        print("  Found train/val split structure.")

        if os.path.isdir(val_dir):
            print("  Validation directory found.")
            return train_dir, val_dir

        print("  Validation directory not found.")
        print("  Training data will be split automatically.")

        return train_dir, None

    # --------------------------------------------------------
    # Structure 2: class folders directly inside domain
    # --------------------------------------------------------

    if not os.path.isdir(domain_path):
        return None, None

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    )

    class_folder_found = False

    for item in os.listdir(domain_path):

        item_path = os.path.join(domain_path, item)

        if not os.path.isdir(item_path):
            continue

        for file_name in os.listdir(item_path):

            if file_name.lower().endswith(valid_extensions):
                class_folder_found = True
                break

        if class_folder_found:
            break

    if class_folder_found:

        print("  Found class-folder structure.")
        print("  Automatic 80/20 train-validation split will be used.")

        return domain_path, None

    return None, None


# ============================================================
# CREATE DATA TRANSFORMS
# ============================================================

def create_transform():

    return transforms.Compose([
        transforms.Resize(
            (config.IMG_SIZE, config.IMG_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=config.NORM_MEAN,
            std=config.NORM_STD
        ),
    ])


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    epoch,
    class_names,
    accuracy,
    checkpoint_path
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": accuracy,
        },
        checkpoint_path
    )


# ============================================================
# TRAIN ONE SPECIALIST
# ============================================================

def train_specialist(
    domain_name,
    data_dir,
    epochs=40,
    patience=7
):

    separator = "=" * 70

    print()
    print(separator)
    print(f"Training specialist: {domain_name}")
    print(separator)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    domain_path = os.path.join(
        data_dir,
        domain_name
    )

    if not os.path.isdir(domain_path):

        print(
            f"Skipping {domain_name}: "
            f"domain directory does not exist."
        )

        return

    train_dir, val_dir = find_data_structure(
        domain_path
    )

    if train_dir is None:

        print(
            f"Skipping {domain_name}: "
            f"no valid image dataset found."
        )

        return

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = create_transform()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    try:

        full_dataset = datasets.ImageFolder(
            train_dir,
            transform=transform
        )

    except Exception as error:

        print(
            f"ERROR loading dataset for {domain_name}:"
        )
        print(error)

        return

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if len(full_dataset) == 0:

        print(
            f"Skipping {domain_name}: "
            f"dataset contains no images."
        )

        return

    num_classes = len(
        full_dataset.classes
    )

    if num_classes < 2:

        print(
            f"Skipping {domain_name}: "
            f"at least 2 classes are required."
        )

        print(
            f"Classes found: "
            f"{full_dataset.classes}"
        )

        return

    print(
        f"Classes: {full_dataset.classes}"
    )

    print(
        f"Total images: {len(full_dataset)}"
    )

    # --------------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------------

    if val_dir is None:

        if len(full_dataset) < 2:

            print(
                "Not enough images to create "
                "train/validation split."
            )

            return

        train_size = int(
            0.8 * len(full_dataset)
        )

        val_size = (
            len(full_dataset) - train_size
        )

        # Make sure both sets contain at least 1 image
        if train_size == 0:
            train_size = 1
            val_size = len(full_dataset) - 1

        if val_size == 0:
            val_size = 1
            train_size = len(full_dataset) - 1

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        print(
            f"Train images: {len(train_dataset)}"
        )

        print(
            f"Validation images: {len(val_dataset)}"
        )

    else:

        train_dataset = full_dataset

        try:

            val_dataset = datasets.ImageFolder(
                val_dir,
                transform=transform
            )

        except Exception as error:

            print(
                f"ERROR loading validation dataset:"
            )
            print(error)

            return

        if len(val_dataset) == 0:

            print(
                "Validation dataset is empty."
            )

            val_dataset = None

        else:

            print(
                f"Train images: {len(train_dataset)}"
            )

            print(
                f"Validation images: {len(val_dataset)}"
            )

            # Make sure validation classes match training classes
            if val_dataset.classes != full_dataset.classes:

                print(
                    "WARNING: Validation classes do not "
                    "exactly match training classes."
                )

                print(
                    f"Train classes: "
                    f"{full_dataset.classes}"
                )

                print(
                    f"Val classes: "
                    f"{val_dataset.classes}"
                )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = None

    if val_dataset is not None:

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = config.DEVICE

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    try:

        model = MSANet(
            num_classes=num_classes,
            base_channels=config.BASE_CHANNELS
        ).to(device)

    except Exception as error:

        print(
            "ERROR creating MSANet model:"
        )
        print(error)

        return

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    checkpoint_dir = os.path.join(
        "checkpoints",
        domain_name
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "best_model.pth"
    )

    best_val_acc = 0.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, epochs + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0
        train_batches = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        if train_batches == 0:

            print(
                "ERROR: Training DataLoader "
                "contains no batches."
            )

            return

        train_loss /= train_batches

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if val_loader is not None:

            model.eval()

            val_loss = 0.0
            correct = 0
            total = 0
            val_batches = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    val_loss += loss.item()
                    val_batches += 1

                    predictions = outputs.argmax(
                        dim=1
                    )

                    total += labels.size(0)

                    correct += (
                        predictions == labels
                    ).sum().item()

            if val_batches == 0:

                print(
                    "ERROR: Validation DataLoader "
                    "contains no batches."
                )

                return

            val_loss /= val_batches

            val_acc = (
                100.0 * correct / total
                if total > 0
                else 0.0
            )

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # ------------------------------------------------
            # SAVE BEST MODEL
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                save_checkpoint(
                    model=model,
                    epoch=epoch,
                    class_names=full_dataset.classes,
                    accuracy=val_acc,
                    checkpoint_path=checkpoint_path
                )

                print(
                    f"  ✓ Best model saved "
                    f"({best_val_acc:.2f}%)"
                )

            # ------------------------------------------------
            # EARLY STOPPING
            # ------------------------------------------------

            early_stopping(val_loss)

            if early_stopping.early_stop:

                print(
                    f"  Early stopping at "
                    f"epoch {epoch}"
                )

                break

        # ----------------------------------------------------
        # NO VALIDATION DATA
        # ----------------------------------------------------

        else:

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}"
            )

            # Save every 5 epochs
            if epoch % 5 == 0:

                save_checkpoint(
                    model=model,
                    epoch=epoch,
                    class_names=full_dataset.classes,
                    accuracy=0.0,
                    checkpoint_path=checkpoint_path
                )

                print(
                    f"  ✓ Checkpoint saved "
                    f"at epoch {epoch}"
                )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print(
        f"Finished training: {domain_name}"
    )

    if val_loader is not None:

        print(
            f"Best validation accuracy: "
            f"{best_val_acc:.2f}%"
        )

    if os.path.exists(checkpoint_path):

        print(
            f"Model saved to: "
            f"{checkpoint_path}"
        )

    print(separator)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MANUFACTURING DEFECT CLASSIFICATION")
    print("SPECIALIST MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset directory
    # --------------------------------------------------------

    data_dir = "datasets"

    if not os.path.isdir(data_dir):

        print()
        print(
            f"ERROR: Dataset directory "
            f"'{data_dir}' was not found."
        )

        print(
            "Make sure you are running this script "
            "from the project folder."
        )

        return

    # --------------------------------------------------------
    # Find domains
    # --------------------------------------------------------

    excluded_folders = {
        "domain_classifier",
        "__pycache__"
    }

    domains = []

    for item in sorted(
        os.listdir(data_dir)
    ):

        item_path = os.path.join(
            data_dir,
            item
        )

        if (
            os.path.isdir(item_path)
            and item not in excluded_folders
        ):

            domains.append(item)

    if not domains:

        print()
        print(
            "ERROR: No domain folders were found "
            "inside datasets/."
        )

        print()
        print(
            "Expected example:"
        )

        print(
            "datasets/"
        )

        print(
            "  domain1/"
        )

        print(
            "  domain2/"
        )

        return

    print()
    print(
        f"Found domains: {domains}"
    )

    # --------------------------------------------------------
    # Check existing models
    # --------------------------------------------------------

    print()
    print(
        "Checking existing checkpoints..."
    )

    trained_domains = []

    for domain in domains:

        checkpoint_path = os.path.join(
            "checkpoints",
            domain,
            "best_model.pth"
        )

        if os.path.isfile(checkpoint_path):

            trained_domains.append(domain)

    if trained_domains:

        print(
            f"Already trained: {trained_domains}"
        )

    else:

        print(
            "Already trained: none"
        )

    # --------------------------------------------------------
    # Train remaining domains
    # --------------------------------------------------------

    for domain in domains:

        checkpoint_path = os.path.join(
            "checkpoints",
            domain,
            "best_model.pth"
        )

        if os.path.isfile(checkpoint_path):

            print()
            print(
                f"Skipping {domain}: "
                f"best_model.pth already exists."
            )

            continue

        try:

            train_specialist(
                domain_name=domain,
                data_dir=data_dir,
                epochs=40,
                patience=7
            )

        except KeyboardInterrupt:

            print()
            print(
                "Training interrupted by user."
            )

            return

        except Exception as error:

            print()
            print(
                f"ERROR while training "
                f"{domain}:"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            print(
                "Moving to the next domain..."
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL SPECIALIST TRAINING COMPLETED")
    print("=" * 70)


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()