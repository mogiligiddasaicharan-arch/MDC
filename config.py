"""Central configuration for the Manufacturing Defect Classification project."""

import os
import torch

# ---------------- Paths ----------------
DATA_DIR = "datasets/steel"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Data ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------- Model ----------------
NUM_CLASSES = None
BASE_CHANNELS = 32

# ---------------- Training ----------------
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTHING = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
