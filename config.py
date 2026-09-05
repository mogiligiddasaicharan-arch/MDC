"""
Central configuration for the Manufacturing Defect Classification project.
Edit values here rather than scattering magic numbers through the codebase.
"""

import os
import torch

# ---------------- Paths ----------------
DATA_DIR = "dataset"          # expects DATA_DIR/train, /val, /test
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Data ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2               # set to 0 on Windows if you hit multiprocessing errors

# If your dataset is grayscale (like NEU-CLS), we still load as 3-channel
# (replicated) so pretrained-style normalization stays consistent.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------- Model ----------------
NUM_CLASSES = None            # auto-detected from dataset folder at runtime
BASE_CHANNELS = 32            # width of first MSA block; doubles each stage

# ---------------- Training ----------------
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTHING = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
