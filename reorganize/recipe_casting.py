import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import copy_presplit

SRC = r"raw_casting\casting_data\casting_data"
DOMAIN = "casting"
VAL_FRACTION = 0.15
random.seed(42)

# map raw class folder names -> clean class names
CLASS_MAP = {"def_front": "defective", "ok_front": "ok"}

for raw_cls, clean_cls in CLASS_MAP.items():
    # test split: copy as-is
    test_dir = os.path.join(SRC, "test", raw_cls)
    test_files = os.listdir(test_dir)
    copy_presplit(test_files, test_dir, DOMAIN, "test", clean_cls)

    # train split: carve out val
    train_dir = os.path.join(SRC, "train", raw_cls)
    train_files = os.listdir(train_dir)
    random.shuffle(train_files)
    n_val = int(len(train_files) * VAL_FRACTION)
    val_files = train_files[:n_val]
    train_files = train_files[n_val:]

    copy_presplit(val_files, train_dir, DOMAIN, "val", clean_cls)
    copy_presplit(train_files, train_dir, DOMAIN, "train", clean_cls)

print("Done.")