import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import copy_presplit

SRC = r"raw_welding\The Welding Defect Dataset\The Welding Defect Dataset"
DOMAIN = "welding"

CLASS_NAMES = {0: "bad_weld", 1: "good_weld", 2: "defect"}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

for raw_split, clean_split in SPLIT_MAP.items():
    img_dir = os.path.join(SRC, raw_split, "images")
    lbl_dir = os.path.join(SRC, raw_split, "labels")

    buckets = {}  # class_name -> list of image filenames
    for img_file in os.listdir(img_dir):
        base = os.path.splitext(img_file)[0]
        lbl_file = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_file):
            continue
        with open(lbl_file) as f:
            line = f.readline().strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        cls_name = CLASS_NAMES.get(cls_id)
        if cls_name is None:
            continue
        buckets.setdefault(cls_name, []).append(img_file)

    for cls_name, files in buckets.items():
        copy_presplit(files, img_dir, DOMAIN, clean_split, cls_name)

print("Done.")