import os, random, sys
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import copy_presplit

BASE = r"raw_pcb_aoi"
DOMAIN = "pcb_aoi"
random.seed(42)

def load_buckets(split_dir):
    ann_dir = os.path.join(split_dir, "Annotations")
    img_dir = os.path.join(split_dir, "JPEGImages")
    buckets = {}
    for f in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, f))
        objs = tree.findall("object")
        if not objs:
            continue
        cls_name = objs[0].find("name").text
        base = os.path.splitext(f)[0]
        img_file = None
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidate = base + ext
            if os.path.exists(os.path.join(img_dir, candidate)):
                img_file = candidate
                break
        if img_file is None:
            continue
        buckets.setdefault(cls_name, []).append(img_file)
    return buckets, img_dir

# TEST split from test_data (unmodified)
test_buckets, test_img_dir = load_buckets(os.path.join(BASE, "test_data"))
for cls, files in test_buckets.items():
    copy_presplit(files, test_img_dir, DOMAIN, "test", cls)

# TRAIN/VAL from train_data + train_data_augmentation combined, 85/15 split
train_buckets = {}
for sub in ["train_data", "train_data_augmentation"]:
    b, img_dir = load_buckets(os.path.join(BASE, sub))
    for cls, files in b.items():
        train_buckets.setdefault(cls, []).extend([(img_dir, f) for f in files])

for cls, items in train_buckets.items():
    random.shuffle(items)
    n_val = int(len(items) * 0.15)
    val_items = items[:n_val]
    train_items = items[n_val:]
    # copy_presplit expects one src_dir; since images may come from 2 dirs, copy manually
    import shutil
    for split, group in [("val", val_items), ("train", train_items)]:
        outdir = os.path.join("datasets", DOMAIN, split, cls)
        os.makedirs(outdir, exist_ok=True)
        for src_dir, fn in group:
            shutil.copy(os.path.join(src_dir, fn), os.path.join(outdir, fn))
    print(f"{cls} [train/val]: train={len(train_items)} val={len(val_items)}")

print("Done.")