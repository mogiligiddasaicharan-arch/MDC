import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import copy_presplit

SRC = r"raw_pcb2\DsPCBSD+\Data_YOLO"
DOMAIN = "pcb2"
random.seed(42)

CLASS_NAMES = {
    0: "SH", 1: "SP", 2: "SC", 3: "OP", 4: "MB",
    5: "HB", 6: "CS", 7: "CFO", 8: "BMFO",
}

# images/val + labels/val becomes our TEST set (already held out)
# images/train + labels/train gets split into train/val internally
def load_buckets(img_dir, lbl_dir):
    buckets = {}
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
    return buckets

# TEST split (from val folder, unmodified)
test_img_dir = os.path.join(SRC, "images", "val")
test_lbl_dir = os.path.join(SRC, "labels", "val")
test_buckets = load_buckets(test_img_dir, test_lbl_dir)
for cls_name, files in test_buckets.items():
    copy_presplit(files, test_img_dir, DOMAIN, "test", cls_name)

# TRAIN/VAL split (from train folder, 85/15)
train_img_dir = os.path.join(SRC, "images", "train")
train_lbl_dir = os.path.join(SRC, "labels", "train")
train_buckets = load_buckets(train_img_dir, train_lbl_dir)
for cls_name, files in train_buckets.items():
    random.shuffle(files)
    n_val = int(len(files) * 0.15)
    val_files = files[:n_val]
    train_files = files[n_val:]
    copy_presplit(val_files, train_img_dir, DOMAIN, "val", cls_name)
    copy_presplit(train_files, train_img_dir, DOMAIN, "train", cls_name)

print("Done.")