import os, csv, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC_IMG = r"raw_severstal\train_images_sliced\train_images_sliced"
CSV_PATH = r"raw_severstal\train_sliced_ALL.csv"
DOMAIN = "severstal"
random.seed(42)

# Build image -> set of defect classes present (non-empty EncodedPixels)
image_classes = {}
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:    
    reader = csv.DictReader(f)
    for row in reader:
        img = row["ImageId"]
        cls_id = row["ClassId"]
        pixels = row["EncodedPixels"].strip()
        if img not in image_classes:
            image_classes[img] = set()
        if pixels:
            image_classes[img].add(cls_id)

# Assign one label per image: no_defect, or defect_<classid> if exactly one
# class present, or skip images with multiple simultaneous defect classes
# (rare and ambiguous for a single-label classifier).
buckets = {}  # class label -> list of filenames
skipped_multi = 0
for img, classes in image_classes.items():
    if len(classes) == 0:
        label = "no_defect"
    elif len(classes) == 1:
        label = f"defect_{next(iter(classes))}"
    else:
        skipped_multi += 1
        continue
    buckets.setdefault(label, []).append(img)

print(f"Skipped {skipped_multi} images with multiple simultaneous defect classes")

for label, files in buckets.items():
    # only keep files that actually exist on disk (safety check)
    files = [f for f in files if os.path.exists(os.path.join(SRC_IMG, f))]
    split_and_copy(files, SRC_IMG, DOMAIN, label)

print("Done.")