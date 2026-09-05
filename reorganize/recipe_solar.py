import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC_ROOT = r"raw_solar"
LABELS = r"raw_solar\labels.csv"
DOMAIN = "solar_elpv"
random.seed(42)

PROB_LABELS = {
    "0.0": "functional",
    "0.3333333333333333": "partial_defect_low",
    "0.6666666666666666": "partial_defect_high",
    "1.0": "defective",
}

buckets = {}  # class label -> list of (dir, filename)
with open(LABELS) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 3:
            continue
        rel_path, prob, wafer = parts[0], parts[1], parts[2]
        prob_label = PROB_LABELS.get(prob)
        if prob_label is None:
            continue
        label = f"{prob_label}_{wafer}"
        full_dir = os.path.join(SRC_ROOT, os.path.dirname(rel_path))
        filename = os.path.basename(rel_path)
        buckets.setdefault(label, []).append((full_dir, filename))

for label, items in buckets.items():
    # split_and_copy needs one src_dir + filenames; all live under raw_solar/images
    src_dir = items[0][0]
    files = [fn for _, fn in items]
    split_and_copy(files, src_dir, DOMAIN, label)

print("Done.")