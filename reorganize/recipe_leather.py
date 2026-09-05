import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_leather\Leather Defect Classification"
DOMAIN = "leather"
random.seed(42)

CLASS_MAP = {
    "Folding marks": "folding_marks",
    "Grain off": "grain_off",
    "Growth marks": "growth_marks",
    "loose grains": "loose_grains",
    "non defective": "non_defective",
    "pinhole": "pinhole",
}

for raw_cls, clean_cls in CLASS_MAP.items():
    cls_dir = os.path.join(SRC, raw_cls)
    files = os.listdir(cls_dir)
    split_and_copy(files, cls_dir, DOMAIN, clean_cls)

print("Done.")