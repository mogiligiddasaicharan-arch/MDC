import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_fabric\Fabric Defects Dataset\Fabric Defect Dataset"
DOMAIN = "fabric"
random.seed(42)

CLASS_MAP = {
    "defect free": "defect_free",
    "hole": "hole",
    "horizontal": "horizontal",
    "lines": "lines",
    "stain": "stain",
    "Vertical": "vertical",
}

for raw_cls, clean_cls in CLASS_MAP.items():
    cls_dir = os.path.join(SRC, raw_cls)
    files = os.listdir(cls_dir)
    split_and_copy(files, cls_dir, DOMAIN, clean_cls)

print("Done.")