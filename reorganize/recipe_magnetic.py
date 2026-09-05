import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_magnetic"
DOMAIN = "magnetic"
random.seed(42)

CLASS_MAP = {
    "MT_Blowhole": "blowhole",
    "MT_Break": "break",
    "MT_Crack": "crack",
    "MT_Fray": "fray",
    "MT_Free": "free",
    "MT_Uneven": "uneven",
}

for raw_cls, clean_cls in CLASS_MAP.items():
    cls_dir = os.path.join(SRC, raw_cls, "Imgs")
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
    split_and_copy(files, cls_dir, DOMAIN, clean_cls)

print("Done.")