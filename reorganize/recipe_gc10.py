import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_gc10\images\images"
DOMAIN = "gc10"
random.seed(42)

classes = os.listdir(SRC)

for cls in classes:
    cls_dir = os.path.join(SRC, cls)
    files = os.listdir(cls_dir)
    clean_cls = cls.strip().replace(" ", "_")
    split_and_copy(files, cls_dir, DOMAIN, clean_cls)

print("Done.")