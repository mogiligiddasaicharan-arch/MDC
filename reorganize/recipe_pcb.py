import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_pcb\PCB_DATASET\images"
DOMAIN = "pcb"
random.seed(42)

classes = os.listdir(SRC)

for cls in classes:
    cls_dir = os.path.join(SRC, cls)
    files = os.listdir(cls_dir)
    split_and_copy(files, cls_dir, DOMAIN, cls)

print("Done.")