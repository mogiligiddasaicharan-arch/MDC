import os, json, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_solder\SolDef_AI\Labeled"
DOMAIN = "solder"
random.seed(42)

buckets = {}  # label -> list of image filenames
for f in os.listdir(SRC):
    if not f.endswith(".json"):
        continue
    base = os.path.splitext(f)[0]
    img_file = base + ".jpg"
    if not os.path.exists(os.path.join(SRC, img_file)):
        continue
    data = json.load(open(os.path.join(SRC, f)))
    shapes = data.get("shapes", [])
    if not shapes:
        continue
    label = shapes[0].get("label")
    if label is None:
        continue
    buckets.setdefault(label, []).append(img_file)

for label, files in buckets.items():
    split_and_copy(files, SRC, DOMAIN, label)

print("Done.")