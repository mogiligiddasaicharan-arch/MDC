import os, random, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = r"raw_solar2\Faulty_solar_panel"
DOMAIN = "solar_panel"
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
random.seed(42)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

CLASS_MAP = {
    "Bird-drop": "bird_drop",
    "Clean": "clean",
    "Dusty": "dusty",
    "Electrical-damage": "electrical_damage",
    "Physical-Damage": "physical_damage",
    "Snow-Covered": "snow_covered",
}

for raw_cls, clean_cls in CLASS_MAP.items():
    cls_dir = os.path.join(SRC, raw_cls)
    pool = []  # (src_dir, filename)
    for root, dirs, files in os.walk(cls_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                pool.append((root, f))

    random.shuffle(pool)
    n = len(pool)
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)
    splits = {
        "val": pool[:n_val],
        "test": pool[n_val:n_val+n_test],
        "train": pool[n_val+n_test:],
    }
    for split, items in splits.items():
        outdir = os.path.join("datasets", DOMAIN, split, clean_cls)
        os.makedirs(outdir, exist_ok=True)
        for src_dir, fn in items:
            dst = os.path.join(outdir, fn)
            if os.path.exists(dst):
                dst = os.path.join(outdir, f"{src_dir.replace(os.sep,'_')}_{fn}")
            shutil.copy(os.path.join(src_dir, fn), dst)
    print(f"{clean_cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

print("Done.")