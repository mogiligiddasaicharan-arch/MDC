import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reorganize.common import split_and_copy

SRC = r"raw_mvtec"
random.seed(42)

categories = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]

for cat in categories:
    cat_dir = os.path.join(SRC, cat)
    train_good_dir = os.path.join(cat_dir, "train", "good")
    test_dir = os.path.join(cat_dir, "test")

    domain = f"mvtec_{cat}"
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    for cls in classes:
        # gather files, tagging source dir since good comes from two places
        pool = []  # list of (full_dir, filename)
        test_cls_dir = os.path.join(test_dir, cls)
        for f in os.listdir(test_cls_dir):
            pool.append((test_cls_dir, f))
        if cls == "good":
            for f in os.listdir(train_good_dir):
                pool.append((train_good_dir, f))

        # split_and_copy expects one src_dir; since "good" pools two dirs,
        # copy manually here reusing the same split logic inline
        filenames = [f"{i}__{fn}" for i, (_, fn) in enumerate(pool)]
        # build a temp mapping and copy directly
        import shutil
        random.shuffle(pool)
        n = len(pool)
        n_val = int(n * 0.15)
        n_test = int(n * 0.15)
        splits = {
            "val": pool[:n_val],
            "test": pool[n_val:n_val+n_test],
            "train": pool[n_val+n_test:],
        }
        for split, items in splits.items():
            outdir = os.path.join("datasets", domain, split, cls)
            os.makedirs(outdir, exist_ok=True)
            for src_dir, fn in items:
                shutil.copy(os.path.join(src_dir, fn), os.path.join(outdir, fn))
        print(f"{domain}/{cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

print("Done.")