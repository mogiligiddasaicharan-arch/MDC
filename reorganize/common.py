import os, random, shutil

def split_and_copy(files, src_dir, dst_domain, cls, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Splits a flat list of filenames into train/val/test and copies them
    into datasets/<domain>/<split>/<cls>/. Use when the raw dataset has NO
    pre-existing train/test split."""
    random.seed(seed)
    files = list(files)
    random.shuffle(files)
    n = len(files)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    train_files = files[n_val + n_test:]

    for split, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
        outdir = os.path.join("datasets", dst_domain, split, cls)
        os.makedirs(outdir, exist_ok=True)
        for f in flist:
            shutil.copy(os.path.join(src_dir, f), os.path.join(outdir, f))

    print(f"{cls}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")


def copy_presplit(files, src_dir, dst_domain, split, cls):
    """Copies files into datasets/<domain>/<split>/<cls>/ when the raw
    dataset ALREADY has its own train/val/test-like split (like NEU)."""
    outdir = os.path.join("datasets", dst_domain, split, cls)
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(outdir, f))
    print(f"{cls} [{split}]: {len(files)} files")