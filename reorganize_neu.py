import os, random, shutil

SRC_TRAIN = r"raw_neu\NEU-DET\train\images"
SRC_TEST  = r"raw_neu\NEU-DET\validation\images"
DST = r"datasets\steel"
VAL_FRACTION = 0.15
random.seed(42)

classes = os.listdir(SRC_TRAIN)

for cls in classes:
    files = os.listdir(os.path.join(SRC_TRAIN, cls))
    random.shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    val_files = files[:n_val]
    train_files = files[n_val:]

    for split, flist, srcdir in [("train", train_files, SRC_TRAIN), ("val", val_files, SRC_TRAIN)]:
        outdir = os.path.join(DST, split, cls)
        os.makedirs(outdir, exist_ok=True)
        for f in flist:
            shutil.copy(os.path.join(srcdir, cls, f), os.path.join(outdir, f))

    test_files = os.listdir(os.path.join(SRC_TEST, cls))
    outdir = os.path.join(DST, "test", cls)
    os.makedirs(outdir, exist_ok=True)
    for f in test_files:
        shutil.copy(os.path.join(SRC_TEST, cls, f), os.path.join(outdir, f))

    print(f"{cls}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

print("Done.")