"""Build domain classifier dataset by sampling images from all specialist domains.

Creates:
    datasets/domain_classifier/train/<domain>/ *.jpg
    datasets/domain_classifier/val/<domain>/   *.jpg
    datasets/domain_classifier/test/<domain>/  *.jpg

Usage:
    python build_domain_classifier_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

DOMAINS = [
    "casting", "concrete", "fabric", "gc10", "leather", "magnetic",
    "mvtec_bottle", "mvtec_cable", "mvtec_capsule", "mvtec_carpet",
    "mvtec_grid", "mvtec_hazelnut", "mvtec_leather", "mvtec_metal_nut",
    "mvtec_pill", "mvtec_screw", "mvtec_tile", "mvtec_toothbrush",
    "mvtec_transistor", "mvtec_wood", "mvtec_zipper",
    "pcb", "pcb2", "pcb_aoi", "severstal", "solar_elpv", "solar_panel",
    "solder", "steel", "welding",
]

SRC_ROOT = "datasets"
DST_ROOT = "datasets/domain_classifier"
SAMPLES_PER_DOMAIN_PER_SPLIT = {
    "train": 200,
    "val": 40,
    "test": 40,
}


def build():
    if os.path.exists(DST_ROOT):
        print(f"Removing old {DST_ROOT} ...")
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DST_ROOT, split), exist_ok=True)

    for domain in DOMAINS:
        for split in ["train", "val", "test"]:
            src_dir = os.path.join(SRC_ROOT, domain, split)
            if not os.path.isdir(src_dir):
                print(f"  SKIP: {src_dir} not found")
                continue

            # Collect all images recursively
            images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                images.extend(Path(src_dir).rglob(ext))
            images = [p for p in images if p.is_file()]
            images.sort(key=lambda p: p.name)

            if not images:
                print(f"  SKIP: no images in {src_dir}")
                continue

            n = SAMPLES_PER_DOMAIN_PER_SPLIT[split]
            if len(images) < n:
                # If not enough unique images, sample with replacement
                sampled = random.choices(images, k=n)
                print(f"  {domain}/{split}: {len(images)} unique, sampling {n} with replacement")
            else:
                sampled = random.sample(images, n)
                print(f"  {domain}/{split}: sampled {n}/{len(images)}")

            dst_split_dir = os.path.join(DST_ROOT, split, domain)
            os.makedirs(dst_split_dir, exist_ok=True)

            for i, img_path in enumerate(sampled):
                dst_name = f"{domain}_{split}_{i:04d}{img_path.suffix.lower()}"
                shutil.copy2(str(img_path), os.path.join(dst_split_dir, dst_name))

    print(f"\nDomain classifier dataset built at: {DST_ROOT}")
    for split in ["train", "val", "test"]:
        domains = os.listdir(os.path.join(DST_ROOT, split))
        total = sum(len(os.listdir(os.path.join(DST_ROOT, split, d))) for d in domains)
        print(f"  {split}: {len(domains)} domains, {total} images")


if __name__ == "__main__":
    build()
