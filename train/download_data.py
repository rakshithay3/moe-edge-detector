"""Download datasets from Roboflow and organize into ImageFolder structure.

Downloads from 2 workspaces. Uses 'multiclass' format for classification
projects and 'yolov5pytorch' for object-detection projects, then extracts
images into a balanced ImageFolder layout.

Usage:
    python train/download_data.py
"""

import os
import shutil
import random
import glob

from roboflow import Roboflow

# ── Config ───────────────────────────────────────────────────────
MAX_IMAGES_PER_CLASS = 150  # Cap for balancing

DOWNLOADS = [
    {
        "api_key": "KePGmBlSS2JkQYwdakn6",
        "workspace": "rakshithas-workspace",
        "project": "television-gdkgp-i4jjn",
        "version": 1,
        "target_class": "tv",
        "format": "yolov5pytorch",  # object-detection project
    },
    {
        "api_key": "KePGmBlSS2JkQYwdakn6",
        "workspace": "rakshithas-workspace",
        "project": "refrigerator-ib7go-ijfst",
        "version": 1,
        "target_class": "refrigerator",
        "format": "yolov5pytorch",  # object-detection project
    },
    {
        "api_key": "ugnLNKRNpGSQotLNl2Ua",
        "workspace": "joans-workspace-0ta4b",
        "project": "alpa-dataset-ac",
        "version": 1,
        "target_class": "air_conditioner",
        "format": "yolov5pytorch",  # object-detection project
    },
]

# ── Output directories ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
TEMP_DIR = os.path.join(BASE_DIR, "data", "_temp_downloads")

for d in [TRAIN_DIR, VAL_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)


def collect_images(source_dir):
    """Recursively find all image files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for root, _, files in os.walk(source_dir):
        # Skip label directories
        if "labels" in root.lower():
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(root, f))
    return images


def download_and_organize():
    """Download all datasets and organize into balanced ImageFolder."""

    for cfg in DOWNLOADS:
        target = cfg["target_class"]
        dl_format = cfg.get("format", "yolov5pytorch")
        print(f"\n{'='*60}")
        print(f"Downloading: {cfg['project']} → {target} (format: {dl_format})")
        print(f"{'='*60}")

        # Download via Roboflow SDK
        rf = Roboflow(api_key=cfg["api_key"])
        project = rf.workspace(cfg["workspace"]).project(cfg["project"])
        download_loc = os.path.join(TEMP_DIR, target)

        try:
            project.version(cfg["version"]).download(
                dl_format,
                location=download_loc,
                overwrite=True,
            )
        except Exception as e:
            print(f"  ⚠ Failed with format '{dl_format}': {e}")
            # Try alternative formats
            for alt_fmt in ["multiclass", "folder", "coco"]:
                if alt_fmt == dl_format:
                    continue
                try:
                    print(f"  Retrying with format '{alt_fmt}'...")
                    project.version(cfg["version"]).download(
                        alt_fmt,
                        location=download_loc,
                        overwrite=True,
                    )
                    break
                except Exception as e2:
                    print(f"  ⚠ Also failed with '{alt_fmt}': {e2}")
                    continue

        # Collect all downloaded images (skip label files)
        all_images = collect_images(download_loc)
        print(f"  Found {len(all_images)} images total")

        if not all_images:
            print(f"  ✗ No images found for {target}, skipping!")
            continue

        # Separate train/val from Roboflow splits
        train_images = [p for p in all_images if "/train/" in p]
        val_images = [p for p in all_images if "/valid/" in p or "/val/" in p]
        test_images = [p for p in all_images if "/test/" in p]

        # If no split structure, use all as train
        if not train_images and not val_images:
            # Random split: 85% train, 15% val
            random.seed(42)
            random.shuffle(all_images)
            split_idx = int(len(all_images) * 0.85)
            train_images = all_images[:split_idx]
            val_images = all_images[split_idx:]
        elif not train_images:
            train_images = all_images

        # Add test to train (we do our own splits via val)
        train_images += test_images

        print(f"  Train: {len(train_images)}, Val: {len(val_images)}")

        # Cap for balance
        if len(train_images) > MAX_IMAGES_PER_CLASS:
            random.seed(42)
            train_images = random.sample(train_images, MAX_IMAGES_PER_CLASS)
            print(f"  Capped train to {MAX_IMAGES_PER_CLASS} for balance")

        # Cap val proportionally
        max_val = max(15, int(MAX_IMAGES_PER_CLASS * 0.15))
        if len(val_images) > max_val:
            random.seed(42)
            val_images = random.sample(val_images, max_val)

        # Copy to final directories
        train_target = os.path.join(TRAIN_DIR, target)
        val_target = os.path.join(VAL_DIR, target)
        os.makedirs(train_target, exist_ok=True)
        os.makedirs(val_target, exist_ok=True)

        for i, src in enumerate(train_images):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(train_target, f"{target}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        for i, src in enumerate(val_images):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(val_target, f"{target}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        print(f"  ✓ Saved {len(train_images)} train, {len(val_images)} val → data/")

    # Cleanup temp downloads
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR)]:
        print(f"\n{split_name}:")
        if os.path.exists(split_dir):
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir)
                                if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
                    print(f"  {cls}: {count} images")


if __name__ == "__main__":
    download_and_organize()
