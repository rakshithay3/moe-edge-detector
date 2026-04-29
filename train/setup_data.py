"""Copy images from ../datasets/ into data/train/ and data/val/ ImageFolder layout.

The source datasets are in YOLO format with images in train/images/, valid/images/, test/images/.
We flatten these into an ImageFolder structure:
    data/train/<class_name>/<image_files>
    data/val/<class_name>/<image_files>

Test images are merged into train for more training data.
Dishwasher only has test images, so those go to train with a small val split.

Usage:
    python train/setup_data.py
"""

import os
import shutil
import random

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "datasets")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "air_conditioner",
    "air_purifier",
    "dishwasher",
    "microwave",
    "refrigerator",
    "tv",
    "washing_machine",
]


def collect_images(directory):
    """Collect all image files from a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def setup():
    print(f"Source: {os.path.abspath(DATASETS_DIR)}")
    print(f"Target: {os.path.abspath(os.path.join(BASE_DIR, 'data'))}\n")

    for cls in CLASSES:
        cls_dir = os.path.join(DATASETS_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ✗ {cls}: directory not found, skipping")
            continue

        # Collect from each split
        train_imgs = collect_images(os.path.join(cls_dir, "train", "images"))
        val_imgs = collect_images(os.path.join(cls_dir, "valid", "images"))
        test_imgs = collect_images(os.path.join(cls_dir, "test", "images"))

        # Merge test into train for more data
        train_imgs += test_imgs

        # Handle classes with no train/val (like dishwasher)
        if not train_imgs and not val_imgs:
            print(f"  ✗ {cls}: no images found, skipping")
            continue

        if not val_imgs and train_imgs:
            # Split off 15% for validation
            random.seed(42)
            random.shuffle(train_imgs)
            split = max(1, int(len(train_imgs) * 0.15))
            val_imgs = train_imgs[:split]
            train_imgs = train_imgs[split:]

        # Create output dirs
        train_out = os.path.join(TRAIN_DIR, cls)
        val_out = os.path.join(VAL_DIR, cls)
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(val_out, exist_ok=True)

        # Copy train images
        for i, src in enumerate(train_imgs):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(train_out, f"{cls}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        # Copy val images
        for i, src in enumerate(val_imgs):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(val_out, f"{cls}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        print(f"  ✓ {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

    # Summary
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR)]:
        print(f"\n{split_name}:")
        if os.path.exists(split_dir):
            total = 0
            for cls in sorted(os.listdir(split_dir)):
                cls_path = os.path.join(split_dir, cls)
                if os.path.isdir(cls_path):
                    count = len(collect_images(cls_path))
                    total += count
                    print(f"  {cls}: {count}")
            print(f"  TOTAL: {total}")


if __name__ == "__main__":
    setup()
