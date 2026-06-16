"""
Prepare drone imagery for YOLO training.

Supports:
  1. YOLO txt labels already in data/raw/annotations/
  2. A single COCO JSON at data/raw/annotations/annotations.json

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running as: python scripts/prepare_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import (
    ensure_dir,
    list_images,
    load_coco_annotations,
    load_config,
    write_yolo_dataset_yaml,
)


def convert_coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    labels_out: Path,
) -> int:
    """Write one .txt label file per image from COCO JSON."""
    ensure_dir(labels_out)
    mapping = load_coco_annotations(coco_json)
    written = 0

    for img_path in list_images(images_dir, [".jpg", ".jpeg", ".png", ".tif", ".tiff"]):
        boxes = mapping.get(img_path.name, [])
        label_path = labels_out / f"{img_path.stem}.txt"
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            written += 1

    return written


def copy_yolo_pairs(
    images_dir: Path,
    labels_dir: Path,
    images_out: Path,
    labels_out: Path,
) -> list[Path]:
    """Copy image/label pairs where a label file exists."""
    ensure_dir(images_out)
    ensure_dir(labels_out)
    paired: list[Path] = []

    for img_path in list_images(images_dir, [".jpg", ".jpeg", ".png", ".tif", ".tiff"]):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil_copy(img_path, images_out / img_path.name)
        shutil_copy(label_path, labels_out / f"{img_path.stem}.txt")
        paired.append(img_path)

    return paired


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)


def split_dataset(
    image_paths: list[Path],
    labels_dir: Path,
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, int]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    counts: dict[str, int] = {}
    for split_name, paths in splits.items():
        images_out = dataset_root / "images" / split_name
        labels_out = dataset_root / "labels" / split_name
        ensure_dir(images_out)
        ensure_dir(labels_out)

        for img_path in paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            shutil_copy(img_path, images_out / img_path.name)
            if label_path.exists():
                shutil_copy(label_path, labels_out / f"{img_path.stem}.txt")

        counts[split_name] = len(paths)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from drone images")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_images = Path(cfg["paths"]["raw_images"])
    raw_annotations = Path(cfg["paths"]["raw_annotations"])
    dataset_root = Path(cfg["paths"]["dataset_root"])
    staging_labels = dataset_root / "labels" / "all"
    staging_images = dataset_root / "images" / "all"

    # Clean up old splits to avoid dataset contamination from previous runs
    import shutil
    for folder in ["images", "labels"]:
        path = dataset_root / folder
        if path.exists():
            print(f"Cleaning existing split directory: {path}")
            shutil.rmtree(path)

    ensure_dir(raw_images)
    ensure_dir(raw_annotations)

    coco_json = raw_annotations / "annotations.json"
    if coco_json.exists():
        print(f"Converting COCO annotations: {coco_json}")
        n = convert_coco_to_yolo(coco_json, raw_images, staging_labels)
        print(f"  Wrote labels for {n} annotated images")
        labels_source = staging_labels
        # Copy all images to staging for consistent splitting
        ensure_dir(staging_images)
        for img in list_images(raw_images, cfg["dataset"]["image_extensions"]):
            shutil_copy(img, staging_images / img.name)
        image_source = staging_images
    else:
        labels_source = raw_annotations
        image_source = raw_images

    images = list_images(image_source, cfg["dataset"]["image_extensions"])
    labeled = [p for p in images if (labels_source / f"{p.stem}.txt").exists()]

    if not labeled:
        raise SystemExit(
            "No labeled images found.\n"
            f"  Put drone images in: {raw_images}\n"
            f"  Put YOLO .txt labels in: {raw_annotations}\n"
            f"  OR place annotations.json (COCO format) in: {raw_annotations}"
        )

    print(f"Found {len(labeled)} labeled images (of {len(images)} total)")

    counts = split_dataset(
        labeled,
        labels_source,
        dataset_root,
        cfg["dataset"]["train_ratio"],
        cfg["dataset"]["val_ratio"],
        cfg["dataset"]["test_ratio"],
        seed=args.seed,
    )

    yaml_path = write_yolo_dataset_yaml(
        dataset_root,
        cfg["project"]["class_name"],
    )

    print("Dataset split:")
    for split, count in counts.items():
        print(f"  {split}: {count}")
    print(f"Dataset YAML: {yaml_path}")
    print("Ready for training: python scripts/train.py")


if __name__ == "__main__":
    main()
