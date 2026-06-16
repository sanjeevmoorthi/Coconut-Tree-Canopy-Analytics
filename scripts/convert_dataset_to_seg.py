"""
Convert YOLO bounding box annotations into YOLO segmentation polygon annotations.
Supports both rectangular and circular (octagon) polygon approximations.

Usage:
    python scripts/convert_dataset_to_seg.py
    python scripts/convert_dataset_to_seg.py --type octagon
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import configure_ultralytics, ensure_dir, list_images, load_config, write_yolo_dataset_yaml, copy_split


def box_to_rectangle_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert box to a 4-point rectangle polygon."""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def box_to_octagon_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert box to an 8-point octagon polygon (approximation of a circle/crown)."""
    # Offset factor at 45 degrees: cos(45) * 0.5 = 0.7071 * 0.5 = 0.35355
    offset_w = 0.35355 * w
    offset_h = 0.35355 * h
    
    half_w = w / 2
    half_h = h / 2
    
    points = [
        cx, cy - half_h,                         # Top center
        cx + offset_w, cy - offset_h,            # Top right
        cx + half_w, cy,                         # Right center
        cx + offset_w, cy + offset_h,            # Bottom right
        cx, cy + half_h,                         # Bottom center
        cx - offset_w, cy + offset_h,            # Bottom left
        cx - half_w, cy,                         # Left center
        cx - offset_w, cy - offset_h             # Top left
    ]
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO boxes to segmentation polygons")
    parser.add_argument("--config", default="config/config_seg.yaml")
    parser.add_argument("--type", choices=["rectangle", "octagon"], default="octagon",
                        help="Polygon approximation style (octagon is recommended for tree crowns)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_images_dir = Path(cfg["paths"]["raw_images"])
    raw_boxes_dir = Path(cfg["paths"]["raw_annotations"])
    raw_seg_dir = Path(cfg["paths"]["seg_annotations"])
    dataset_root = Path(cfg["paths"]["dataset_root"])
    
    ensure_dir(raw_seg_dir)
    
    print(f"Converting box annotations to segmentation polygons ({args.type} approximation)...")
    converted_count = 0
    
    box_files = list(raw_boxes_dir.glob("*.txt"))
    for box_file in box_files:
        # Read boxes
        lines = box_file.read_text(encoding="utf-8").strip().split("\n")
        seg_lines = []
        for line in lines:
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = parts[0]
            cx, cy, w, h = map(float, parts[1:5])
            
            if args.type == "octagon":
                poly = box_to_octagon_polygon(cx, cy, w, h)
            else:
                poly = box_to_rectangle_polygon(cx, cy, w, h)
                
            poly_str = " ".join(f"{p:.6f}" for p in poly)
            seg_lines.append(f"{class_id} {poly_str}")
            
        # Write segmentation label
        seg_file = raw_seg_dir / box_file.name
        seg_file.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""), encoding="utf-8")
        converted_count += 1

    print(f"Successfully converted {converted_count} files.")
    
    # Split the dataset
    import random
    random.seed(args.seed)
    
    image_extensions = cfg["dataset"]["image_extensions"]
    images = list_images(raw_images_dir, image_extensions)
    labeled_images = [img for img in images if (raw_seg_dir / f"{img.stem}.txt").exists()]
    
    print(f"Found {len(labeled_images)} labeled images for splitting.")
    
    train_ratio = cfg["dataset"]["train_ratio"]
    val_ratio = cfg["dataset"]["val_ratio"]
    test_ratio = cfg["dataset"]["test_ratio"]
    
    # Shuffle
    shuffled = labeled_images.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :]
    }
    
    # Clean up old dirs
    import shutil
    for folder in ["images", "labels"]:
        path = dataset_root / folder
        if path.exists():
            print(f"Cleaning existing split directory: {path}")
            shutil.rmtree(path)
            
    # Write new splits
    counts = {}
    for split_name, paths in splits.items():
        images_out = dataset_root / "images" / split_name
        labels_out = dataset_root / "labels" / split_name
        
        ensure_dir(images_out)
        ensure_dir(labels_out)
        
        for img_path in paths:
            # Copy image
            shutil.copy2(img_path, images_out / img_path.name)
            # Copy segmentation label
            label_src = raw_seg_dir / f"{img_path.stem}.txt"
            shutil.copy2(label_src, labels_out / f"{img_path.stem}.txt")
            
        counts[split_name] = len(paths)
        
    # Write dataset yaml
    # YOLO segmentation uses the exact same YAML format as detection!
    yaml_path = write_yolo_dataset_yaml(dataset_root, cfg["project"]["class_name"])
    
    print("Dataset split complete:")
    for split, count in counts.items():
        print(f"  {split}: {count}")
    print(f"Dataset YAML: {yaml_path}")
    print("Ready for segmentation training!")


if __name__ == "__main__":
    main()
