"""
Count coconut trees in a plantation / farm drone photo.

This is the main entry point for the project objective:
  your farm image  ->  tree count + annotated output

Usage:
  python scripts/count_farm.py path/to/farm.jpg
  python scripts/count_farm.py path/to/orthomosaic.tif --conf 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.predict_and_count import process_source
from scripts.utils import configure_ultralytics, find_best_weights, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count coconut trees in a farm or plantation drone image",
    )
    parser.add_argument("image", help="Path to your coconut farm photo (.jpg, .png, .tif)")
    parser.add_argument("--weights", default=None, help="Trained model .pt (auto-detected if omitted)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None, help="YOLO inference image size")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for large maps (pixels)")
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else find_best_weights(cfg)
    if weights is None or not weights.exists():
        raise SystemExit(
            "No trained model found.\n"
            "Train first:\n"
            "  python scripts/prepare_dataset.py\n"
            "  python scripts/train.py --device cpu"
        )

    conf = args.conf if args.conf is not None else cfg["inference"]["confidence"]
    iou = args.iou if args.iou is not None else cfg["inference"]["iou"]
    img_size = args.img_size if args.img_size is not None else cfg["inference"]["img_size"]
    tile_size = args.tile_size if args.tile_size is not None else cfg["inference"]["tile_size"]
    overlap = args.overlap if args.overlap is not None else cfg["inference"]["tile_overlap"]
    output_dir = Path(args.output_dir)

    print(f"Image : {image_path}")
    print(f"Model : {weights}")
    print("Running detection...")

    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = process_source(
        model,
        image_path,
        conf,
        iou,
        img_size,
        tile_size,
        overlap,
        save=True,
        output_dir=output_dir,
    )

    count = result["tree_count"]
    print()
    print("=" * 48)
    print(f"  COCONUT TREE COUNT: {count}")
    print("=" * 48)
    print()
    if result.get("output_image"):
        print(f"Annotated image : {result['output_image']}")
    if result.get("detections_csv"):
        print(f"Tree locations  : {result['detections_csv']}")


if __name__ == "__main__":
    main()
