"""
Train a YOLO model to detect coconut trees in drone imagery.

Usage:
  python scripts/train.py
  python scripts/train.py --epochs 150 --img-size 1280 --weights yolov8m.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train coconut tree detector")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--weights", default=None, help="Base model, e.g. yolov8n.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_yaml = Path(cfg["paths"]["dataset_root"]) / "dataset.yaml"

    if not dataset_yaml.exists():
        raise SystemExit(
            f"Dataset not found: {dataset_yaml}\n"
            "Run first: python scripts/prepare_dataset.py"
        )

    from ultralytics import YOLO

    weights = args.weights or cfg["model"]["base_weights"]
    model = YOLO(weights)

    epochs = args.epochs or cfg["model"]["epochs"]
    batch = args.batch_size or cfg["model"]["batch_size"]
    img_size = args.img_size or cfg["model"]["img_size"]
    device = args.device if args.device is not None else cfg["model"]["device"]

    # CPU fallback logic if GPU is not available
    import torch
    if device in [0, "0", "cuda"] or (isinstance(device, str) and device.startswith("cuda")):
        if not torch.cuda.is_available():
            print("WARNING: CUDA GPU requested but not available. Falling back to CPU training.")
            device = "cpu"

    print(f"Training {weights} on {dataset_yaml}")
    print(f"  epochs={epochs}, batch={batch}, img_size={img_size}, device={device}")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        patience=cfg["model"]["patience"],
        project=cfg["paths"]["output_dir"],
        name=cfg["project"]["name"],
        device=device,
        # Aerial / drone-friendly augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,      # rotation — useful for arbitrary drone heading
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,        # aerial images have no fixed "up"
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        save=True,
        plots=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print(f"Run inference: python scripts/predict_and_count.py --weights {best_weights} --source <image_or_folder>")


if __name__ == "__main__":
    main()
