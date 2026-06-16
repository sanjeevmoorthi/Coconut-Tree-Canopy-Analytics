"""Shared helpers for the coconut tree detection pipeline."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def configure_ultralytics() -> None:
    """Keep Ultralytics runtime settings inside the project workspace."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root()))


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_weights(cfg: dict | None = None) -> Path | None:
    """Return the newest trained model weights, if any exist."""
    root = project_root()
    cfg = cfg or load_config()
    candidates: list[Path] = []

    configured = cfg.get("paths", {}).get("weights")
    if configured:
        configured_path = (root / configured).resolve()
        if configured_path.exists():
            return configured_path

    candidates.extend(
        [
            root / "runs" / "detect" / "runs" / "coconut_quick" / "weights" / "best.pt",
            root / "runs" / "detect" / "runs" / cfg["project"]["name"] / "weights" / "best.pt",
            root / "runs" / cfg["project"]["name"] / "weights" / "best.pt",
        ]
    )

    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        existing.append(resolved)

    if not existing:
        return None

    return max(existing, key=lambda p: p.stat().st_mtime)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(directory: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    images = []
    for ext in exts:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def write_yolo_dataset_yaml(
    dataset_root: Path,
    class_name: str,
    output_path: Path | None = None,
) -> Path:
    """Write Ultralytics dataset YAML pointing at train/val/test splits."""
    output_path = output_path or dataset_root / "dataset.yaml"
    content = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    return output_path


def copy_split(
    image_paths: list[Path],
    labels_dir: Path,
    images_out: Path,
    labels_out: Path,
) -> None:
    ensure_dir(images_out)
    ensure_dir(labels_out)
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        shutil.copy2(img_path, images_out / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, labels_out / f"{img_path.stem}.txt")


def coco_bbox_to_yolo(
    bbox: list[float],
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, width, height] to YOLO normalized cx, cy, w, h."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    return cx, cy, w / img_width, h / img_height


def load_coco_annotations(coco_json: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """
    Load COCO-format JSON and return mapping:
    image_filename -> list of YOLO boxes (class 0 assumed).
    """
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco["images"]}
    filename_to_boxes: dict[str, list[tuple[float, float, float, float]]] = {}

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img = id_to_image[ann["image_id"]]
        box = coco_bbox_to_yolo(
            ann["bbox"],
            img["width"],
            img["height"],
        )
        filename_to_boxes.setdefault(img["file_name"], []).append(box)

    return filename_to_boxes
