"""
Detect and count coconut trees in drone images or orthomosaics.

Usage:
  python scripts/predict_and_count.py --weights runs/coconut_tree_detection/weights/best.pt --source data/raw/images
  python scripts/predict_and_count.py --weights best.pt --source field_map.tif --save
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import configure_ultralytics, ensure_dir, list_images, load_config


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Non-maximum suppression to merge duplicate detections across overlapping tiles."""
    if not detections:
        return []

    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.confidence for d in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_rest - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def generate_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) windows covering the image."""
    stride = max(1, int(tile_size * (1 - overlap)))
    tiles: list[tuple[int, int, int, int]] = []
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))

    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    for y in ys:
        for x in xs:
            tiles.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))

    return tiles


def run_yolo_on_image(
    model,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    img_size: int,
) -> list[Detection]:
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    detections: list[Detection] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(x1, y1, x2, y2, float(box.conf[0]))
            )

    return detections


def detect_in_large_image(
    model,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    img_size: int,
    tile_size: int,
    overlap: float,
) -> list[Detection]:
    h, w = image_bgr.shape[:2]

    if max(h, w) <= tile_size:
        return run_yolo_on_image(model, image_bgr, conf, iou, img_size)

    all_detections: list[Detection] = []
    for x1, y1, x2, y2 in generate_tiles(w, h, tile_size, overlap):
        tile = image_bgr[y1:y2, x1:x2]
        for det in run_yolo_on_image(model, tile, conf, iou, img_size):
            all_detections.append(
                Detection(
                    det.x1 + x1,
                    det.y1 + y1,
                    det.x2 + x1,
                    det.y2 + y1,
                    det.confidence,
                )
            )

    return nms(all_detections, iou_threshold=iou)


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        pt1 = (int(det.x1), int(det.y1))
        pt2 = (int(det.x2), int(det.y2))
        cv2.rectangle(out, pt1, pt2, (0, 200, 0), 2)
        cv2.circle(out, (int(det.center[0]), int(det.center[1])), 4, (0, 0, 255), -1)

    label = f"Coconut trees: {len(detections)}"
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return out


def process_source(
    model,
    source_path: Path,
    conf: float,
    iou: float,
    img_size: int,
    tile_size: int,
    overlap: float,
    save: bool,
    output_dir: Path,
) -> dict:
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    detections = detect_in_large_image(model, image, conf, iou, img_size, tile_size, overlap)
    count = len(detections)

    result = {
        "source": str(source_path),
        "tree_count": count,
        "output_image": "",
    }

    if save:
        ensure_dir(output_dir)
        vis = draw_detections(image, detections)
        out_path = output_dir / f"{source_path.stem}_count{count}{source_path.suffix}"
        cv2.imwrite(str(out_path), vis)
        result["output_image"] = str(out_path)

        # Save detection centers as CSV for GIS / further analysis
        csv_path = output_dir / f"{source_path.stem}_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tree_id", "center_x", "center_y", "confidence", "x1", "y1", "x2", "y2"],
            )
            writer.writeheader()
            for i, det in enumerate(detections, start=1):
                writer.writerow(
                    {
                        "tree_id": i,
                        "center_x": round(det.center[0], 2),
                        "center_y": round(det.center[1], 2),
                        "confidence": round(det.confidence, 4),
                        "x1": round(det.x1, 2),
                        "y1": round(det.y1, 2),
                        "x2": round(det.x2, 2),
                        "y2": round(det.y2, 2),
                    }
                )
        result["detections_csv"] = str(csv_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Count coconut trees in drone imagery")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--weights", default=None, help="Path to trained .pt weights")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--save", action="store_true", help="Save annotated images and CSV")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conf = args.conf if args.conf is not None else cfg["inference"]["confidence"]
    iou = args.iou if args.iou is not None else cfg["inference"]["iou"]
    img_size = args.img_size if args.img_size is not None else cfg["inference"]["img_size"]
    tile_size = args.tile_size if args.tile_size is not None else cfg["inference"]["tile_size"]
    overlap = args.overlap if args.overlap is not None else cfg["inference"]["tile_overlap"]

    configure_ultralytics()
    from ultralytics import YOLO

    from scripts.utils import find_best_weights

    weights = Path(args.weights) if args.weights else find_best_weights(cfg)
    if weights is None or not weights.exists():
        raise SystemExit(
            "No trained model found.\n"
            "Train first:\n"
            "  python scripts/prepare_dataset.py\n"
            "  python scripts/train.py --device cpu"
        )

    model = YOLO(str(weights))
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if source.is_dir():
        sources = list_images(source, cfg["dataset"]["image_extensions"])
    else:
        sources = [source]

    if not sources:
        raise SystemExit(f"No images found at: {source}")

    summary_path = output_dir / "count_summary.csv"
    rows: list[dict] = []

    print(f"Model: {weights}")
    print(f"conf={conf}, iou={iou}, img_size={img_size}, tile_size={tile_size}, overlap={overlap}")
    print(f"Processing {len(sources)} image(s)...")

    for img_path in sources:
        result = process_source(
            model, img_path, conf, iou, img_size, tile_size, overlap, args.save, output_dir
        )
        rows.append(result)
        print(f"  {img_path.name}: {result['tree_count']} trees")

    if args.save:
        ensure_dir(output_dir)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source", "tree_count", "output_image", "detections_csv"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved: {summary_path}")

    total = sum(r["tree_count"] for r in rows)
    print(f"\nTotal coconut trees detected: {total}")


if __name__ == "__main__":
    main()
