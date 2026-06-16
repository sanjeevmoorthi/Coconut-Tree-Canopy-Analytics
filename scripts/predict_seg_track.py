"""
Run YOLOv11-Segmentation and ByteTrack to detect, track, segment, and count unique coconut trees.
Includes a baseline health classification module (Healthy, Deficient, Diseased) as a hook for future analytics.

Usage:
    python scripts/predict_seg_track.py --source data/raw/images/000000.jpg --save
    python scripts/predict_seg_track.py --weights runs/segment/train/weights/best.pt --source drone_flight.mp4 --save
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import configure_ultralytics, ensure_dir, list_images, load_config


def classify_tree_health(crop_bgr: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """
    Classify tree health based on leaf color analysis in the HSV color space.
    This acts as a functional baseline and hook for future multispectral/NDVI models.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "Healthy", (0, 200, 0) # Green
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
    # Green leaves
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Yellow/brown leaves (deficiency or disease indicator)
    lower_yellow = np.array([10, 40, 40])
    upper_yellow = np.array([34, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    total_pixels = hsv.shape[0] * hsv.shape[1] + 1e-6
    green_ratio = np.sum(green_mask > 0) / total_pixels
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
    
    if yellow_ratio > 0.12:
        if yellow_ratio > 0.28:
            return "Diseased", (0, 0, 220) # Red
        return "Deficient", (0, 140, 255) # Orange
    return "Healthy", (0, 200, 0) # Green


def process_video_or_stream(
    model,
    source_path: str,
    conf: float,
    iou: float,
    img_size: int,
    save: bool,
    output_dir: Path,
) -> None:
    # Open video capture
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    out_writer = None
    if save:
        ensure_dir(output_dir)
        source_name = Path(source_path).stem
        out_path = output_dir / f"{source_name}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"Saving tracked video to: {out_path}")

    counted_ids: set[int] = set()
    health_counts = {"Healthy": 0, "Deficient": 0, "Diseased": 0}
    
    frame_idx = 0
    t0 = time.time()
    
    # We use Ultralytics track generator for processing streams frame-by-frame
    # tracker="bytetrack.yaml" is standard
    print("Initializing tracker...")
    results = model.track(
        source=source_path,
        conf=conf,
        iou=iou,
        imgsz=img_size,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False
    )
    
    for r in results:
        frame_idx += 1
        frame = r.orig_img.copy()
        h, w = frame.shape[:2]
        
        # Transparent overlay mask for segmentation
        mask_overlay = frame.copy()
        
        if r.boxes is not None and r.masks is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
            confs = r.boxes.conf.cpu().numpy()
            masks_xy = r.masks.xy
            
            for box, track_id, conf_val, mask_poly in zip(boxes, track_ids, confs, masks_xy):
                x1, y1, x2, y2 = map(int, box)
                
                # Health Classification
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                health_status, color = classify_tree_health(crop)
                
                # Update Unique Counter
                if track_id is not None and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    health_counts[health_status] += 1
                
                # Draw Mask
                if len(mask_poly) > 0:
                    pts = np.array(mask_poly, dtype=np.int32)
                    cv2.fillPoly(mask_overlay, [pts], color)
                    cv2.polylines(frame, [pts], True, color, 1)
                
                # Draw Box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_str = f"ID: {track_id}" if track_id is not None else "DET"
                label = f"{id_str} | {health_status} ({conf_val:.2f})"
                
                # Text background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend segmentation mask overlay (alpha blending)
        cv2.addWeighted(mask_overlay, 0.35, frame, 0.65, 0, frame)
        
        # Draw HUD Dashboard
        total_unique = len(counted_ids)
        # Dashboard Background Card
        cv2.rectangle(frame, (10, 10), (320, 150), (17, 24, 39), -1)
        cv2.rectangle(frame, (10, 10), (320, 150), (55, 65, 81), 2)
        
        cv2.putText(frame, "🌴 COCONUT TREE COUNT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
        cv2.putText(frame, f"Unique Trees: {total_unique}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Healthy: {health_counts['Healthy']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Deficient: {health_counts['Deficient']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(frame, f"Diseased: {health_counts['Diseased']}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if out_writer:
            out_writer.write(frame)
            
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames. Total unique trees detected: {total_unique}")

    cap.release()
    if out_writer:
        out_writer.release()
        
    elapsed = time.time() - t0
    print(f"\nProcessing Complete.")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
    print(f"Final Count of Unique Coconut Trees: {len(counted_ids)}")
    print(f"Health Breakdown: {health_counts}")


def process_image(
    model,
    source_path: Path,
    conf: float,
    iou: float,
    img_size: int,
    save: bool,
    output_dir: Path,
) -> dict:
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    # For single images, track mode behaves like standard prediction but returns IDs if consecutive
    results = model.track(image, conf=conf, iou=iou, imgsz=img_size, tracker="bytetrack.yaml", persist=True, verbose=False)
    
    total_trees = 0
    health_counts = {"Healthy": 0, "Deficient": 0, "Diseased": 0}
    detections = []
    
    mask_overlay = image.copy()
    h, w = image.shape[:2]
    
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
            confs = r.boxes.conf.cpu().numpy()
            
            masks_xy = r.masks.xy if r.masks is not None else [[]] * len(boxes)
            
            for idx, (box, track_id, conf_val, mask_poly) in enumerate(zip(boxes, track_ids, confs, masks_xy)):
                x1, y1, x2, y2 = map(int, box)
                total_trees += 1
                
                # Health Classification
                crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                health_status, color = classify_tree_health(crop)
                health_counts[health_status] += 1
                
                # Draw Mask
                if len(mask_poly) > 0:
                    pts = np.array(mask_poly, dtype=np.int32)
                    cv2.fillPoly(mask_overlay, [pts], color)
                    cv2.polylines(image, [pts], True, color, 1)
                
                # Draw Box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                id_str = f"ID: {track_id}" if track_id is not None else f"Tree {idx+1}"
                label = f"{id_str} | {health_status}"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(image, (x1, y1 - 15), (x1 + tw, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                detections.append({
                    "id": track_id if track_id is not None else idx + 1,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": float(conf_val),
                    "health": health_status
                })

    cv2.addWeighted(mask_overlay, 0.3, image, 0.7, 0, image)
    
    # Render dashboard card
    cv2.rectangle(image, (10, 10), (320, 140), (17, 24, 39), -1)
    cv2.rectangle(image, (10, 10), (320, 140), (55, 65, 81), 2)
    cv2.putText(image, "🌴 COCONUT TREE COUNT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
    cv2.putText(image, f"Total Trees: {total_trees}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Healthy: {health_counts['Healthy']} | Deficient: {health_counts['Deficient']} | Diseased: {health_counts['Diseased']}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    result = {
        "source": str(source_path),
        "tree_count": total_trees,
        "health_breakdown": health_counts,
        "output_image": "",
    }

    if save:
        ensure_dir(output_dir)
        out_path = output_dir / f"{source_path.stem}_seg_count{total_trees}{source_path.suffix}"
        cv2.imwrite(str(out_path), image)
        result["output_image"] = str(out_path)

        csv_path = output_dir / f"{source_path.stem}_seg_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tree_id", "x1", "y1", "x2", "y2", "confidence", "health"],
            )
            writer.writeheader()
            for d in detections:
                writer.writerow({
                    "tree_id": d["id"],
                    "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                    "confidence": round(d["confidence"], 4),
                    "health": d["health"]
                })
        result["detections_csv"] = str(csv_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Track and segment coconut trees in video or image")
    parser.add_argument("--config", default="config/config_seg.yaml")
    parser.add_argument("--weights", default=None, help="Trained yolo11-seg weights")
    parser.add_argument("--source", required=True, help="Path to video or image file")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="Save tracked video/image output")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conf = args.conf if args.conf is not None else cfg["inference"]["confidence"]
    iou = args.iou if args.iou is not None else cfg["inference"]["iou"]
    img_size = args.img_size if args.img_size is not None else cfg["inference"]["img_size"]
    output_dir = Path(args.output_dir)

    # Load YOLOv11-Segmentation model
    configure_ultralytics()
    from ultralytics import YOLO
    
    # Fallback to base weights if trained weights do not exist yet
    weights = Path(args.weights) if args.weights else Path(cfg["paths"]["weights"])
    if not weights.exists():
        print(f"Trained weights not found at {weights}. Using base YOLO11-Seg weights instead.")
        weights = Path(cfg["model"]["base_weights"])

    model = YOLO(str(weights))
    source = Path(args.source)

    print(f"Model: {weights}")
    print(f"Source: {source}")
    print(f"conf={conf}, iou={iou}, img_size={img_size}")

    # Check if video
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    if source.suffix.lower() in video_exts:
        process_video_or_stream(model, str(source), conf, iou, img_size, args.save, output_dir)
    else:
        res = process_image(model, source, conf, iou, img_size, args.save, output_dir)
        print(f"\nProcessing Complete.")
        print(f"Total coconut trees detected: {res['tree_count']}")
        print(f"Health Breakdown: {res['health_breakdown']}")
        if args.save:
            print(f"Saved annotated image: {res['output_image']}")
            print(f"Saved detection CSV  : {res['detections_csv']}")


if __name__ == "__main__":
    main()
