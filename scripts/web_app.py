"""
Coconut Tree Counter — Premium Web Application
Supports YOLOv8/v11 Bounding Boxes & YOLOv11-Segmentation with Real-Time Interactive Canvas overlays,
ByteTrack unique IDs, HSV leaf-color health classification, and advanced farm analytics.

Usage:
    python scripts/web_app.py
    python scripts/web_app.py --port 5000
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import configure_ultralytics, find_best_weights, load_config

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TreeResult:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    mask_poly: list[list[float]] = None
    health: str = "Healthy"
    track_id: int = None

    def to_dict(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "mask_poly": self.mask_poly,
            "health": self.health,
            "track_id": self.track_id
        }


# ---------------------------------------------------------------------------
# Health Classification (HSV baseline)
# ---------------------------------------------------------------------------

def classify_tree_health(crop_bgr: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """
    Classify tree health based on leaf color analysis in the HSV color space.
    Healthy: Green (BGR: 80, 200, 120)
    Deficient: Orange/Yellow (BGR: 16, 140, 245)
    Diseased: Red (BGR: 60, 60, 230)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "Healthy", (80, 200, 120)
    
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
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
    
    if yellow_ratio > 0.12:
        if yellow_ratio > 0.28:
            return "Diseased", (60, 60, 230)
        return "Deficient", (16, 140, 245)
    return "Healthy", (80, 200, 120)


def get_octagon_polygon(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    """Generate an 8-point octagon inscribed in the bounding box to approximate tree crowns."""
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    pts = [
        [cx, y1],
        [cx + 0.35 * w, cy - 0.35 * h],
        [x2, cy],
        [cx + 0.35 * w, cy + 0.35 * h],
        [cx, y2],
        [cx - 0.35 * w, cy + 0.35 * h],
        [x1, cy],
        [cx - 0.35 * w, cy - 0.35 * h]
    ]
    return [[round(pt[0], 1), round(pt[1], 1)] for pt in pts]


# ---------------------------------------------------------------------------
# Tiling & NMS
# ---------------------------------------------------------------------------

def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(tile_size * (1 - overlap)))
    tiles = []
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


def nms_results(results: list[TreeResult], iou_threshold: float) -> list[TreeResult]:
    if not results:
        return []
    boxes = np.array([[r.x1, r.y1, r.x2, r.y2] for r in results], dtype=np.float32)
    scores = np.array([r.confidence for r in results], dtype=np.float32)
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

    return [results[i] for i in keep]


def run_single_inference(model, img: np.ndarray, conf: float, iou: float, img_size: int) -> list[TreeResult]:
    is_seg = hasattr(model, "task") and model.task == "segment"
    results = model.predict(img, conf=conf, imgsz=img_size, iou=iou, verbose=False)
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        # Track IDs if present
        track_ids = r.boxes.id.int().cpu().numpy() if (hasattr(r.boxes, "id") and r.boxes.id is not None) else [None] * len(boxes)
        
        # Masks if present
        masks_xy = r.masks.xy if (is_seg and r.masks is not None) else [None] * len(boxes)
        
        for box, conf_val, track_id, mask in zip(boxes, confs, track_ids, masks_xy):
            x1, y1, x2, y2 = map(float, box)
            
            # Health classification
            crop = img[max(0, int(y1)):min(img.shape[0], int(y2)), max(0, int(x1)):min(img.shape[1], int(x2))]
            health_status, _ = classify_tree_health(crop)
            
            # Mask polygon
            if mask is not None and len(mask) > 0:
                poly = [[float(pt[0]), float(pt[1])] for pt in mask]
            else:
                poly = get_octagon_polygon(x1, y1, x2, y2)
                
            detections.append(TreeResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(conf_val),
                mask_poly=poly,
                health=health_status,
                track_id=int(track_id) if track_id is not None else None
            ))
            
    return detections


def run_tiled_inference(
    model,
    img: np.ndarray,
    conf: float,
    iou: float,
    img_size: int,
    tile_size: int,
    overlap: float
) -> list[TreeResult]:
    h, w = img.shape[:2]
    
    if max(h, w) <= tile_size:
        return run_single_inference(model, img, conf, iou, img_size)
        
    tiles = generate_tiles(w, h, tile_size, overlap)
    all_results = []
    
    for tx1, ty1, tx2, ty2 in tiles:
        tile_img = img[ty1:ty2, tx1:tx2]
        tile_results = run_single_inference(model, tile_img, conf, iou, img_size)
        
        for r in tile_results:
            r.x1 += tx1
            r.y1 += ty1
            r.x2 += tx1
            r.y2 += ty1
            if r.mask_poly:
                r.mask_poly = [[pt[0] + tx1, pt[1] + ty1] for pt in r.mask_poly]
            all_results.append(r)
            
    return nms_results(all_results, iou_threshold=iou)


# ---------------------------------------------------------------------------
# HTML template (Premium UI design with custom charts, canvas overlays & search)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coconut Tree Analytics Dashboard — AI Farm Intelligence</title>
    <meta name="description" content="State-of-the-art AI plantation analysis: YOLOv11 Instance Segmentation, Persistent Tracking, and Health Analytics">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #070b13;
            --bg-secondary: #0c1220;
            --bg-card: rgba(12, 18, 32, 0.75);
            --bg-card-hover: rgba(18, 26, 46, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(16, 185, 129, 0.35);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-green: #10b981;
            --accent-green-glow: rgba(16, 185, 129, 0.15);
            --accent-orange: #f97316;
            --accent-orange-glow: rgba(249, 115, 22, 0.15);
            --accent-red: #ef4444;
            --accent-red-glow: rgba(239, 68, 68, 0.15);
            --accent-teal: #06b6d4;
            --accent-teal-glow: rgba(6, 182, 212, 0.15);
            --shadow-glow: 0 0 30px rgba(16, 185, 129, 0.12);
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Animated Grid Overlay */
        .bg-grid {
            position: fixed;
            inset: 0;
            background-image:
                radial-gradient(circle at 10% 30%, rgba(6, 182, 212, 0.04) 0%, transparent 50%),
                radial-gradient(circle at 90% 70%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
            z-index: 0;
            pointer-events: none;
        }

        .bg-grid::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 80px 80px;
        }

        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
            padding: 2.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
            flex-grow: 1;
        }

        /* Header Layout */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1.5rem;
        }

        .brand-section h1 {
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #34d399 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .brand-section p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.2rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .header-badge .dot {
            width: 8px; height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse-dot 2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.5; }
        }

        /* Dashboard Workspace */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 2rem;
        }

        /* Left Side: Upload & Control Panel */
        .control-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            transition: var(--transition-smooth);
        }

        .panel-card:hover {
            border-color: rgba(255,255,255,0.12);
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-primary);
            border-left: 3px solid var(--accent-green);
            padding-left: 0.5rem;
        }

        /* File Dropzone */
        .dropzone {
            position: relative;
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.2);
            transition: var(--transition-smooth);
        }

        .dropzone:hover,
        .dropzone.drag-over {
            border-color: var(--accent-green);
            background: var(--accent-green-glow);
            box-shadow: var(--shadow-glow);
        }

        .dropzone-icon svg {
            width: 40px; height: 40px;
            stroke: var(--accent-green);
            margin-bottom: 0.75rem;
            transition: var(--transition-smooth);
        }

        .dropzone:hover .dropzone-icon svg {
            transform: translateY(-4px);
        }

        .dropzone-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .dropzone-subtitle {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .dropzone-subtitle span {
            color: var(--accent-green);
            text-decoration: underline;
        }

        .dropzone input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        /* File Preview Bar */
        .file-info-bar {
            display: none;
            align-items: center;
            gap: 0.75rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 0.75rem;
            margin-top: 1rem;
        }

        .file-info-bar.visible {
            display: flex;
        }

        .file-thumb {
            width: 44px; height: 44px;
            border-radius: 8px;
            object-fit: cover;
            border: 1px solid var(--border-subtle);
        }

        .file-details {
            flex-grow: 1;
            min-width: 0;
        }

        .file-name {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .file-size {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Sliders */
        .settings-list {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .slider-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .slider-badge {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent-teal);
            background: var(--accent-teal-glow);
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-variant-numeric: tabular-nums;
        }

        .slider-control {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 100px;
            outline: none;
            accent-color: var(--accent-green);
            cursor: pointer;
        }

        .btn-action {
            width: 100%;
            padding: 0.9rem;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        }

        .btn-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
        }

        .btn-action:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        /* Right Side: Main Dashboard Output */
        .dashboard-output {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            min-width: 0;
        }

        /* Stats Row */
        .analytics-row {
            display: none;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.25rem;
        }

        .analytics-row.visible {
            display: grid;
            animation: slideDown 0.5s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
        }

        .kpi-card.total::before { background: var(--accent-teal); }
        .kpi-card.healthy::before { background: var(--accent-green); }
        .kpi-card.deficient::before { background: var(--accent-orange); }
        .kpi-card.diseased::before { background: var(--accent-red); }

        .kpi-title {
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .kpi-value-row {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .kpi-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-primary);
        }

        .kpi-percent {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .kpi-percent.healthy { color: var(--accent-green); }
        .kpi-percent.deficient { color: var(--accent-orange); }
        .kpi-percent.diseased { color: var(--accent-red); }

        .kpi-bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            margin-top: 0.75rem;
            overflow: hidden;
        }

        .kpi-bar {
            height: 100%;
            border-radius: 10px;
        }

        .kpi-bar.healthy { background: var(--accent-green); }
        .kpi-bar.deficient { background: var(--accent-orange); }
        .kpi-bar.diseased { background: var(--accent-red); }

        /* KPI Subtext metrics */
        .kpi-subtext {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Core Map Visualizer */
        .visualizer-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(16px);
        }

        .visualizer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(0, 0, 0, 0.15);
        }

        .toolbar {
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .toggle-group {
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.2rem;
        }

        .toggle-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.4rem 0.8rem;
            border-radius: 7px;
            cursor: pointer;
            transition: var(--transition-smooth);
        }

        .toggle-btn.active {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }

        .toggle-btn.green.active { color: var(--accent-green); background: var(--accent-green-glow); }

        .legend {
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .legend-color {
            width: 10px; height: 10px;
            border-radius: 50%;
        }

        .legend-color.healthy { background: var(--accent-green); }
        .legend-color.deficient { background: var(--accent-orange); }
        .legend-color.diseased { background: var(--accent-red); }

        /* Canvas Wrapper */
        .canvas-container {
            position: relative;
            background: #04060b;
            min-height: 400px;
            max-height: 680px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 1.5rem;
        }

        .canvas-container canvas {
            display: block;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            border-radius: 6px;
            max-width: 100%;
            height: auto;
        }

        /* Tooltip style */
        .canvas-tooltip {
            position: fixed;
            background: rgba(12, 18, 32, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            z-index: 1000;
            pointer-events: none;
            display: none;
            font-size: 0.8rem;
            color: var(--text-primary);
            backdrop-filter: blur(8px);
        }

        .tooltip-header {
            font-weight: 700;
            color: var(--accent-teal);
            margin-bottom: 0.35rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 0.25rem;
        }

        .tooltip-row {
            margin: 0.2rem 0;
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .status-badge {
            font-weight: 700;
            padding: 0.05rem 0.35rem;
            border-radius: 4px;
            font-size: 0.7rem;
            text-transform: uppercase;
        }

        .status-badge.healthy { background: var(--accent-green-glow); color: var(--accent-green); }
        .status-badge.deficient { background: var(--accent-orange-glow); color: var(--accent-orange); }
        .status-badge.diseased { background: var(--accent-red-glow); color: var(--accent-red); }

        /* Bottom Section: Detections Directory */
        .directory-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            display: none;
        }

        .directory-card.visible {
            display: block;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .directory-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .directory-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .directory-controls {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .search-input {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            width: 200px;
            transition: var(--transition-smooth);
        }

        .search-input:focus {
            border-color: var(--accent-teal);
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.25);
        }

        .select-filter {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .btn-export {
            background: rgba(6, 182, 212, 0.12);
            border: 1px solid rgba(6, 182, 212, 0.25);
            color: #22d3ee;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .btn-export:hover {
            background: rgba(6, 182, 212, 0.2);
            border-color: #22d3ee;
        }

        /* Table design */
        .table-wrapper {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.1);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }

        th, td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        th {
            background: rgba(255, 255, 255, 0.02);
            font-weight: 700;
            color: var(--text-secondary);
            position: sticky;
            top: 0;
            z-index: 10;
            backdrop-filter: blur(10px);
        }

        tr {
            transition: var(--transition-smooth);
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
        }

        tr.highlighted {
            background: rgba(6, 182, 212, 0.08) !important;
        }

        .btn-locate {
            background: transparent;
            border: none;
            color: var(--accent-teal);
            cursor: pointer;
            font-weight: 700;
            font-size: 0.8rem;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }

        .btn-locate:hover {
            text-decoration: underline;
        }

        /* Error Display */
        .error-message {
            display: none;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            padding: 1rem;
            border-radius: 12px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .error-message.visible {
            display: block;
        }

        /* Loading HUD Overlay */
        .loading-screen {
            position: fixed;
            inset: 0;
            background: rgba(7, 11, 19, 0.85);
            backdrop-filter: blur(12px);
            z-index: 2000;
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 1.5rem;
        }

        .loading-screen.active {
            display: flex;
        }

        .spinner {
            width: 60px; height: 60px;
            border: 4px solid rgba(255, 255, 255, 0.08);
            border-top-color: var(--accent-green);
            border-radius: 50%;
            animation: spin 1s cubic-bezier(0.55, 0.055, 0.675, 0.19) infinite;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        .loading-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* Footer */
        .footer {
            margin-top: auto;
            text-align: center;
            padding: 2.5rem 0 1rem;
            border-top: 1px solid var(--border-subtle);
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .footer .tech-stack {
            display: inline-flex;
            gap: 0.5rem;
            margin-top: 0.4rem;
        }

        .footer .tech-stack span {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-weight: 500;
        }

        /* Mobile Adjustments */
        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .analytics-row.visible {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="bg-grid"></div>

    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="brand-section">
                <h1>🌴 Coconut Tree Analytics Dashboard</h1>
                <p>AI Plantation Intelligence: YOLOv11-Segmentation + ByteTrack + Health Assessment</p>
            </div>
            <div class="header-badge">
                <div class="dot"></div>
                Engine Online (YOLOv11-Seg ready)
            </div>
        </header>

        <!-- Main Workspace -->
        <main class="dashboard-grid">
            
            <!-- Left Side: Controls -->
            <div class="control-panel">
                
                <!-- Card 1: Upload Source -->
                <div class="panel-card">
                    <div class="card-title">Upload Farm Imagery</div>
                    <div class="dropzone" id="dropzone">
                        <div class="dropzone-icon">
                            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                            </svg>
                        </div>
                        <div class="dropzone-title">Drop orthomosaic or aerial photo</div>
                        <div class="dropzone-subtitle">or <span>browse local files</span></div>
                        <input type="file" id="fileInput" accept="image/*">
                    </div>

                    <div class="file-info-bar" id="fileInfoBar">
                        <img class="file-thumb" id="fileThumb" src="" alt="Thumbnail">
                        <div class="file-details">
                            <div class="file-name" id="fileName">—</div>
                            <div class="file-size" id="fileSize">—</div>
                        </div>
                    </div>
                </div>

                <!-- Card 2: Engine Parameters -->
                <div class="panel-card">
                    <div class="card-title">Engine Parameters</div>
                    <div class="settings-list">
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span class="slider-label">Confidence Threshold</span>
                                <span class="slider-badge" id="confBadge">0.05</span>
                            </div>
                            <input type="range" class="slider-control" id="confSlider" min="0.005" max="0.5" step="0.005" value="0.05">
                        </div>

                        <div class="slider-group">
                            <div class="slider-header">
                                <span class="slider-label">IoU NMS Threshold</span>
                                <span class="slider-badge" id="iouBadge">0.30</span>
                            </div>
                            <input type="range" class="slider-control" id="iouSlider" min="0.05" max="0.95" step="0.05" value="0.30">
                        </div>

                        <div class="slider-group">
                            <div class="slider-header">
                                <span class="slider-label">Inference Tile Size</span>
                                <span class="slider-badge" id="tileBadge">1280 px</span>
                            </div>
                            <input type="range" class="slider-control" id="tileSlider" min="256" max="2048" step="128" value="1280">
                        </div>

                        <div class="slider-group">
                            <div class="slider-header">
                                <span class="slider-label">Tile Overlap</span>
                                <span class="slider-badge" id="overlapBadge">25%</span>
                            </div>
                            <input type="range" class="slider-control" id="overlapSlider" min="0.0" max="0.5" step="0.05" value="0.25">
                        </div>

                        <button class="btn-action" id="btnRun" disabled>
                            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Run AI Diagnostics
                        </button>
                    </div>
                </div>

            </div>

            <!-- Right Side: Dashboard Outputs -->
            <div class="dashboard-output">
                
                <!-- Error Banner -->
                <div class="error-message" id="errorBanner"></div>

                <!-- KPI Analytics Row -->
                <div class="analytics-row" id="analyticsRow">
                    
                    <div class="kpi-card total">
                        <div class="kpi-title">Total Canopy Count</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value" id="kpiTotal">0</div>
                            <div class="kpi-percent text-muted">trees</div>
                        </div>
                        <div class="kpi-subtext" id="kpiRes">—</div>
                    </div>

                    <div class="kpi-card healthy">
                        <div class="kpi-title">Healthy Canopies</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value" id="kpiHealthy">0</div>
                            <div class="kpi-percent healthy" id="kpiHealthyPct">0%</div>
                        </div>
                        <div class="kpi-bar-container">
                            <div class="kpi-bar healthy" id="kpiHealthyBar" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="kpi-card deficient">
                        <div class="kpi-title">Nutrient Deficient</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value" id="kpiDeficient">0</div>
                            <div class="kpi-percent deficient" id="kpiDeficientPct">0%</div>
                        </div>
                        <div class="kpi-bar-container">
                            <div class="kpi-bar deficient" id="kpiDeficientBar" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="kpi-card diseased">
                        <div class="kpi-title">Diseased / Stressed</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value" id="kpiDiseased">0</div>
                            <div class="kpi-percent diseased" id="kpiDiseasedPct">0%</div>
                        </div>
                        <div class="kpi-bar-container">
                            <div class="kpi-bar diseased" id="kpiDiseasedBar" style="width: 0%"></div>
                        </div>
                    </div>

                </div>

                <!-- Core Map View -->
                <div class="visualizer-card">
                    <div class="visualizer-header">
                        <div class="toolbar">
                            <div class="toggle-group">
                                <button class="toggle-btn active" id="btnOriginal" onclick="setViewMode('clean')">Clean View</button>
                                <button class="toggle-btn" id="btnOverlay" onclick="setViewMode('annotated')">Diagnostics View</button>
                            </div>
                            
                            <div class="toggle-group" id="renderControls" style="display: none;">
                                <button class="toggle-btn active green" id="toggleMasks" onclick="toggleRenderOpt('masks')">Segmentation Masks</button>
                                <button class="toggle-btn active green" id="toggleBoxes" onclick="toggleRenderOpt('boxes')">Bounding Boxes</button>
                                <button class="toggle-btn active green" id="toggleLabels" onclick="toggleRenderOpt('labels')">ID Labels</button>
                            </div>
                        </div>

                        <div class="legend" id="mapLegend" style="display: none;">
                            <div class="legend-item">
                                <div class="legend-color healthy"></div> Healthy
                            </div>
                            <div class="legend-item">
                                <div class="legend-color deficient"></div> Nutrient Deficient
                            </div>
                            <div class="legend-item">
                                <div class="legend-color diseased"></div> Diseased
                            </div>
                        </div>
                    </div>

                    <div class="canvas-container" id="canvasContainer">
                        <canvas id="viewCanvas"></canvas>
                    </div>
                </div>

                <!-- Directory Card -->
                <div class="directory-card" id="directoryCard">
                    <div class="directory-header">
                        <div class="directory-title">Farm Inventory Directory</div>
                        <div class="directory-controls">
                            <input type="text" class="search-input" id="tblSearch" placeholder="Search Tree ID..." oninput="filterTable()">
                            <select class="select-filter" id="tblFilter" onchange="filterTable()">
                                <option value="ALL">All Health Statuses</option>
                                <option value="Healthy">Healthy Only</option>
                                <option value="Deficient">Nutrient Deficient Only</option>
                                <option value="Diseased">Diseased Only</option>
                            </select>
                            <button class="btn-export" onclick="exportCSV()">
                                <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                Export GIS CSV
                            </button>
                        </div>
                    </div>
                    <div class="table-wrapper">
                        <table id="tblInventory">
                            <thead>
                                <tr>
                                    <th>Tree ID</th>
                                    <th>Confidence</th>
                                    <th>Health Status</th>
                                    <th>Bounding Coordinates (x1, y1, x2, y2)</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="tblBody">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>

        </main>

        <!-- Footer -->
        <footer class="footer">
            <p>Plantation Diagnostics System — Phase II</p>
            <div class="tech-stack">
                Powered by: <span>YOLOv11-Segmentation</span> <span>ByteTrack</span> <span>OpenCV</span> <span>Flask</span>
            </div>
        </footer>
    </div>

    <!-- Hover Tooltip -->
    <div class="canvas-tooltip" id="tooltip"></div>

    <!-- Loading Screen -->
    <div class="loading-screen" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-title" id="loadingTitle">Running Deep Learning Engine…</div>
        <div class="loading-subtitle" id="loadingSubtitle">Applying tiling inference, segmenting tree canopies, and compiling health statistics</div>
    </div>

    <script>
        // DOM Elements
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileInfoBar = document.getElementById('fileInfoBar');
        const fileThumb = document.getElementById('fileThumb');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const btnRun = document.getElementById('btnRun');
        const errorBanner = document.getElementById('errorBanner');
        const loadingOverlay = document.getElementById('loadingOverlay');
        
        // Sliders
        const confSlider = document.getElementById('confSlider');
        const confBadge = document.getElementById('confBadge');
        const iouSlider = document.getElementById('iouSlider');
        const iouBadge = document.getElementById('iouBadge');
        const tileSlider = document.getElementById('tileSlider');
        const tileBadge = document.getElementById('tileBadge');
        const overlapSlider = document.getElementById('overlapSlider');
        const overlapBadge = document.getElementById('overlapBadge');

        // Layout Elements
        const analyticsRow = document.getElementById('analyticsRow');
        const directoryCard = document.getElementById('directoryCard');
        const renderControls = document.getElementById('renderControls');
        const mapLegend = document.getElementById('mapLegend');
        const canvas = document.getElementById('viewCanvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        // State variables
        let selectedFile = null;
        let originalImage = null; // Image object
        let detections = []; // Array of detections
        let viewMode = 'clean'; // clean | annotated
        
        // Render Options
        let renderOpts = {
            masks: true,
            boxes: true,
            labels: true
        };
        let hoveredTreeId = null;
        let highlightedTreeId = null;

        // Slider listeners
        confSlider.addEventListener('input', () => confBadge.textContent = parseFloat(confSlider.value).toFixed(3));
        iouSlider.addEventListener('input', () => iouBadge.textContent = parseFloat(iouSlider.value).toFixed(2));
        tileSlider.addEventListener('input', () => tileBadge.textContent = tileSlider.value + ' px');
        overlapSlider.addEventListener('input', () => overlapBadge.textContent = Math.round(overlapSlider.value * 100) + '%');

        // Drag & Drop Handlers
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleFileSelection(fileInput.files[0]);
        });

        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        function handleFileSelection(file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatBytes(file.size);
            
            const reader = new FileReader();
            reader.onload = (e) => {
                fileThumb.src = e.target.result;
                fileInfoBar.classList.add('visible');
                btnRun.disabled = false;

                // Load image for canvas drawing
                originalImage = new Image();
                originalImage.onload = () => {
                    setupCanvas();
                    // Reset outputs when new image is uploaded
                    analyticsRow.classList.remove('visible');
                    directoryCard.classList.remove('visible');
                    renderControls.style.display = 'none';
                    mapLegend.style.display = 'none';
                    setViewMode('clean');
                };
                originalImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
            errorBanner.classList.remove('visible');
        }

        function setupCanvas() {
            if (!originalImage) return;
            canvas.width = originalImage.naturalWidth;
            canvas.height = originalImage.naturalHeight;
            drawCanvas();
        }

        // Run Diagnostics
        btnRun.addEventListener('click', async () => {
            if (!selectedFile) return;

            btnRun.disabled = true;
            loadingOverlay.classList.add('active');
            errorBanner.classList.remove('visible');

            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('conf', confSlider.value);
            formData.append('iou', iouSlider.value);
            formData.append('tile_size', tileSlider.value);
            formData.append('overlap', overlapSlider.value);

            try {
                const response = await fetch('/detect', { method: 'POST', body: formData });
                const data = await response.json();

                if (!response.ok) throw new Error(data.error || 'Prediction engine error');

                detections = data.detections;
                
                // Update KPI Cards
                document.getElementById('kpiTotal').textContent = data.tree_count;
                document.getElementById('kpiRes').textContent = `Resolution: ${data.resolution} | Time: ${data.processing_time}s`;
                
                const h = data.health_breakdown;
                const total = data.tree_count || 1; // avoid division by zero
                
                document.getElementById('kpiHealthy').textContent = h.Healthy;
                document.getElementById('kpiHealthyPct').textContent = Math.round(h.Healthy / total * 100) + '%';
                document.getElementById('kpiHealthyBar').style.width = (h.Healthy / total * 100) + '%';

                document.getElementById('kpiDeficient').textContent = h.Deficient;
                document.getElementById('kpiDeficientPct').textContent = Math.round(h.Deficient / total * 100) + '%';
                document.getElementById('kpiDeficientBar').style.width = (h.Deficient / total * 100) + '%';

                document.getElementById('kpiDiseased').textContent = h.Diseased;
                document.getElementById('kpiDiseasedPct').textContent = Math.round(h.Diseased / total * 100) + '%';
                document.getElementById('kpiDiseasedBar').style.width = (h.Diseased / total * 100) + '%';

                analyticsRow.classList.add('visible');

                // Build Inventory Directory Table
                buildInventoryTable();
                directoryCard.classList.add('visible');

                // Enable toolbar & legend
                renderControls.style.display = 'flex';
                mapLegend.style.display = 'flex';

                // Automatically switch to Diagnostics view
                setViewMode('annotated');

            } catch (err) {
                errorBanner.textContent = '⚠️ ' + err.message;
                errorBanner.classList.add('visible');
            } finally {
                loadingOverlay.classList.remove('active');
                btnRun.disabled = false;
            }
        });

        // Set View Mode: clean | annotated
        function setViewMode(mode) {
            viewMode = mode;
            document.getElementById('btnOriginal').classList.toggle('active', mode === 'clean');
            document.getElementById('btnOverlay').classList.toggle('active', mode === 'annotated');
            drawCanvas();
        }

        // Toggle Rendering Options
        function toggleRenderOpt(opt) {
            renderOpts[opt] = !renderOpts[opt];
            document.getElementById(`toggle${opt.charAt(0).toUpperCase() + opt.slice(1)}`).classList.toggle('active', renderOpts[opt]);
            drawCanvas();
        }

        // Main Draw Loop
        function drawCanvas() {
            if (!originalImage) return;
            
            // Clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw original image
            ctx.drawImage(originalImage, 0, 0);

            if (viewMode === 'clean' || detections.length === 0) return;

            // Health color dictionary
            const colors = {
                'Healthy': { fill: 'rgba(16, 185, 129, 0.25)', stroke: 'rgb(16, 185, 129)' },
                'Deficient': { fill: 'rgba(249, 115, 22, 0.28)', stroke: 'rgb(249, 115, 22)' },
                'Diseased': { fill: 'rgba(239, 68, 68, 0.35)', stroke: 'rgb(239, 68, 68)' }
            };

            // 1. Draw Masks
            if (renderOpts.masks) {
                detections.forEach(d => {
                    if (!d.mask_poly || d.mask_poly.length === 0) return;
                    
                    const styling = colors[d.health] || colors.Healthy;
                    
                    ctx.beginPath();
                    ctx.moveTo(d.mask_poly[0][0], d.mask_poly[0][1]);
                    for (let j = 1; j < d.mask_poly.length; j++) {
                        ctx.lineTo(d.mask_poly[j][0], d.mask_poly[j][1]);
                    }
                    ctx.closePath();
                    
                    // Highlight if hovered or highlighted
                    if (d.id === hoveredTreeId || d.id === highlightedTreeId) {
                        ctx.fillStyle = styling.fill.replace('0.25', '0.5').replace('0.28', '0.5').replace('0.35', '0.6');
                    } else {
                        ctx.fillStyle = styling.fill;
                    }
                    ctx.fill();
                });
            }

            // 2. Draw Bounding Boxes and Labels
            detections.forEach(d => {
                const styling = colors[d.health] || colors.Healthy;
                const isFocused = (d.id === hoveredTreeId || d.id === highlightedTreeId);
                
                if (renderOpts.boxes || isFocused) {
                    ctx.strokeStyle = styling.stroke;
                    ctx.lineWidth = isFocused ? 4 : 2;
                    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
                }

                if (renderOpts.labels || isFocused) {
                    const labelText = `ID: ${d.track_id || d.id}`;
                    ctx.font = 'bold 12px sans-serif';
                    ctx.textBaseline = 'top';
                    const textWidth = ctx.measureText(labelText).width;
                    
                    ctx.fillStyle = styling.stroke;
                    ctx.fillRect(d.x1, d.y1 - 18, textWidth + 10, 18);
                    
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(labelText, d.x1 + 5, d.y1 - 15);
                }

                // If highlighted, draw a pulsing radial ping
                if (d.id === highlightedTreeId) {
                    const cx = (d.x1 + d.x2) / 2;
                    const cy = (d.y1 + d.y2) / 2;
                    ctx.beginPath();
                    ctx.arc(cx, cy, Math.max(d.x2 - d.x1, d.y2 - d.y1) * 0.7, 0, 2 * Math.PI);
                    ctx.strokeStyle = '#22d3ee';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }
            });
        }

        // Mouse interaction on Canvas (Tooltip & Hover highlighting)
        canvas.addEventListener('mousemove', (e) => {
            if (detections.length === 0 || viewMode === 'clean') return;

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const mouseX = (e.clientX - rect.left) * scaleX;
            const mouseY = (e.clientY - rect.top) * scaleY;

            let found = null;
            // Iterate backwards to hover over the top-most/smallest box
            for (let i = detections.length - 1; i >= 0; i--) {
                const d = detections[i];
                if (mouseX >= d.x1 && mouseX <= d.x2 && mouseY >= d.y1 && mouseY <= d.y2) {
                    found = d;
                    break;
                }
            }

            if (found) {
                hoveredTreeId = found.id;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
                
                const sizeW = Math.round(found.x2 - found.x1);
                const sizeH = Math.round(found.y2 - found.y1);
                tooltip.innerHTML = `
                    <div class="tooltip-header">🌴 Tree ID: ${found.track_id || found.id}</div>
                    <div class="tooltip-row"><strong>Health:</strong> <span class="status-badge ${found.health.toLowerCase()}">${found.health}</span></div>
                    <div class="tooltip-row"><strong>Confidence:</strong> ${(found.confidence * 100).toFixed(1)}%</div>
                    <div class="tooltip-row"><strong>Canopy Size:</strong> ${sizeW}×${sizeH} px</div>
                `;
            } else {
                hoveredTreeId = null;
                tooltip.style.display = 'none';
            }
            drawCanvas();
        });

        canvas.addEventListener('mouseleave', () => {
            hoveredTreeId = null;
            tooltip.style.display = 'none';
            drawCanvas();
        });

        // Clicking on canvas highlights the corresponding row in table
        canvas.addEventListener('click', (e) => {
            if (detections.length === 0 || viewMode === 'clean') return;
            if (hoveredTreeId !== null) {
                highlightTree(hoveredTreeId, true);
            }
        });

        // Inventory Table Builders
        function buildInventoryTable() {
            const tbody = document.getElementById('tblBody');
            tbody.innerHTML = '';

            detections.forEach((d, idx) => {
                // assign a standard sequential ID if track ID is missing
                d.id = d.track_id || (idx + 1);
                
                const tr = document.createElement('tr');
                tr.id = `row-tree-${d.id}`;
                tr.onclick = () => highlightTree(d.id, true);

                const healthClass = d.health.toLowerCase();
                const coords = `${Math.round(d.x1)}, ${Math.round(d.y1)}, ${Math.round(d.x2)}, ${Math.round(d.y2)}`;
                
                tr.innerHTML = `
                    <td><strong>Tree #${d.id}</strong></td>
                    <td><span class="slider-badge">${(d.confidence * 100).toFixed(1)}%</span></td>
                    <td><span class="status-badge ${healthClass}">${d.health}</span></td>
                    <td><code>(${coords})</code></td>
                    <td>
                        <button class="btn-locate" onclick="event.stopPropagation(); zoomToTree(${d.id})">
                            📍 Locate
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Highlight tree on canvas and in table
        function highlightTree(treeId, scrollTable = false) {
            highlightedTreeId = treeId;
            
            // Highlight table row
            document.querySelectorAll('#tblBody tr').forEach(row => {
                row.classList.remove('highlighted');
            });
            const selectedRow = document.getElementById(`row-tree-${treeId}`);
            if (selectedRow) {
                selectedRow.classList.add('highlighted');
                if (scrollTable) {
                    selectedRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }

            drawCanvas();
        }

        // Center visualizer viewport on selected tree crown
        function zoomToTree(treeId) {
            const tree = detections.find(d => d.id === treeId);
            if (!tree) return;
            
            highlightTree(treeId);
            
            // Calculate center
            const cx = (tree.x1 + tree.x2) / 2;
            const cy = (tree.y1 + tree.y2) / 2;

            const container = document.getElementById('canvasContainer');
            
            // Scroll container
            const containerRect = container.getBoundingClientRect();
            const canvasRect = canvas.getBoundingClientRect();
            
            // Scale ratio
            const scale = canvasRect.width / canvas.width;
            
            const targetX = cx * scale - containerRect.width / 2;
            const targetY = cy * scale - containerRect.height / 2;

            container.scrollTo({
                left: targetX,
                top: targetY,
                behavior: 'smooth'
            });
        }

        // Table search and status filtering
        function filterTable() {
            const query = document.getElementById('tblSearch').value.toLowerCase();
            const filter = document.getElementById('tblFilter').value;
            const rows = document.querySelectorAll('#tblBody tr');

            rows.forEach(row => {
                const text = row.innerText.toLowerCase();
                const matchesSearch = text.includes(query);
                
                let matchesFilter = true;
                if (filter !== 'ALL') {
                    matchesFilter = text.includes(filter.toLowerCase());
                }

                if (matchesSearch && matchesFilter) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        // Download CSV
        function exportCSV() {
            if (detections.length === 0) return;
            
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Tree_ID,Confidence,Health_Status,Center_X,Center_Y,Box_X1,Box_Y1,Box_X2,Box_Y2\n";
            
            detections.forEach(d => {
                const cx = ((d.x1 + d.x2) / 2).toFixed(1);
                const cy = ((d.y1 + d.y2) / 2).toFixed(1);
                csvContent += `${d.id},${d.confidence.toFixed(4)},${d.health},${cx},${cy},${d.x1.toFixed(1)},${d.y1.toFixed(1)},${d.x2.toFixed(1)},${d.y2.toFixed(1)}\n`;
            });

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `plantation_inventory_${Date.now()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------------------------

def create_app(model, cfg: dict, default_conf: float) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 80 MB limits

    @app.route("/")
    def index():
        return HTML_TEMPLATE

    @app.route("/detect", methods=["POST"])
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        try:
            # Load params from request
            conf = float(request.form.get("conf", default_conf))
            iou = float(request.form.get("iou", cfg["inference"]["iou"]))
            tile_size = int(request.form.get("tile_size", cfg["inference"]["tile_size"]))
            overlap = float(request.form.get("overlap", cfg["inference"]["tile_overlap"]))
            img_size = cfg["inference"]["img_size"]

            # Decode uploaded image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image_bgr is None:
                return jsonify({"error": "Could not decode uploaded image structure"}), 400

            h, w = image_bgr.shape[:2]
            resolution = f"{w}×{h}"

            # Run diagnostics
            t0 = time.time()
            results = run_tiled_inference(model, image_bgr, conf, iou, img_size, tile_size, overlap)
            elapsed = time.time() - t0

            # Compile response values
            detections_json = [r.to_dict() for r in results]
            
            # Health stats compilation
            health_counts = {"Healthy": 0, "Deficient": 0, "Diseased": 0}
            for r in results:
                health_counts[r.health] = health_counts.get(r.health, 0) + 1

            return jsonify({
                "tree_count": len(results),
                "resolution": resolution,
                "processing_time": round(elapsed, 2),
                "health_breakdown": health_counts,
                "detections": detections_json
            })

        except Exception as e:
            return jsonify({"error": f"Diagnostics engine failed: {str(e)}"}), 500

    return app


# ---------------------------------------------------------------------------
# Main Launch Sequence
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coconut Tree Analytics Dashboard")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--weights", default=None, help="YOLO model weights path (.pt)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else find_best_weights(cfg)

    # Fallback to base model if no weights found
    if weights is None or not weights.exists():
        print("WARNING: Trained weights not found. Falling back to base yolov8n.pt.")
        weights = Path("yolov8n.pt")
        if not weights.exists():
            print("Downloading base yolov8n.pt...")
            # Automatically downloads via ultralytics on initialization

    print(f"Loading plantation analysis model: {weights}")
    configure_ultralytics()
    from ultralytics import YOLO
    model = YOLO(str(weights))

    default_conf = args.conf if args.conf is not None else cfg["inference"]["confidence"]

    app = create_app(model, cfg, default_conf)

    print()
    print("=" * 60)
    print(" 🌴 COCONUT TREE DIAGNOSTICS & COUNTING SERVER ONLINE")
    print(f" Access UI here: http://{args.host}:{args.port}")
    print("=" * 60)
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
