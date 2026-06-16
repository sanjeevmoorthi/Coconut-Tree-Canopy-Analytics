"""
Print a quick reference for labeling coconut trees in YOLO format.

For GUI labeling, install labelImg or CVAT:
  pip install labelImg
  labelImg data/raw/images data/raw/annotations
"""

from __future__ import annotations

GUIDE = """
================================================================================
COCONUT TREE LABELING GUIDE (YOLO format)
================================================================================

1. IMAGE REQUIREMENTS
   - Use nadir (top-down) or near-nadir drone photos
   - Recommended GSD: 2-5 cm/pixel for individual crown detection
   - Include varied lighting, growth stages, and plantation densities

2. WHAT TO LABEL
   - Draw a tight bounding box around each visible coconut crown
   - Label partially visible trees at field edges
   - Skip dead stumps or non-coconut palms unless you want a separate class

3. YOLO LABEL FORMAT (one .txt per image, same filename as image)
   class_id center_x center_y width height
   (all coordinates normalized 0-1 relative to image size)

   Example for a single tree (class 0):
   0 0.512000 0.348000 0.045000 0.052000

4. RECOMMENDED ANNOTATION TOOLS
   - labelImg  : pip install labelImg  ->  save as YOLO format
   - CVAT      : https://cvat.ai  ->  export as YOLO 1.1
   - Roboflow  : https://roboflow.com  ->  upload & export YOLO

5. MINIMUM DATASET SIZE (rule of thumb)
   - Proof of concept : 100-200 annotated images
   - Production use   : 500+ images across seasons/locations
   - Aim for 50+ tree instances per 100 images in varied layouts

6. AFTER LABELING
   python scripts/prepare_dataset.py
   python scripts/train.py
   python scripts/predict_and_count.py --weights runs/.../best.pt --source <image> --save
================================================================================
"""


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
