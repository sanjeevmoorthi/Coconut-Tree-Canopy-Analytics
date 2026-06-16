# Dataset Guide

This project uses YOLO-format labels for coconut tree crown detection.

## Detection Dataset Layout

Place raw training data here:

```text
data/raw/
|-- images/
|   |-- 000000.jpg
|   |-- 000001.jpg
|   `-- ...
`-- annotations/
    |-- 000000.txt
    |-- 000001.txt
    `-- ...
```

Every image should have a matching label file with the same stem.

Example:

```text
data/raw/images/000030.jpg
data/raw/annotations/000030.txt
```

## YOLO Detection Label Format

Each line describes one tree crown:

```text
class_id center_x center_y width height
```

All coordinates are normalized from 0 to 1 relative to image width and height.

Example:

```text
0 0.512000 0.348000 0.045000 0.052000
```

For this project:

```text
0 = coconut_tree
```

## Prepare Splits

Run:

```bash
python scripts/prepare_dataset.py
```

The script creates:

```text
data/processed/
|-- images/train
|-- images/val
|-- images/test
|-- labels/train
|-- labels/val
|-- labels/test
`-- dataset.yaml
```

The default split is:

- 70% train
- 20% validation
- 10% test

## Segmentation Labels

The script below converts bounding boxes into approximate rectangle or octagon polygons:

```bash
python scripts/convert_dataset_to_seg.py --type octagon
```

This creates labels for YOLO segmentation under:

```text
data/raw/annotations_seg/
data/processed_seg/
```

Important: these generated masks are approximations. For higher-quality segmentation, manually annotate true canopy polygons with a tool such as CVAT, Label Studio, Roboflow, or LabelMe.

## GitHub Storage Rule

Training datasets are usually too large for normal GitHub repositories. Keep the dataset local or host it separately, then document the download link.

This repo ignores local dataset contents by default.
