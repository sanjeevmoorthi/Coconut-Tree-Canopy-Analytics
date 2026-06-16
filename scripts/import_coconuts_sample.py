"""
Import the public aerial coconut dataset (fast.ai / daveluo) into data/raw/.

Source: https://www.dropbox.com/s/g4isvnc577bg6ud/coconuts_0329_train.zip
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import ensure_dir

IMG_SIZE = 224
SRC = Path("_coconuts_train/train")
IMAGES_SRC = SRC / "jpegs"
IMAGES_DST = Path("data/raw/images")
LABELS_DST = Path("data/raw/annotations")


def mbb_to_yolo_line(ymin: float, xmin: float, ymax: float, xmax: float) -> str:
    cx = (xmin + xmax) / 2 / IMG_SIZE
    cy = (ymin + ymax) / 2 / IMG_SIZE
    w = (xmax - xmin) / IMG_SIZE
    h = (ymax - ymin) / IMG_SIZE
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    mbb_csv = SRC / "mbb_noempties.csv"
    if not mbb_csv.exists():
        raise SystemExit(
            f"Dataset not found at {SRC}.\n"
            "Download _coconuts_train.zip first (see README or run setup)."
        )

    ensure_dir(IMAGES_DST)
    ensure_dir(LABELS_DST)

    copied = 0
    skipped = 0

    with open(mbb_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["fn"]
            bbox_str = row["bbox"].strip()
            img_src = IMAGES_SRC / fn
            if not img_src.exists() or not bbox_str:
                skipped += 1
                continue

            coords = list(map(float, bbox_str.split()))
            lines = [
                mbb_to_yolo_line(*coords[i : i + 4])
                for i in range(0, len(coords), 4)
            ]

            shutil.copy2(img_src, IMAGES_DST / fn)
            (LABELS_DST / f"{Path(fn).stem}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            copied += 1

    print(f"Imported {copied} image/label pairs into data/raw/")
    if skipped:
        print(f"Skipped {skipped} rows (missing image or empty bbox)")


if __name__ == "__main__":
    main()
