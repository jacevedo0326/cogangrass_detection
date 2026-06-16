"""Precompute CLAHE-normalized tiles ONCE.

CLAHE is deterministic normalization (not random augmentation), so re-running it
every epoch in the dataloader is pure waste. This applies it once to disk:
tiles_dataset/ -> tiles_dataset_clahe/ (same structure). The DA training scripts
then read the precomputed folder and drop CLAHE from their transforms -> big
speedup, removes the heaviest CPU op from the per-epoch pipeline.

Run:  python precompute_clahe.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC = Path("tiles_dataset")
DST = Path("tiles_dataset_clahe")


def clahe(img):
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = cl.apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def main():
    n = 0
    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        out = DST / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for p in cls_dir.glob("*.jpg"):
            clahe(Image.open(p).convert("RGB")).save(out / p.name, quality=90)
            n += 1
        print(f"  {cls_dir.name}: done")
    print(f"wrote {n} CLAHE tiles -> {DST}/")


if __name__ == "__main__":
    main()
