"""Prepare raw drone frames for the tiling pipeline.

Downscales JPGs to drone_dataset/images (long side 1280) and copies YOLO box
labels to drone_dataset/labels. Images with NO matching .txt are treated as
all-negative: an empty label is written (every tile becomes not_cogongrass).
Appends to the existing dataset, so you can add new fields incrementally.

Run:  python prep_images.py <input_folder> [more_folders ...]
"""
import os
import sys
import shutil
from pathlib import Path

from PIL import Image

MAX = int(os.environ.get("PREP_MAX", "1280"))   # set PREP_MAX=4096 for full resolution
OUT_IMG = Path("drone_dataset/images")
OUT_LBL = Path("drone_dataset/labels")


def prepare(src):
    src = Path(src)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")]
    labels = {p.stem: p for p in src.rglob("*.txt")}
    n_pos = n_neg = 0
    for ip in imgs:
        st = ip.stem
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        if max(W, H) > MAX:
            s = MAX / max(W, H)
            im = im.resize((round(W * s), round(H * s)))
        im.save(OUT_IMG / f"{st}.jpg", "JPEG", quality=88)
        lp = labels.get(st)
        if lp and lp.read_text().strip():
            shutil.copy(lp, OUT_LBL / f"{st}.txt"); n_pos += 1
        else:
            (OUT_LBL / f"{st}.txt").write_text("")   # negative frame -> empty label
            n_neg += 1
    print(f"  {src}: {n_pos} labeled (+boxes) + {n_neg} negative (no boxes)")
    return n_pos, n_neg


def main(folders):
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)
    tp = tn = 0
    for f in folders:
        p, n = prepare(f); tp += p; tn += n
    print(f"\ntotal added: {tp} labeled + {tn} negative -> {OUT_IMG}/")
    print("next:  python boxes_to_tiles.py   then   python train_tiles.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python prep_images.py <input_folder> [more_folders ...]"); sys.exit(1)
    main(sys.argv[1:])
