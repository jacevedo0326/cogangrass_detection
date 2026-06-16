"""Convert existing YOLO patch boxes -> per-tile cogongrass labels + a tile dataset.

Bootstraps the tiling dataset from the boxes you already have: a tile is labeled
cogongrass if enough of its area falls inside a box. Writes the same JSON format
as label_tiles.py (so you can review/correct any frame in the GUI), and cuts the
tile images into class folders ready for training.

Run:  python boxes_to_tiles.py

Outputs:
  tile_labels/<stem>.json                 (review/correct in label_tiles.py)
  tiles_dataset/cogongrass/*.jpg          (each tile resized to 224x224)
  tiles_dataset/not_cogongrass/*.jpg
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMG_DIR = Path("drone_dataset/images")
LBL_DIR = Path("drone_dataset/labels")
OUT_LABELS = Path("tile_labels")
OUT_TILES = Path("tiles_dataset")
TILE = int(os.environ.get("TILE_PX", "160"))   # 160 on 1280px frames; set TILE_PX=512 for full-res 4096px
COVER_THRESH = 0.30   # tile = cogongrass if >= this fraction of it is inside a box
CNN_SIZE = int(os.environ.get("TILE_SAVE_PX", "224"))   # on-disk tile size; set 512 to store full detail (model input resize happens in training)
CUT_IMAGES = True     # also export the tile-image dataset
VEG_THRESH = 0.03     # drop sky/non-vegetation tiles: skip tile if mean ExG < this


def boxes_mask(label_path, W, H):
    """Boolean HxW mask, True where any YOLO box (normalized cx cy w h) covers a pixel."""
    m = np.zeros((H, W), dtype=bool)
    if not label_path.exists():
        return m
    for ln in label_path.read_text().split("\n"):
        if not ln.strip():
            continue
        _, cx, cy, bw, bh = ln.split()
        cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
        l, t = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        r, b = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
        m[t:b, l:r] = True
    return m


def main():
    OUT_LABELS.mkdir(exist_ok=True)
    if CUT_IMAGES:
        if OUT_TILES.exists():
            shutil.rmtree(OUT_TILES)          # start clean so no stale tiles linger
        (OUT_TILES / "cogongrass").mkdir(parents=True, exist_ok=True)
        (OUT_TILES / "not_cogongrass").mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    n_pos = n_neg = n_sky = 0
    for img in imgs:
        im = Image.open(img).convert("RGB")
        W, H = im.size
        arr = np.asarray(im).astype(np.float32)
        ssum = arr.sum(2) + 1e-6
        exg = 2 * arr[..., 1] / ssum - arr[..., 0] / ssum - arr[..., 2] / ssum  # green-ness map
        mask = boxes_mask(LBL_DIR / f"{img.stem}.txt", W, H)
        cols, rows = -(-W // TILE), -(-H // TILE)
        pos = []
        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * TILE, c * TILE
                y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
                if exg[y0:y1, x0:x1].mean() < VEG_THRESH:   # sky / non-vegetation -> drop entirely
                    n_sky += 1
                    continue
                is_cog = mask[y0:y1, x0:x1].mean() >= COVER_THRESH
                if is_cog:
                    pos.append([r, c])
                if CUT_IMAGES:
                    cls = "cogongrass" if is_cog else "not_cogongrass"
                    im.crop((x0, y0, x1, y1)).resize((CNN_SIZE, CNN_SIZE)).save(
                        OUT_TILES / cls / f"{img.stem}_r{r}_c{c}.jpg", quality=88)
                    n_pos, n_neg = n_pos + is_cog, n_neg + (not is_cog)
        (OUT_LABELS / f"{img.stem}.json").write_text(json.dumps({
            "image": img.name, "tile_px": TILE, "rows": rows, "cols": cols,
            "cogongrass": sorted(pos),
        }))

    print(f"wrote tile labels for {len(imgs)} images -> {OUT_LABELS}/")
    print(f"dropped {n_sky} sky/non-vegetation tiles (ExG < {VEG_THRESH})")
    if CUT_IMAGES:
        tot = n_pos + n_neg
        print(f"tile dataset: {n_pos} cogongrass + {n_neg} not_cogongrass "
              f"= {tot} tiles ({100*n_pos/tot:.0f}% positive) -> {OUT_TILES}/")


if __name__ == "__main__":
    main()
