"""Convert existing YOLO patch boxes -> per-tile cogongrass labels + a tile dataset.

Bootstraps the tiling dataset from the boxes you already have: a tile is labeled
cogongrass if enough of its area falls inside a box. Writes the same JSON format
as label_tiles.py (so you can review/correct any frame in the GUI), and cuts the
tile images into class folders ready for training.

Identity/tiling/veg-filter logic is shared via tile_common (plan U2, R1/R5).

Safety (R7/R8):
  * A tiling-provenance manifest (tiles_dataset/_provenance.json) is written
    atomically after a successful run. On the next run, if the existing
    manifest's params differ from the requested ones, the wipe is REFUSED
    (naming each differing param) unless FORCE_RETILE=1 or --force is set.
  * The wipe only ever touches tiles_dataset/{cogongrass,not_cogongrass} and the
    manifest — per-collection subdirectories (e.g. future orthomosaic
    collections) are out of its reach.
  * tile_labels/<stem>.json files carrying "human_edited": true (set by
    label_tiles.py on save) are never overwritten by this bootstrap.

Run:  python boxes_to_tiles.py [--force]

Outputs:
  tile_labels/<stem>.json                 (review/correct in label_tiles.py)
  tiles_dataset/cogongrass/*.jpg          (each tile resized to TILE_SAVE_PX)
  tiles_dataset/not_cogongrass/*.jpg
  tiles_dataset/_provenance.json          (tiling-provenance manifest)
"""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import tile_common

IMG_DIR = Path("drone_dataset/images")
LBL_DIR = Path("drone_dataset/labels")
OUT_LABELS = Path("tile_labels")
OUT_TILES = Path("tiles_dataset")
TILE = int(os.environ.get("TILE_PX", "512"))   # full-res default; TILE_PX=160 for the legacy 1280px layout
COVER_THRESH = tile_common.COVER_THRESH   # tile = cogongrass if >= this fraction of it is inside a box
CNN_SIZE = int(os.environ.get("TILE_SAVE_PX", "512"))   # on-disk tile size (model input resize happens in training)
CUT_IMAGES = True     # also export the tile-image dataset
VEG_THRESH = tile_common.VEG_THRESH   # drop sky/non-vegetation tiles: skip tile if mean ExG < this
JPEG_QUALITY = 88

# The manifest params that must match for a silent re-wipe to be allowed. prep_max is
# compared only when known on both sides (it comes from the env / prep provenance).
_COMPARED_PARAMS = ("tile_px", "tile_save_px", "jpeg_quality", "veg_thresh", "prep_max")


def requested_params() -> dict:
    """The tiling params this run would use (compared against the existing manifest)."""
    prep_max = os.environ.get("PREP_MAX")
    if prep_max is None and Path("drone_dataset/_prep_provenance.json").exists():
        prep_max = json.loads(
            Path("drone_dataset/_prep_provenance.json").read_text()).get("prep_max")
    return {"tile_px": TILE, "tile_save_px": CNN_SIZE,
            "prep_max": int(prep_max) if prep_max is not None else None,
            "jpeg_quality": JPEG_QUALITY, "veg_thresh": VEG_THRESH}


def check_wipe_allowed(existing_manifest: dict | None, params: dict,
                       force: bool = False) -> bool:
    """Refuse to wipe a dataset whose recorded provenance differs from this run (R7).

    Returns True when the wipe may proceed (no manifest, matching params, or
    ``force``). Raises SystemExit naming EACH differing param otherwise.
    ``prep_max`` is only compared when known on both sides.
    """
    if force or existing_manifest is None:
        return True
    diffs = []
    for k in _COMPARED_PARAMS:
        old, new = existing_manifest.get(k), params.get(k)
        if k == "prep_max" and (old is None or new is None):
            continue
        if old != new:
            diffs.append(f"{k}: existing={old!r} requested={new!r}")
    if diffs:
        raise SystemExit(
            "REFUSING to wipe tiles_dataset/: the existing provenance manifest "
            "differs from the requested run on:\n  " + "\n  ".join(diffs) +
            "\nRe-run with FORCE_RETILE=1 (or --force) to retile anyway.")
    return True


def wipe_class_dirs(out_tiles: Path = OUT_TILES) -> None:
    """Remove ONLY the two class dirs + the manifest — never per-collection subdirs.

    Orthomosaic collections live under tiles_dataset/<collection>/ (plan U6); the
    oblique wipe must not be able to reach them, so the rmtree is scoped to the
    class folders instead of the dataset root.
    """
    out_tiles = Path(out_tiles)
    for cls in tile_common.CLASSES:
        d = out_tiles / cls
        if d.exists():
            shutil.rmtree(d)
    manifest = out_tiles / tile_common.PROVENANCE_NAME
    if manifest.exists():
        manifest.unlink()


def write_tile_label(path, payload: dict) -> bool:
    """Write a bootstrap tile-label JSON unless the existing one is human-edited (R8).

    label_tiles.py sets ``"human_edited": true`` on save; such files survive
    re-runs of this bootstrap. Returns True if written, False if skipped.
    """
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = {}
        if existing.get("human_edited"):
            print(f"  keeping human-edited labels: {path.name} (skip bootstrap overwrite)")
            return False
    path.write_text(json.dumps(payload))
    return True


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


def main(force: bool = False):
    force = force or os.environ.get("FORCE_RETILE") == "1"
    params = requested_params()
    OUT_LABELS.mkdir(exist_ok=True)
    if CUT_IMAGES:
        existing = None
        if (OUT_TILES / tile_common.PROVENANCE_NAME).exists():
            existing = tile_common.read_provenance(OUT_TILES)
        check_wipe_allowed(existing, params, force=force)
        wipe_class_dirs(OUT_TILES)                # scoped: class dirs + manifest only
        (OUT_TILES / "cogongrass").mkdir(parents=True, exist_ok=True)
        (OUT_TILES / "not_cogongrass").mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    n_pos = n_neg = n_sky = 0
    for img in imgs:
        im = Image.open(img).convert("RGB")
        W, H = im.size
        exg = tile_common.exg_map(im)             # green-ness map
        mask = boxes_mask(LBL_DIR / f"{img.stem}.txt", W, H)
        cols, rows = tile_common.tile_grid(W, H, TILE)
        pos = []
        for r, c, box in tile_common.tile_boxes(W, H, TILE):
            if not tile_common.tile_is_veg(exg, box, VEG_THRESH):  # sky / non-vegetation -> drop
                n_sky += 1
                continue
            x0, y0, x1, y1 = box
            is_cog = mask[y0:y1, x0:x1].mean() >= COVER_THRESH
            if is_cog:
                pos.append([r, c])
            if CUT_IMAGES:
                cls = "cogongrass" if is_cog else "not_cogongrass"
                tile_common.cut_tile(im, box, CNN_SIZE).save(
                    OUT_TILES / cls / f"{img.stem}_r{r}_c{c}.jpg", quality=JPEG_QUALITY)
                n_pos, n_neg = n_pos + is_cog, n_neg + (not is_cog)
        write_tile_label(OUT_LABELS / f"{img.stem}.json", {
            "image": img.name, "tile_px": TILE, "rows": rows, "cols": cols,
            "cogongrass": sorted(pos),
        })

    print(f"wrote tile labels for {len(imgs)} images -> {OUT_LABELS}/")
    print(f"dropped {n_sky} sky/non-vegetation tiles (ExG < {VEG_THRESH})")
    if CUT_IMAGES:
        tot = n_pos + n_neg
        print(f"tile dataset: {n_pos} cogongrass + {n_neg} not_cogongrass "
              f"= {tot} tiles ({100*n_pos/tot:.0f}% positive) -> {OUT_TILES}/")
        tile_common.write_provenance(OUT_TILES, {
            **params,
            "source_digest": tile_common.source_digest(imgs),
            "created_at": datetime.now().isoformat(),
        })
        print(f"provenance manifest -> {OUT_TILES / tile_common.PROVENANCE_NAME}")


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])
