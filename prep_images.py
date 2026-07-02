"""Prepare raw drone frames for the tiling pipeline.

Normalizes JPGs into drone_dataset/images (long side PREP_MAX, default 4096 —
full resolution) and copies YOLO box labels to drone_dataset/labels. Images with
NO matching .txt are treated as all-negative: an empty label is written (every
tile becomes not_cogongrass). Appends to the existing dataset, so you can add
new fields incrementally.

Safety (plan U2, R9):
  * Frame-stem collisions are an ERROR — the same stem arriving from two
    different source paths in one run, or an output overwrite whose content
    differs from what is already on disk, would silently blend/replace frames.
    Set ALLOW_OVERWRITE=1 to bypass deliberately.
  * The effective PREP_MAX is recorded in drone_dataset/_prep_provenance.json
    (atomic write) so a retile over mixed-resolution frames is flaggable.

Run:  python prep_images.py <input_folder> [more_folders ...]
"""
import io
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image

import tile_common

MAX = int(os.environ.get("PREP_MAX", "4096"))   # full-res default; set PREP_MAX=1280 for the legacy downscale
JPEG_QUALITY = 88
OUT_IMG = Path("drone_dataset/images")
OUT_LBL = Path("drone_dataset/labels")
PREP_PROVENANCE = Path("drone_dataset/_prep_provenance.json")
ALLOW_OVERWRITE = os.environ.get("ALLOW_OVERWRITE") == "1"


class StemCollision(RuntimeError):
    """Two different sources map to one output frame stem — refusing to blend them."""


def check_stem_collision(stem: str, src, seen: dict, allow_overwrite: bool = False):
    """Error if ``stem`` was already produced by a DIFFERENT source path this run.

    ``seen`` maps stem -> the source path that first produced it; the caller keeps
    it across intake folders. Raises ``StemCollision`` naming BOTH sources (R9).
    """
    prev = seen.get(stem)
    if prev is not None and Path(prev) != Path(src) and not allow_overwrite:
        raise StemCollision(
            f"frame stem {stem!r} collides: already produced from {prev}, now also "
            f"from {src}; rename one source (or set ALLOW_OVERWRITE=1 to bypass)")
    seen[stem] = src


def check_output_overwrite(out_path, new_bytes: bytes, src,
                           allow_overwrite: bool = False):
    """Error if writing would replace an existing output with DIFFERENT content.

    Content difference is detected by size (the outputs are deterministic JPEG
    re-encodes, so identical input -> identical bytes). Raises ``StemCollision``
    naming the existing file and the new source.
    """
    out_path = Path(out_path)
    if (out_path.exists() and out_path.stat().st_size != len(new_bytes)
            and not allow_overwrite):
        raise StemCollision(
            f"{out_path} already exists with different content "
            f"({out_path.stat().st_size} bytes on disk vs {len(new_bytes)} new) — "
            f"source {src} would silently replace an earlier frame; "
            f"set ALLOW_OVERWRITE=1 to bypass")


def write_prep_provenance(path=PREP_PROVENANCE, prep_max: int = MAX):
    """Record the effective PREP_MAX atomically (tmp -> fsync -> os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"prep_max": prep_max, "jpeg_quality": JPEG_QUALITY,
               "created_at": datetime.now().isoformat(),
               "git": tile_common.git_state()}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def prepare(src, seen: dict):
    src = Path(src)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")]
    labels = {p.stem: p for p in src.rglob("*.txt")}
    n_pos = n_neg = 0
    for ip in imgs:
        st = ip.stem
        check_stem_collision(st, ip, seen, allow_overwrite=ALLOW_OVERWRITE)
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        if max(W, H) > MAX:
            s = MAX / max(W, H)
            im = im.resize((round(W * s), round(H * s)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
        check_output_overwrite(OUT_IMG / f"{st}.jpg", data, ip,
                               allow_overwrite=ALLOW_OVERWRITE)
        (OUT_IMG / f"{st}.jpg").write_bytes(data)
        lp = labels.get(st)
        if lp and lp.read_text().strip():
            (OUT_LBL / f"{st}.txt").write_text(lp.read_text()); n_pos += 1
        else:
            (OUT_LBL / f"{st}.txt").write_text("")   # negative frame -> empty label
            n_neg += 1
    print(f"  {src}: {n_pos} labeled (+boxes) + {n_neg} negative (no boxes)")
    return n_pos, n_neg


def main(folders):
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)
    seen: dict = {}                       # stem -> source path, shared across folders
    tp = tn = 0
    for f in folders:
        p, n = prepare(f, seen); tp += p; tn += n
    write_prep_provenance()
    print(f"\ntotal added: {tp} labeled + {tn} negative -> {OUT_IMG}/  (PREP_MAX={MAX})")
    print("next:  python boxes_to_tiles.py   then   python train_tiles.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python prep_images.py <input_folder> [more_folders ...]"); sys.exit(1)
    main(sys.argv[1:])
