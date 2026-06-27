"""Data-input variants for the sweep: tile size × CLAHE × source-resolution × ExG (U2).

Materializes the R1 variant set the sweep consumes and records which exist in a manifest.
Each variant is a torchvision-`ImageFolder` directory (``cogongrass/`` + ``not_cogongrass/``)
built by the *same* tiling rule as ``boxes_to_tiles.py`` (the ≥30%-area label rule + the
ExG green-filter) and the *same* CLAHE as ``precompute_clahe.py`` — ported here (not
imported) so this module stays self-contained and the per-tile rule can be unit-tested on
synthetic fixtures without the heavy re-tiling pipeline.

The current ``tiles_dataset/`` is the **reference** variant; ``tiles_dataset_clahe/`` is
reference + CLAHE. Other variants get canonical ``tiles_dataset_<tag>/`` dirs. Building is
idempotent (a populated variant dir is left untouched).

Run:
    python arch_sweep/data_variants.py --list                 # manifest of available variants
    python arch_sweep/data_variants.py --build tile224        # build one variant
    python arch_sweep/data_variants.py --build-all            # build the standard grid

Note (CLAUDE.md): the ``prep_max`` source-resolution knob is realized by re-running
``prep_images.py`` with ``PREP_MAX`` before tiling. ``ensure_prepped`` shells that out; the
default flow tiles from whatever ``drone_dataset/images`` already holds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "drone_dataset" / "images"
LBL_DIR = REPO_ROOT / "drone_dataset" / "labels"
MANIFEST_PATH = Path(__file__).resolve().parent / "results" / "variants_manifest.json"
CLASSES = ("cogongrass", "not_cogongrass")


# ---------------------------------------------------------------------------
# Ported per-tile rule  (mirrors boxes_to_tiles.py:34-83 — copied, not imported)
# ---------------------------------------------------------------------------
def exg_map(arr: np.ndarray) -> np.ndarray:
    """Excess-green (green-ness) map for an HxWx3 RGB array (boxes_to_tiles.py:64-65)."""
    arr = arr.astype(np.float32)
    ssum = arr.sum(2) + 1e-6
    return 2 * arr[..., 1] / ssum - arr[..., 0] / ssum - arr[..., 2] / ssum


def boxes_mask(label_path: Path, W: int, H: int) -> np.ndarray:
    """Boolean HxW mask, True where any YOLO box covers a pixel (boxes_to_tiles.py:34-47)."""
    m = np.zeros((H, W), dtype=bool)
    if not Path(label_path).exists():
        return m
    for ln in Path(label_path).read_text().split("\n"):
        if not ln.strip():
            continue
        _, cx, cy, bw, bh = ln.split()
        cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
        l, t = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        r, b = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
        m[t:b, l:r] = True
    return m


def tile_records(arr: np.ndarray, mask: np.ndarray, tile_px: int,
                 cover_thresh: float = 0.30, veg_thresh: float = 0.03,
                 exg_filter: bool = True) -> list[dict]:
    """Per-tile labels for one frame — the pure core of the tiling rule.

    A tile is ``is_cog`` when ≥``cover_thresh`` of its area is inside a box; it is ``kept``
    unless the ExG filter is on and its mean green-ness is below ``veg_thresh`` (sky /
    bare-ground drop). Both toggles are explicit so the ablations are testable in isolation.
    """
    H, W = arr.shape[:2]
    exg = exg_map(arr)
    cols, rows = -(-W // tile_px), -(-H // tile_px)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            y1, x1 = min(H, y0 + tile_px), min(W, x0 + tile_px)
            mean_exg = float(exg[y0:y1, x0:x1].mean())
            kept = (not exg_filter) or (mean_exg >= veg_thresh)
            is_cog = bool(mask[y0:y1, x0:x1].mean() >= cover_thresh)
            out.append({"r": r, "c": c, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "is_cog": is_cog, "kept": kept, "exg": mean_exg})
    return out


# ---------------------------------------------------------------------------
# Multi-scale / flip TTA views  (U6 — deterministic test-time views, averaged downstream)
# ---------------------------------------------------------------------------
# A "view" is a deterministic, label-preserving image transform applied at extract time; each
# view yields its own cached features, and the per-tile head probs are averaged across views
# (reuse ensemble.average_probs — a view is just a pseudo-member). Pure array transforms so the
# shape-preservation invariant is unit-testable without the backbones.
def hflip_view(arr: np.ndarray) -> np.ndarray:
    """Horizontal flip (cogongrass texture is orientation-agnostic, so the label is preserved)."""
    return np.ascontiguousarray(np.asarray(arr)[:, ::-1, :])


def scale_view(arr: np.ndarray, factor: float = 0.9) -> np.ndarray:
    """Center scale-and-crop/pad back to the original HxW (a multi-scale view), shape-preserving."""
    from PIL import Image

    a = np.asarray(arr)
    H, W = a.shape[:2]
    nh, nw = max(1, int(round(H * factor))), max(1, int(round(W * factor)))
    resized = np.asarray(Image.fromarray(a).resize((nw, nh)))
    out = np.zeros_like(a)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    if factor <= 1.0:                          # smaller view -> pad-center into the frame
        out[y0:y0 + nh, x0:x0 + nw] = resized
    else:                                      # larger view -> center-crop back to HxW
        cy, cx = (nh - H) // 2, (nw - W) // 2
        out = resized[cy:cy + H, cx:cx + W]
    return np.ascontiguousarray(out)


VIEW_TRANSFORMS = {
    "identity": lambda a: np.ascontiguousarray(np.asarray(a)),
    "hflip": hflip_view,
    "scale90": lambda a: scale_view(a, 0.9),
}


def clahe(img):
    """CLAHE on the L channel (precompute_clahe.py:21-25). Lazy cv2 import (heavy dep)."""
    import cv2
    from PIL import Image
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = cl.apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ---------------------------------------------------------------------------
# Variant specs + canonical naming
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantSpec:
    name: str
    tile_px: int = 512
    tile_save_px: int = 512
    prep_max: int = 4096
    clahe: bool = False
    exg_filter: bool = True
    cover_thresh: float = 0.30
    veg_thresh: float = 0.03
    dir_override: str | None = None   # reference variants point at the existing dirs

    def dir_name(self) -> str:
        if self.dir_override:
            return self.dir_override
        tag = f"t{self.tile_px}s{self.tile_save_px}p{self.prep_max}"
        if not self.exg_filter:
            tag += "_noexg"
        if self.clahe:
            tag += "_clahe"
        return f"tiles_dataset_{tag}"

    def path(self, root: Path | str = REPO_ROOT) -> Path:
        return Path(root) / self.dir_name()


# The standard grid the sweep consumes: tile-size × CLAHE primary, plus one-axis
# PREP_MAX and ExG ablations. ``reference`` is the as-built ``tiles_dataset/``.
STANDARD_VARIANTS = [
    VariantSpec("reference", dir_override="tiles_dataset"),
    VariantSpec("reference_clahe", clahe=True, dir_override="tiles_dataset_clahe"),
    # Cleaned 0422 eval variant (U3): reference tiling with suspect 0422 negatives corrected.
    VariantSpec("reference_0422clean", dir_override="tiles_dataset_0422clean"),
    VariantSpec("tile224", tile_px=224, tile_save_px=224),
    VariantSpec("tile224_clahe", tile_px=224, tile_save_px=224, clahe=True),
    VariantSpec("tile512", tile_px=512, tile_save_px=512),
    VariantSpec("tile512_clahe", tile_px=512, tile_save_px=512, clahe=True),
    VariantSpec("prep1280", prep_max=1280),          # source-resolution ablation
    VariantSpec("noexg", exg_filter=False),          # ExG green-filter ablation
]
VARIANTS_BY_NAME = {v.name: v for v in STANDARD_VARIANTS}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def count_tiles(variant_dir: Path | str) -> dict | None:
    """Per-class jpg counts for a variant dir, or None if it doesn't exist."""
    d = Path(variant_dir)
    if not d.exists():
        return None
    counts = {cls: len(list((d / cls).glob("*.jpg"))) for cls in CLASSES}
    counts["total"] = sum(counts[c] for c in CLASSES)
    return counts


def build_manifest(specs=STANDARD_VARIANTS, root: Path | str = REPO_ROOT,
                   manifest_path: Path | str = MANIFEST_PATH) -> dict:
    """Scan every spec's dir, record existence + tile counts, write the manifest JSON.

    Pure scan — never builds anything — so it is safe to call repeatedly and is the
    single source of "which variants exist" for the orchestrator/report.
    """
    entries = {}
    for s in specs:
        d = s.path(root)
        entries[s.name] = {**asdict(s), "dir": str(d), "exists": d.exists(),
                           "counts": count_tiles(d)}
    out = {"variants": entries}
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out


def read_manifest(manifest_path: Path | str = MANIFEST_PATH) -> dict:
    p = Path(manifest_path)
    return json.loads(p.read_text()) if p.exists() else {"variants": {}}


def available_variants(manifest_path: Path | str = MANIFEST_PATH) -> list[str]:
    return [name for name, e in read_manifest(manifest_path)["variants"].items() if e["exists"]]


# ---------------------------------------------------------------------------
# Materialization (heavy; idempotent). No unit test for the wrapper (plan U2).
# ---------------------------------------------------------------------------
def ensure_prepped(prep_max: int, raw_folders: list[str]) -> None:
    """(Re)build ``drone_dataset/images`` at a source resolution via prep_images.py.

    Shells out with ``PREP_MAX`` set (the CLAUDE.md env-override convention) so the
    PREP_MAX ablation is realized by the existing, trusted prep pipeline.
    """
    env = {**os.environ, "PREP_MAX": str(prep_max)}
    subprocess.run([sys.executable, str(REPO_ROOT / "prep_images.py"), *raw_folders],
                   cwd=REPO_ROOT, env=env, check=True)


def materialize(spec: VariantSpec, root: Path | str = REPO_ROOT, overwrite: bool = False) -> Path:
    """Tile ``drone_dataset/{images,labels}`` into the variant dir (idempotent).

    Uses the ported per-tile rule so the variant's tile size, label-area threshold, ExG
    toggle, and CLAHE are honored exactly. Skips work if the dir is already populated
    unless ``overwrite``. Returns the variant dir.
    """
    from PIL import Image

    out = spec.path(root)
    if out.exists() and count_tiles(out)["total"] > 0 and not overwrite:
        print(f"[skip] {spec.name}: {out} already populated ({count_tiles(out)['total']} tiles)")
        return out
    img_dir = Path(root) / "drone_dataset" / "images"
    lbl_dir = Path(root) / "drone_dataset" / "labels"
    if not img_dir.exists():
        raise SystemExit(f"{img_dir} missing — run prep_images.py first (PREP_MAX={spec.prep_max})")
    for cls in CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = n_drop = 0
    for img in sorted(img_dir.glob("*.jpg")):
        im = Image.open(img).convert("RGB")
        arr = np.asarray(im)
        W, H = im.size
        mask = boxes_mask(lbl_dir / f"{img.stem}.txt", W, H)
        for t in tile_records(arr, mask, spec.tile_px, spec.cover_thresh,
                              spec.veg_thresh, spec.exg_filter):
            if not t["kept"]:
                n_drop += 1
                continue
            cls = "cogongrass" if t["is_cog"] else "not_cogongrass"
            crop = im.crop((t["x0"], t["y0"], t["x1"], t["y1"])).resize(
                (spec.tile_save_px, spec.tile_save_px))
            if spec.clahe:
                crop = clahe(crop)
            crop.save(out / cls / f"{img.stem}_r{t['r']}_c{t['c']}.jpg", quality=88)
            n_pos += t["is_cog"]
            n_neg += not t["is_cog"]
    print(f"[built] {spec.name} -> {out}: {n_pos} cogongrass + {n_neg} not_cogongrass "
          f"({n_drop} dropped)")
    return out


# ---------------------------------------------------------------------------
# Label cleaning -> cleaned 0422 variant  (U3 — consumes the U1 score sidecars, KTD3)
# ---------------------------------------------------------------------------
def merge_sidecar_scores(sidecars):
    """Average ``p_cogongrass`` across one-or-more U1 sidecars, aligned by tile path.

    Each sidecar is a ``list[common.ScoreRecord]``. Path sets must match exactly across
    sidecars — a mismatch is raised, not silently averaged (the same alignment discipline U5
    uses). With a single sidecar this is the identity. Returns a merged ``list[ScoreRecord]``.
    """
    import common as C

    if not sidecars:
        return []
    ref_paths = [r.path for r in sidecars[0]]
    ref_set = set(ref_paths)
    for s in sidecars[1:]:
        if {r.path for r in s} != ref_set:
            raise ValueError("sidecar path sets differ — cannot ensemble-average for cleaning")
    by_path: dict[str, list] = {}
    for s in sidecars:
        for r in s:
            by_path.setdefault(r.path, []).append(r)
    out = []
    for p in ref_paths:
        recs = by_path[p]
        mean_p = sum(r.p_cogongrass for r in recs) / len(recs)
        out.append(C.ScoreRecord(path=p, frame=recs[0].frame,
                                 true_label=recs[0].true_label, p_cogongrass=mean_p))
    return out


def rank_suspect_negatives(records, *, min_p: float = 0.5, date: str = "20260422"):
    """Rank true-NEGATIVE tiles by model ``P(cogongrass)`` desc — the likely-mislabeled (R1).

    Only ``not_cogongrass`` tiles in collection ``date`` with ``p_cogongrass >= min_p`` are
    returned, highest-confidence first — the review queue for ``label_tiles.py``. ``date=None``
    disables the collection filter (for synthetic fixtures).
    """
    import common as C

    out = []
    for r in records:
        if r.true_label == C.COG_CLASS:
            continue
        if date and C.date_of(r.frame) != date:
            continue
        if r.p_cogongrass >= min_p:
            out.append({"path": r.path, "frame": r.frame, "p_cogongrass": r.p_cogongrass})
    out.sort(key=lambda d: -d["p_cogongrass"])
    return out


def class_counts(samples, *, date: str | None = None) -> dict:
    """Per-class tile counts (optionally for one collection) — the pre-relabel snapshot (U3).

    Class is read from each tile's parent folder (the ImageFolder label), so the snapshot is
    robust to the label-index map. ``date`` restricts to one collection (e.g. ``"20260422"``).
    """
    import common as C

    counts = {c: 0 for c in CLASSES}
    for p, _lab in samples:
        if date and C.date_of(C.frame_of(p)) != date:
            continue
        cls = Path(p).parent.name
        if cls in counts:
            counts[cls] += 1
    counts["total"] = sum(counts[c] for c in CLASSES)
    return counts


def build_clean_variant(relabel: dict, *, root: Path | str = REPO_ROOT, source: str = "reference",
                        out_name: str = "reference_0422clean", overwrite: bool = False) -> tuple[Path, int]:
    """Materialize the cleaned eval variant from a ``{tile_filename: corrected_class}`` map.

    Every source tile is hard-linked (copy fallback) into the out variant under its class;
    only tiles named in ``relabel`` move to the corrected class. Same image bytes, corrected
    ground truth — the clean ruler U2's report column reads. Idempotent. Returns
    ``(out_dir, n_flipped)``.
    """
    import shutil

    src = VARIANTS_BY_NAME[source].path(root)
    out = VARIANTS_BY_NAME[out_name].path(root)
    if out.exists() and (count_tiles(out) or {}).get("total", 0) > 0 and not overwrite:
        print(f"[skip] {out_name}: {out} already populated")
        return out, 0
    for cls in CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)
    n_flipped = 0
    for cls in CLASSES:
        for tile in sorted((src / cls).glob("*.jpg")):
            corrected = relabel.get(tile.name, cls)
            if corrected not in CLASSES:
                corrected = cls
            if corrected != cls:
                n_flipped += 1
            dst = out / corrected / tile.name
            if dst.exists():
                if overwrite:
                    dst.unlink()
                else:
                    continue
            try:
                os.link(tile, dst)              # hardlink: no extra disk for the shared bytes
            except OSError:
                shutil.copy2(tile, dst)         # cross-device / unsupported -> copy
    print(f"[built] {out_name} -> {out}: {n_flipped} tiles relabeled vs {source}")
    return out, n_flipped


def main():
    ap = argparse.ArgumentParser(description="Materialize / list arch_sweep data variants")
    ap.add_argument("--list", action="store_true", help="rebuild + print the variant manifest")
    ap.add_argument("--build", metavar="NAME", help="build one standard variant by name")
    ap.add_argument("--build-all", action="store_true", help="build every standard variant")
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if the dir exists")
    args = ap.parse_args()

    if args.build:
        if args.build not in VARIANTS_BY_NAME:
            raise SystemExit(f"unknown variant {args.build!r}; choices: {list(VARIANTS_BY_NAME)}")
        materialize(VARIANTS_BY_NAME[args.build], overwrite=args.overwrite)
    if args.build_all:
        for s in STANDARD_VARIANTS:
            if s.dir_override:   # reference dirs are built by the existing pipeline, not here
                continue
            materialize(s, overwrite=args.overwrite)

    manifest = build_manifest()
    print(f"\nmanifest -> {MANIFEST_PATH}")
    for name, e in manifest["variants"].items():
        tag = f"{e['counts']['total']} tiles" if e["exists"] and e["counts"] else "MISSING"
        print(f"  {name:18s} {e['dir']:42s} {tag}")


if __name__ == "__main__":
    main()
