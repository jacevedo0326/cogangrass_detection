"""SAM label-repair + segmentation/coverage geometry (U8).

Two uses, both reducing to the same pixel→tile geometry:

(a) **Offline label repair** — intersect each YOLO box with SAM masks so a tile is positive
    only where *grass pixels* fall, not merely where a (loose, rectangular) box sits. The
    box∩mask label is collapsed to the 512px tile grid and compared to the original box-only
    label; tiles that were positive-by-box but contain no grass flip to negative, yielding a
    cleaner-label variant (materialized via ``data_variants.build_clean_variant``).

(b) **Deploy reframe** — segment → per-tile coverage → tile labels for the standard metric,
    plus pixel IoU / coverage-MAE as the richer segmentation read.

Pure geometry (this module's core) is CPU-testable; ``run_sam_smoke`` is the failure-tolerant
fit gate for the actual model load (KTD6). Mask generation uses ultralytics SAM (``sam_explore.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402

DEFAULT_SAM_MODEL = "sam2_l.pt"   # GB10 fits SAM2-large (the old 6 GB card forced sam2_t)


# ---------------------------------------------------------------------------
# Pure mask geometry  (CPU-testable; no model needed)
# ---------------------------------------------------------------------------
def union_masks(masks, shape) -> np.ndarray:
    """Boolean OR of a list of HxW masks (empty list -> all-False of ``shape``)."""
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        out |= np.asarray(m, dtype=bool)
    return out


def box_intersect_mask(box_mask: np.ndarray, sam_masks) -> np.ndarray:
    """Tighten a YOLO-box mask to grass pixels: ``box ∩ (union of SAM masks)``.

    The repair only ever *removes* box pixels that fall on non-grass (it never adds), so a
    loose box over bare ground / shadow stops contributing false-positive tile area.
    """
    box = np.asarray(box_mask, dtype=bool)
    return box & union_masks(sam_masks, box.shape)


def tile_coverage(mask: np.ndarray, tile_px: int) -> list[dict]:
    """Per-tile grass-coverage fraction over the mask grid (mirrors data_variants tiling geometry)."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    rows, cols = -(-H // tile_px), -(-W // tile_px)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            y1, x1 = min(H, y0 + tile_px), min(W, x0 + tile_px)
            out.append({"r": r, "c": c, "coverage": float(mask[y0:y1, x0:x1].mean())})
    return out


def tiles_from_mask(mask: np.ndarray, tile_px: int, cover_thresh: float = 0.30) -> list[dict]:
    """Collapse a pixel mask to per-tile ``is_cog`` (>= ``cover_thresh`` area — the label rule)."""
    return [{**t, "is_cog": t["coverage"] >= cover_thresh} for t in tile_coverage(mask, tile_px)]


def coverage_mae(true_mask: np.ndarray, pred_mask: np.ndarray, tile_px: int) -> float:
    """Mean absolute error between per-tile true and predicted coverage fractions."""
    t = tile_coverage(true_mask, tile_px)
    p = tile_coverage(pred_mask, tile_px)
    return float(np.mean([abs(a["coverage"] - b["coverage"]) for a, b in zip(t, p)]))


def pixel_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks (1.0 when both empty)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter / union) if union else 1.0


def repair_tile_labels(box_mask: np.ndarray, sam_masks, tile_px: int, frame_stem: str,
                       cover_thresh: float = 0.30) -> list[dict]:
    """Per-tile original-vs-repaired labels for one frame (box-only vs box∩mask).

    Returns one record per tile: ``{filename, orig_is_cog, repaired_is_cog}``. Feed the flips
    (orig positive -> repaired negative) into ``data_variants.build_clean_variant`` to
    materialize the repaired-label variant.
    """
    orig = tiles_from_mask(box_mask, tile_px, cover_thresh)
    refined = box_intersect_mask(box_mask, sam_masks)
    rep = tiles_from_mask(refined, tile_px, cover_thresh)
    out = []
    for o, r in zip(orig, rep):
        out.append({"filename": f"{frame_stem}_r{o['r']}_c{o['c']}.jpg",
                    "orig_is_cog": o["is_cog"], "repaired_is_cog": r["is_cog"]})
    return out


def relabel_map_from_repairs(repairs) -> dict:
    """Build a ``{filename: 'not_cogongrass'}`` map for tiles the repair flips positive->negative."""
    out = {}
    for rec in repairs:
        if rec["orig_is_cog"] and not rec["repaired_is_cog"]:
            out[rec["filename"]] = "not_cogongrass"
    return out


# ---------------------------------------------------------------------------
# Failure-tolerant model fit gate  (KTD6 — load failure records a row, never aborts)
# ---------------------------------------------------------------------------
def load_sam(model_name: str = DEFAULT_SAM_MODEL):
    """Load an ultralytics SAM model (``sam_explore.py`` pattern). Raises on failure."""
    from ultralytics import SAM
    return SAM(model_name)


def sam_masks_for_image(sam, image_path) -> list:
    """Run SAM 'segment everything' on one frame -> list of boolean HxW masks (empty if none)."""
    res = sam(str(image_path), verbose=False)
    masks = res[0].masks
    if masks is None:
        return []
    return [np.asarray(m, dtype=bool) for m in masks.data.cpu().numpy()]


def run_sam_smoke(*, model_name: str = DEFAULT_SAM_MODEL, results_dir=C.RESULTS_DIR,
                  loader=load_sam) -> C.ResultRow:
    """Fit gate: try to load SAM, record a ResultRow either way, NEVER abort (KTD6).

    ``loader`` is injectable so the failure-tolerance is unit-testable without the model. A
    success row means the full repair/segmentation pass can run on this machine.
    """
    identity = dict(model="sam_repair", variant="reference", tuning_mode="frozen",
                    head="none", adaptation="none", eval_setting=C.EVAL_CROSS,
                    seed=C.DEFAULT_SEED, extra=f"sam={model_name}")
    try:
        loader(model_name)
        row = C.ResultRow(**identity, status="ok")
    except Exception as e:  # noqa: BLE001 — record + continue, never block the program (KTD6)
        is_oom = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
        row = C.ResultRow(**identity, status="oom" if is_oom else "failed",
                          error=f"{type(e).__name__}: {e}"[:500])
    C.write_result_atomic(row, results_dir)
    return row
