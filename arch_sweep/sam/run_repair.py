"""SAM per-frame label-repair driver (Stage-2-finish U1).

The run-time loop the Stage-2 plan deferred (origin U8): fit-gate SAM, then for every 0422
frame intersect its YOLO boxes with SAM masks, collapse to the 512px tile grid, accumulate the
positive->negative flips, and materialize the cleaned eval variant ``tiles_dataset_0422clean/``.

Thin glue over the already-tested geometry in ``sam/repair.py`` — the accumulation loop is
injectable (``mask_fn`` / ``box_fn``) so it is unit-testable without SAM; only the real pass
needs the model. Failure-tolerant (KTD2): the fit gate records a row and the driver returns
cleanly rather than crashing a long run.

Run:
    python arch_sweep/sam/run_repair.py --model sam2_l.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import data_variants as DV  # noqa: E402
from sam import repair as SR  # noqa: E402

REPO_ROOT = DV.REPO_ROOT


def repair_frames(frame_specs, *, mask_fn, box_fn, tile_px: int = 512,
                  cover_thresh: float = 0.30) -> tuple[dict, float]:
    """Accumulate per-frame box∩mask repairs into a (relabel_map, mean coverage-MAE).

    ``frame_specs`` is an iterable of ``(frame_stem, W, H)``. ``box_fn(stem, W, H)`` returns the
    YOLO box mask; ``mask_fn(stem)`` returns the SAM masks. Pure (no SAM, no disk) so the whole
    accumulation is testable on injected fixtures. The relabel map flips tiles that were
    positive-by-box but contain no grass after intersection.
    """
    all_repairs, maes = [], []
    for stem, W, H in frame_specs:
        box = box_fn(stem, W, H)
        sam_masks = mask_fn(stem)
        all_repairs.extend(SR.repair_tile_labels(box, sam_masks, tile_px, stem, cover_thresh))
        refined = SR.box_intersect_mask(box, sam_masks)
        maes.append(SR.coverage_mae(box, refined, tile_px))
    relabel = SR.relabel_map_from_repairs(all_repairs)
    return relabel, (sum(maes) / len(maes) if maes else 0.0)


def _frame_specs(root: Path, date: str = C.TEST_DATE):
    """(stem, W, H) for every frame in ``drone_dataset/images`` belonging to ``date``."""
    from PIL import Image

    img_dir = Path(root) / "drone_dataset" / "images"
    specs = []
    for img in sorted(img_dir.glob("*.jpg")):
        if C.date_of(img.stem) != date:
            continue
        with Image.open(img) as im:
            W, H = im.size
        specs.append((img.stem, W, H))
    return specs


def run_repair(*, model_name: str = SR.DEFAULT_SAM_MODEL, root: Path | str = REPO_ROOT,
               tile_px: int = 512, cover_thresh: float = 0.30, results_dir=C.RESULTS_DIR,
               loader=SR.load_sam) -> tuple[Path | None, dict]:
    """Fit-gate SAM, repair every 0422 frame, materialize ``tiles_dataset_0422clean/``.

    Returns ``(variant_dir_or_None, stats)``. On a fit-gate failure the gate records a row and
    this returns ``(None, {...})`` without writing a variant (KTD2). ``loader`` is injectable so
    the failure path is testable without SAM.
    """
    gate = SR.run_sam_smoke(model_name=model_name, results_dir=results_dir, loader=loader)
    if gate.status != "ok":
        print(f"[sam-repair] fit gate {gate.status}: {gate.error} — skipping repair pass")
        return None, {"status": gate.status, "flips": 0}

    sam = loader(model_name)
    root = Path(root)
    img_dir = root / "drone_dataset" / "images"
    lbl_dir = root / "drone_dataset" / "labels"
    specs = _frame_specs(root)

    def box_fn(stem, W, H):
        return DV.boxes_mask(lbl_dir / f"{stem}.txt", W, H)

    def mask_fn(stem):
        return SR.sam_masks_for_image(sam, img_dir / f"{stem}.jpg")

    relabel, mae = repair_frames(specs, mask_fn=mask_fn, box_fn=box_fn,
                                 tile_px=tile_px, cover_thresh=cover_thresh)
    out, n_flipped = DV.build_clean_variant(relabel, root=root)
    stats = {"status": "ok", "frames": len(specs), "flips": n_flipped, "coverage_mae": mae}
    print(f"[sam-repair] {len(specs)} frames -> {n_flipped} tiles relabeled "
          f"(mean coverage-MAE {mae:.4f}) -> {out}")
    return out, stats


def main():
    ap = argparse.ArgumentParser(description="SAM per-frame label-repair driver (U1)")
    ap.add_argument("--model", default=SR.DEFAULT_SAM_MODEL, help="ultralytics SAM checkpoint")
    ap.add_argument("--tile-px", type=int, default=512)
    ap.add_argument("--cover-thresh", type=float, default=0.30)
    args = ap.parse_args()
    print(f"== SAM label-repair  model={args.model} ==", flush=True)
    run_repair(model_name=args.model, tile_px=args.tile_px, cover_thresh=args.cover_thresh)


if __name__ == "__main__":
    main()
