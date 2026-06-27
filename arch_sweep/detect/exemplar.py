"""Exemplar-prompted detection -> per-tile predictions on 0422 (U9).

Pipeline: prompt a visual-exemplar detector (T-Rex2, with T-Rex-Omni negative exemplars for
look-alike grasses) with a few 0422 cogongrass boxes, detect across the 0422 frames, then
**rasterize the detection boxes onto the 512px tile grid** so the output is per-tile
predictions scored on the *identical* 0422 protocol as every other cell. Sweep the detector
confidence for the F2 curve.

The rasterization (box→tile by exact rectangle-overlap area) and the confidence sweep are pure
and CPU-tested. ``run_detect_smoke`` is the failure-tolerant fit gate for the model load (KTD6).
The diffuse-texture / oblique-view risk is assessed at run time from the resulting F2 curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402

DEFAULT_DETECTOR = "trex2"
# Detector-confidence grid for the F2 curve (mirrors common.SWEEP_THRESHOLDS in spirit).
CONF_GRID = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]


def tile_records_from_boxes(boxes, W: int, H: int, tile_px: int,
                            cover_thresh: float = 0.30) -> list[dict]:
    """Rasterize detection boxes onto the tile grid via exact rectangle-overlap area.

    ``boxes`` are ``(x0, y0, x1, y1, conf)`` in pixel coords. A tile is a candidate positive
    when some box covers ``>= cover_thresh`` of its area (the same area rule as the labels); its
    ``score`` is the max confidence among such boxes (0 if none) — the per-tile detection score
    the F2 sweep and AUROC consume. Exact intersection (not pixel painting) avoids aliasing.
    """
    rows, cols = -(-H // tile_px), -(-W // tile_px)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            y1, x1 = min(H, y0 + tile_px), min(W, x0 + tile_px)
            area = max(1, (y1 - y0) * (x1 - x0))
            best = 0.0
            covered = False
            for bx0, by0, bx1, by1, conf in boxes:
                inter = max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0))
                if inter / area >= cover_thresh:
                    covered = True
                    best = max(best, float(conf))
            out.append({"r": r, "c": c, "is_cog": covered, "score": best})
    return out


def confidence_sweep(tile_scores, y_true_cog, confs=CONF_GRID) -> list[dict]:
    """recall / precision / F1 / F2 / FN at each detector confidence (the F2 curve, U9).

    A thin wrapper over ``common.f2_sweep`` on the per-tile detection scores — same machinery
    every other cell uses, so the detector's operating curve is directly comparable.
    """
    return C.f2_sweep(y_true_cog, tile_scores, thresholds=confs)


def run_detection(tiles, y_true_cog, paths, *, detector=DEFAULT_DETECTOR, n_exemplars=3,
                  conf=0.30, eval_setting=C.EVAL_FEWSHOT, results_dir=C.RESULTS_DIR,
                  write_scores=False) -> C.ResultRow:
    """Score exemplar-prompted detection on the standard 0422 tile protocol -> a ResultRow.

    ``tiles`` is the per-tile records (``tile_records_from_boxes`` over every 0422 frame,
    concatenated), aligned with ``y_true_cog`` and ``paths``. The headline is balanced accuracy
    at the operating ``conf``; AUROC/AP/F2-sweep come from the per-tile scores. Tagged
    ``few_shot`` by default (exemplars are 0422 labels; KTD5) with the exemplar count as budget.
    """
    scores = [float(t["score"]) for t in tiles]
    y_pred = [1 if s >= conf else 0 for s in scores]
    rec = C.per_class_recall(y_true_cog, y_pred)
    both = len(set(y_true_cog)) == 2
    row = C.ResultRow(
        model=detector, variant="reference", tuning_mode="frozen", head="detector",
        adaptation="exemplar", eval_setting=eval_setting, seed=C.DEFAULT_SEED,
        extra=f"detect={detector},n_ex={n_exemplars}", status="ok",
        balanced_accuracy=C.balanced_accuracy(y_true_cog, y_pred),
        recall_cogongrass=rec[C.COG_CLASS], recall_not_cogongrass=rec["not_cogongrass"],
        auroc=C.auroc(y_true_cog, scores) if both else None,
        average_precision=C.average_precision(y_true_cog, scores) if both else None,
        threshold=conf, f2_sweep=confidence_sweep(scores, y_true_cog),
        n_test=len(y_true_cog), n_cog_test=int(sum(y_true_cog)), budget=n_exemplars)
    C.write_result_atomic(row, results_dir)
    if write_scores:
        recs = C.build_score_records(paths, y_true_cog, scores)
        C.write_scores_atomic(row.identity(), recs, results_dir)
    return row


# ---------------------------------------------------------------------------
# Failure-tolerant detector fit gate  (KTD6)
# ---------------------------------------------------------------------------
def load_detector(name: str = DEFAULT_DETECTOR):
    """Load the visual-exemplar detector (T-Rex2 / T-Rex-Omni). Raises on failure.

    Thin lazy delegate to ``detect.trex.load_trex`` so the heavy SDK import only happens on a
    real run; callers wrap with ``run_detect_smoke`` so a missing dep is recorded, not crashed.
    """
    from detect import trex
    return trex.load_trex(name)


def run_detect_smoke(*, detector: str = DEFAULT_DETECTOR, results_dir=C.RESULTS_DIR,
                     loader=load_detector) -> C.ResultRow:
    """Fit gate: try to load the detector, record a ResultRow either way, NEVER abort (KTD6)."""
    identity = dict(model=detector, variant="reference", tuning_mode="frozen", head="detector",
                    adaptation="exemplar", eval_setting=C.EVAL_FEWSHOT, seed=C.DEFAULT_SEED,
                    extra=f"detect={detector},smoke")
    try:
        loader(detector)
        row = C.ResultRow(**identity, status="ok")
    except Exception as e:  # noqa: BLE001 — record + continue, never block the program (KTD6)
        is_oom = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
        row = C.ResultRow(**identity, status="oom" if is_oom else "failed",
                          error=f"{type(e).__name__}: {e}"[:500])
    C.write_result_atomic(row, results_dir)
    return row
