"""Exemplar-detection driver (Stage-2-finish U2).

The run-time loop deferred from Stage-2 U9: fit-gate the detector, prompt with a few 0422
cogongrass exemplars, detect across the 0422 frames, rasterize each frame's boxes onto the
512px tile grid, and score the concatenated per-tile predictions through the standard protocol
(``detect.exemplar.run_detection`` -> a ``few_shot`` row, KTD5).

The aggregation loop is injectable (``detect_fn`` / ``truth_fn``) so it is CPU-tested without
the detector; only the real pass needs T-Rex2. Failure-tolerant: the fit gate records a row and
the driver returns cleanly (KTD2).

Run:
    DDS_API_TOKEN=... python arch_sweep/detect/run_detect.py --exemplars exemplars.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import data_variants as DV  # noqa: E402
from detect import exemplar as DT  # noqa: E402
from detect import trex as TREX  # noqa: E402

REPO_ROOT = DV.REPO_ROOT


def detect_frames(frame_specs, *, detect_fn, truth_fn, tile_px: int = 512,
                  cover_thresh: float = 0.30):
    """Accumulate per-frame detections into aligned ``(tiles, y_true, paths)``.

    ``frame_specs`` is ``(stem, W, H)``. ``detect_fn(stem, W, H)`` returns detection boxes;
    ``truth_fn(stem, r, c)`` returns 0/1, or ``None`` for a tile absent from the dataset (e.g.
    ExG-dropped) which is skipped so prediction and ground truth stay aligned. Pure — testable
    on injected fixtures.
    """
    tiles, y_true, paths = [], [], []
    for stem, W, H in frame_specs:
        boxes = detect_fn(stem, W, H)
        for t in DT.tile_records_from_boxes(boxes, W, H, tile_px, cover_thresh):
            truth = truth_fn(stem, t["r"], t["c"])
            if truth is None:
                continue
            tiles.append(t)
            y_true.append(int(truth))
            paths.append(f"{stem}_r{t['r']}_c{t['c']}.jpg")
    return tiles, y_true, paths


def run_detection_pass(frame_specs, *, detect_fn, truth_fn, detector="trex2", n_exemplars=3,
                       conf=0.30, tile_px=512, results_dir=C.RESULTS_DIR,
                       write_scores=True) -> C.ResultRow:
    """Accumulate detections over frames and score them as one ``few_shot`` cell."""
    tiles, y_true, paths = detect_frames(frame_specs, detect_fn=detect_fn, truth_fn=truth_fn,
                                         tile_px=tile_px)
    return DT.run_detection(tiles, y_true, paths, detector=detector, n_exemplars=n_exemplars,
                            conf=conf, results_dir=results_dir, write_scores=write_scores)


def _frame_specs(root: Path, date: str = C.TEST_DATE):
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


def _truth_fn(root: Path, variant: str = "reference"):
    """Per-tile ground truth from the dataset class folders (1/0/None-if-absent)."""
    base = DV.VARIANTS_BY_NAME[variant].path(root)

    def truth(stem, r, c):
        name = f"{stem}_r{r}_c{c}.jpg"
        if (base / "cogongrass" / name).exists():
            return 1
        if (base / "not_cogongrass" / name).exists():
            return 0
        return None

    return truth


def run_detect(*, exemplar_boxes, negative_boxes=None, detector="trex2", root=REPO_ROOT,
               conf=0.30, results_dir=C.RESULTS_DIR, loader=DT.load_detector) -> C.ResultRow:
    """Fit-gate the detector, run exemplar detection over the 0422 frames, score a few_shot cell.

    ``loader`` is injectable so the fit-gate-failure path is testable without the detector. On a
    gate failure the recorded row is returned and no detection pass runs (KTD2).
    """
    gate = DT.run_detect_smoke(detector=detector, results_dir=results_dir, loader=loader)
    if gate.status != "ok":
        print(f"[detect] fit gate {gate.status}: {gate.error} — skipping detection pass")
        return gate
    client = loader(detector)
    root = Path(root)
    img_dir = root / "drone_dataset" / "images"
    specs = _frame_specs(root)

    def detect_fn(stem, W, H):
        return TREX.detect_with_exemplars(client, img_dir / f"{stem}.jpg", exemplar_boxes,
                                          negative_boxes=negative_boxes, conf=conf)

    row = run_detection_pass(specs, detect_fn=detect_fn, truth_fn=_truth_fn(root),
                             detector=detector, n_exemplars=len(exemplar_boxes), conf=conf,
                             results_dir=results_dir)
    if row.status == "ok":
        print(f"[detect] {len(specs)} frames -> bacc {row.balanced_accuracy:.3f} "
              f"(recall cog {row.recall_cogongrass:.3f}, AUROC "
              f"{row.auroc if row.auroc is None else round(row.auroc, 3)})")
    return row


def main():
    import json

    ap = argparse.ArgumentParser(description="Exemplar-detection driver (U2)")
    ap.add_argument("--detector", default="trex2")
    ap.add_argument("--exemplars", required=True,
                    help="JSON file: {\"positive\": [[x0,y0,x1,y1],...], \"negative\": [...]}")
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()
    spec = json.loads(Path(args.exemplars).read_text())
    print(f"== exemplar detection  detector={args.detector} "
          f"({len(spec.get('positive', []))} +, {len(spec.get('negative', []))} -) ==", flush=True)
    run_detect(exemplar_boxes=spec.get("positive", []), negative_boxes=spec.get("negative"),
               detector=args.detector, conf=args.conf)


if __name__ == "__main__":
    main()
