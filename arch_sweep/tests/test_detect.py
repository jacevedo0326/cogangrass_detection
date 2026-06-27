"""U9 tests: rasterize boxes->tiles, F2 sweep, standard-protocol scoring, failure-tolerant load."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
from detect import exemplar as DT  # noqa: E402


def test_rasterize_box_lands_on_correct_tiles():
    # 8x8 frame, 4px tiles -> 2x2 grid; a box over the top-left 4x4 tile only
    boxes = [(0, 0, 4, 4, 0.9)]
    tiles = DT.tile_records_from_boxes(boxes, W=8, H=8, tile_px=4)
    by_rc = {(t["r"], t["c"]): t for t in tiles}
    assert by_rc[(0, 0)]["is_cog"] and by_rc[(0, 0)]["score"] == 0.9
    assert not any(by_rc[rc]["is_cog"] for rc in [(0, 1), (1, 0), (1, 1)])


def test_rasterize_respects_area_threshold():
    # a box covering only a 1x4 strip (25%) of the 4x4 tile -> below the 30% rule -> negative
    boxes = [(0, 0, 4, 1, 0.8)]
    tiles = DT.tile_records_from_boxes(boxes, W=4, H=4, tile_px=4, cover_thresh=0.30)
    assert not tiles[0]["is_cog"]
    # lower the rule and it qualifies, carrying the box confidence
    tiles2 = DT.tile_records_from_boxes(boxes, W=4, H=4, tile_px=4, cover_thresh=0.20)
    assert tiles2[0]["is_cog"] and tiles2[0]["score"] == 0.8


def test_confidence_sweep_yields_f2_table():
    # one true-positive tile (score 0.6) and one true-negative tile (score 0.2)
    scores = [0.6, 0.2]
    y_true = [1, 0]
    rows = DT.confidence_sweep(scores, y_true, confs=[0.5, 0.1])
    assert {r["thr"] for r in rows} == {0.5, 0.1}
    hi = next(r for r in rows if r["thr"] == 0.5)
    lo = next(r for r in rows if r["thr"] == 0.1)
    assert hi["fn"] == 0 and hi["recall"] == pytest.approx(1.0)   # 0.6 >= 0.5 catches the positive
    assert lo["recall"] == pytest.approx(1.0)                     # lower conf still catches it


def test_run_detection_scores_on_standard_protocol(tmp_path):
    tiles = [{"r": 0, "c": 0, "is_cog": True, "score": 0.8},
             {"r": 0, "c": 1, "is_cog": False, "score": 0.1}]
    y_true = [1, 0]
    paths = ["tiles/cogongrass/DJI_20260422_0001_r0_c0.jpg",
             "tiles/not_cogongrass/DJI_20260422_0001_r0_c1.jpg"]
    row = DT.run_detection(tiles, y_true, paths, conf=0.5, results_dir=tmp_path, write_scores=True)
    assert row.status == "ok" and row.model == DT.DEFAULT_DETECTOR
    assert row.balanced_accuracy == 1.0 and row.recall_cogongrass == 1.0
    assert row.auroc is not None and row.eval_setting == C.EVAL_FEWSHOT   # exemplar = few_shot
    assert row.budget == 3 and row.f2_sweep
    # scores sidecar lands and does not pollute the result merge
    assert len(list(tmp_path.glob("*.scores.jsonl"))) == 1
    assert len(C.read_all_results(tmp_path)) == 1


def test_detector_load_failure_records_row_and_does_not_abort(tmp_path):
    def broken(name):
        raise RuntimeError("CUDA out of memory loading detector")

    row = DT.run_detect_smoke(results_dir=tmp_path, loader=broken)
    assert row.status == "oom"
    assert list(tmp_path.glob("*.jsonl"))                 # row written, program continues


def test_default_loader_is_not_silently_a_noop():
    # the real loader must be wired at run time, not silently succeed
    import pytest
    with pytest.raises(NotImplementedError):
        DT.load_detector()
