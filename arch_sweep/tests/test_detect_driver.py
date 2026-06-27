"""U2 (finish) tests: per-frame detection aggregation, truth alignment, fit-gate-failure return."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
from detect import run_detect as RD  # noqa: E402


def test_detect_frames_aggregates_and_aligns_truth():
    # 8x4 frame, 4px tiles -> tiles (r0,c0) and (r0,c1); a box over the left tile only
    specs = [("DJI_20260422_0001_r0", 8, 4)]
    detect_fn = lambda stem, W, H: [(0, 0, 4, 4, 0.9)]            # left tile detected
    truth = {("DJI_20260422_0001_r0", 0, 0): 1, ("DJI_20260422_0001_r0", 0, 1): 0}
    truth_fn = lambda stem, r, c: truth.get((stem, r, c))
    tiles, y_true, paths = RD.detect_frames(specs, detect_fn=detect_fn, truth_fn=truth_fn,
                                            tile_px=4)
    assert y_true == [1, 0]
    assert tiles[0]["is_cog"] and tiles[0]["score"] == 0.9 and not tiles[1]["is_cog"]
    assert paths == ["DJI_20260422_0001_r0_r0_c0.jpg", "DJI_20260422_0001_r0_r0_c1.jpg"]


def test_detect_frames_skips_tiles_absent_from_dataset():
    specs = [("f_r0", 8, 4)]
    detect_fn = lambda stem, W, H: [(0, 0, 8, 4, 0.8)]
    truth_fn = lambda stem, r, c: 1 if c == 0 else None          # tile (r0,c1) ExG-dropped
    tiles, y_true, paths = RD.detect_frames(specs, detect_fn=detect_fn, truth_fn=truth_fn,
                                            tile_px=4)
    assert y_true == [1] and len(tiles) == 1                     # the absent tile is skipped


def test_zero_detections_yield_all_negative(tmp_path):
    specs = [("f_r0", 8, 4)]
    detect_fn = lambda stem, W, H: []                            # detector finds nothing
    truth_fn = lambda stem, r, c: 1 if c == 0 else 0
    row = RD.run_detection_pass(specs, detect_fn=detect_fn, truth_fn=truth_fn, tile_px=4,
                                conf=0.3, results_dir=tmp_path)
    assert row.status == "ok" and row.eval_setting == C.EVAL_FEWSHOT
    assert row.recall_cogongrass == 0.0                          # nothing predicted positive


def test_run_detection_pass_scores_few_shot_row(tmp_path):
    specs = [("DJI_20260422_0001_r0", 8, 4)]
    detect_fn = lambda stem, W, H: [(0, 0, 4, 4, 0.9)]
    truth_fn = lambda stem, r, c: 1 if c == 0 else 0
    row = RD.run_detection_pass(specs, detect_fn=detect_fn, truth_fn=truth_fn, detector="trex2",
                                n_exemplars=3, conf=0.5, tile_px=4, results_dir=tmp_path,
                                write_scores=True)
    assert row.status == "ok" and row.model == "trex2" and row.budget == 3
    assert row.balanced_accuracy == 1.0 and row.f2_sweep
    assert len(list(tmp_path.glob("*.scores.jsonl"))) == 1
    assert len(C.read_all_results(tmp_path)) == 1                # sidecar excluded from merge


def test_fit_gate_failure_returns_recorded_row_without_pass(tmp_path):
    def broken(name):
        raise RuntimeError("CUDA out of memory loading detector")

    row = RD.run_detect(exemplar_boxes=[(0, 0, 4, 4)], root=tmp_path, results_dir=tmp_path,
                        loader=broken)
    assert row.status == "oom"                                   # the gate's row, not a crash
    assert list(tmp_path.glob("*.jsonl"))
