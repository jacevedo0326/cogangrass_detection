"""U8 tests: box∩mask repair, collapse-to-tile, coverage-MAE, failure-tolerant SAM load."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
from sam import repair as SR  # noqa: E402


def test_box_intersect_mask_tightens_to_grass_pixels():
    box = np.zeros((8, 8), dtype=bool)
    box[0:8, 0:4] = True                      # a YOLO box over the left half
    grass = np.zeros((8, 8), dtype=bool)
    grass[0:4, 0:8] = True                    # SAM finds grass only in the top half
    refined = SR.box_intersect_mask(box, [grass])
    # only the box∩grass quadrant survives (top-left)
    assert refined[0:4, 0:4].all() and not refined[4:8, 0:4].any()
    assert int(refined.sum()) == 16


def test_box_intersect_mask_no_sam_masks_is_empty():
    box = np.ones((4, 4), dtype=bool)
    assert not SR.box_intersect_mask(box, []).any()


def test_repair_flips_box_positive_tile_to_negative():
    # a 4x4 tile fully inside a box but with NO grass pixels -> repaired negative
    box = np.ones((4, 4), dtype=bool)
    repairs = SR.repair_tile_labels(box, [np.zeros((4, 4), dtype=bool)], tile_px=4,
                                    frame_stem="DJI_20260422_0001_r0", cover_thresh=0.30)
    assert len(repairs) == 1
    assert repairs[0]["orig_is_cog"] is True and repairs[0]["repaired_is_cog"] is False
    relabel = SR.relabel_map_from_repairs(repairs)
    assert relabel == {"DJI_20260422_0001_r0_r0_c0.jpg": "not_cogongrass"}


def test_repair_keeps_genuine_grass_tile_positive():
    box = np.ones((4, 4), dtype=bool)
    grass = np.ones((4, 4), dtype=bool)       # box fully covers real grass
    repairs = SR.repair_tile_labels(box, [grass], tile_px=4, frame_stem="f_r0")
    assert repairs[0]["repaired_is_cog"] is True
    assert SR.relabel_map_from_repairs(repairs) == {}   # nothing to flip


def test_collapsed_tiles_score_on_standard_protocol():
    # collapse a predicted mask to per-tile labels and score with the standard metrics
    true_mask = np.zeros((4, 8), dtype=bool)
    true_mask[:, 0:4] = True                  # left tile is grass, right tile is not
    pred_mask = true_mask.copy()
    y_true = [int(t["is_cog"]) for t in SR.tiles_from_mask(true_mask, tile_px=4)]
    y_pred = [int(t["is_cog"]) for t in SR.tiles_from_mask(pred_mask, tile_px=4)]
    assert y_true == [1, 0]
    assert C.balanced_accuracy(y_true, y_pred) == 1.0


def test_coverage_mae_and_pixel_iou():
    true_mask = np.zeros((4, 4), dtype=bool)
    true_mask[:2, :] = True                   # 50% coverage of the single tile
    pred_mask = np.zeros((4, 4), dtype=bool)
    pred_mask[:1, :] = True                   # 25% coverage
    assert SR.coverage_mae(true_mask, pred_mask, tile_px=4) == 0.25
    assert SR.pixel_iou(true_mask, pred_mask) == 0.5
    assert SR.pixel_iou(true_mask, true_mask) == 1.0


def test_sam_load_failure_records_row_and_does_not_abort(tmp_path):
    def broken_loader(name):
        raise RuntimeError("CUDA out of memory loading SAM")

    row = SR.run_sam_smoke(model_name="sam2_l.pt", results_dir=tmp_path, loader=broken_loader)
    assert row.status == "oom"                          # classified, not raised
    assert row.model == "sam_repair" and "sam=sam2_l.pt" in row.extra
    assert list(tmp_path.glob("*.jsonl"))               # row written (honest accounting, KTD8)


def test_sam_load_success_records_ok_row(tmp_path):
    row = SR.run_sam_smoke(results_dir=tmp_path, loader=lambda name: object())
    assert row.status == "ok"
    assert C.read_all_results(tmp_path)[0].job_id == row.job_id
