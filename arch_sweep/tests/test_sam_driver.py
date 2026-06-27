"""U1 (finish) tests: per-frame repair accumulation, flip edges, fit-gate-failure early return."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import data_variants as DV  # noqa: E402
from sam import run_repair as RR  # noqa: E402


def _box_full(stem, W, H):
    return np.ones((H, W), dtype=bool)            # the whole frame is inside a box


def test_repair_frames_accumulates_flips_across_frames():
    # 2 frames, 4x4 px, 4px tiles -> one tile each; frame A has no grass (flips), B is all grass
    specs = [("DJI_20260422_0001_r0", 4, 4), ("DJI_20260422_0002_r0", 4, 4)]
    masks = {"DJI_20260422_0001_r0": [np.zeros((4, 4), bool)],     # no grass -> flip
             "DJI_20260422_0002_r0": [np.ones((4, 4), bool)]}      # all grass -> keep
    relabel, mae = RR.repair_frames(specs, mask_fn=lambda s: masks[s], box_fn=_box_full,
                                    tile_px=4)
    assert relabel == {"DJI_20260422_0001_r0_r0_c0.jpg": "not_cogongrass"}   # only frame A flips
    assert mae >= 0.0


def test_repair_frames_empty_intersection_flips_all_box_positives():
    specs = [("f_r0", 8, 4)]                       # 8x4 -> two 4px tiles, both box-positive
    relabel, _mae = RR.repair_frames(specs, mask_fn=lambda s: [np.zeros((4, 8), bool)],
                                     box_fn=_box_full, tile_px=4)
    assert len(relabel) == 2 and all(v == "not_cogongrass" for v in relabel.values())


def test_repair_frames_full_grass_flips_nothing():
    specs = [("f_r0", 4, 4)]
    relabel, _mae = RR.repair_frames(specs, mask_fn=lambda s: [np.ones((4, 4), bool)],
                                     box_fn=_box_full, tile_px=4)
    assert relabel == {}


def test_fit_gate_failure_returns_early_without_variant(tmp_path):
    def broken(name):
        raise RuntimeError("CUDA out of memory loading SAM")

    out, stats = RR.run_repair(model_name="sam2_l.pt", root=tmp_path,
                               results_dir=tmp_path, loader=broken)
    assert out is None and stats["status"] == "oom"
    assert list(tmp_path.glob("*.jsonl"))                       # the gate recorded a row
    assert not (tmp_path / "tiles_dataset_0422clean").exists()  # no partial variant written


def test_relabel_map_drives_build_clean_variant(tmp_path):
    # integration: an accumulated relabel map materializes an enumerable cleaned variant
    base = tmp_path / "tiles_dataset"
    (base / "cogongrass").mkdir(parents=True)
    (base / "not_cogongrass").mkdir(parents=True)
    (base / "cogongrass" / "DJI_20260422_0001_r0_c0.jpg").write_bytes(b"x")
    (base / "not_cogongrass" / "DJI_20260422_0002_r1_c0.jpg").write_bytes(b"x")
    (base / "not_cogongrass" / "DJI_20260422_0002_r1_c1.jpg").write_bytes(b"x")  # stays negative
    relabel = {"DJI_20260422_0002_r1_c0.jpg": "cogongrass"}      # a flip the repair found
    out, n = DV.build_clean_variant(relabel, root=tmp_path)
    samples, _classes, _cog = C.enumerate_tiles(str(out))
    counts = DV.class_counts(samples)
    assert n == 1 and counts["cogongrass"] == 2 and counts["not_cogongrass"] == 1
