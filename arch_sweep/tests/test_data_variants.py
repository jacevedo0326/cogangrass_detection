"""U2 tests: variant naming + manifest (correct, idempotent) + ExG / label-area toggles.

No test for the heavy re-tiling wrapper itself (plan U2) — we assert the pure per-tile
rule and the manifest, which is where drift would silently corrupt results.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_variants as DV  # noqa: E402


# --- naming -------------------------------------------------------------------
def test_reference_variants_point_at_existing_dirs():
    assert DV.VARIANTS_BY_NAME["reference"].dir_name() == "tiles_dataset"
    assert DV.VARIANTS_BY_NAME["reference_clahe"].dir_name() == "tiles_dataset_clahe"


def test_canonical_names_encode_the_axes():
    assert DV.VARIANTS_BY_NAME["tile224"].dir_name() == "tiles_dataset_t224s224p4096"
    assert DV.VARIANTS_BY_NAME["tile224_clahe"].dir_name().endswith("_clahe")
    assert "_noexg" in DV.VARIANTS_BY_NAME["noexg"].dir_name()
    # distinct specs never collide on a dir name
    names = [v.dir_name() for v in DV.STANDARD_VARIANTS]
    assert len(names) == len(set(names))


# --- ExG green-filter toggle --------------------------------------------------
def _sky_over_grass(tile_px=4):
    """8x4 frame: top half is grey 'sky' (low ExG), bottom half is green 'grass'."""
    arr = np.zeros((8, 4, 3), dtype=np.uint8)
    arr[:4] = [120, 120, 120]      # grey -> ExG ~ 0 (dropped by the filter)
    arr[4:] = [20, 200, 20]        # green -> ExG high (kept)
    return arr


def test_exg_filter_drops_sky_when_on_keeps_all_when_off():
    arr = _sky_over_grass()
    mask = np.zeros(arr.shape[:2], dtype=bool)
    on = DV.tile_records(arr, mask, tile_px=4, exg_filter=True)
    off = DV.tile_records(arr, mask, tile_px=4, exg_filter=False)
    # filter OFF keeps every tile; filter ON drops at least the grey 'sky' tile(s)
    assert all(t["kept"] for t in off)
    assert sum(t["kept"] for t in on) < len(on)
    # the dropped tiles are exactly the low-green (top) ones
    dropped = [t for t in on if not t["kept"]]
    assert dropped and all(t["r"] == 0 for t in dropped)
    # toggling the filter changes the kept tile set (the ablation actually does something)
    assert {(t["r"], t["c"]) for t in on if t["kept"]} != {(t["r"], t["c"]) for t in off}


# --- label-area threshold -----------------------------------------------------
def test_label_area_threshold_changes_positive_count():
    arr = np.full((4, 4, 3), [20, 200, 20], dtype=np.uint8)   # all green so nothing is ExG-dropped
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :] = True   # box covers top 2 of 4 rows of the single 4x4 tile -> 50% area
    strict = DV.tile_records(arr, mask, tile_px=4, cover_thresh=0.6)   # 50% < 60% -> negative
    loose = DV.tile_records(arr, mask, tile_px=4, cover_thresh=0.4)    # 50% >= 40% -> positive
    assert sum(t["is_cog"] for t in strict) == 0
    assert sum(t["is_cog"] for t in loose) == 1


# --- manifest correctness + idempotency ---------------------------------------
def _make_variant_dir(root: Path, dir_name: str, n_cog: int, n_neg: int):
    for cls, n in (("cogongrass", n_cog), ("not_cogongrass", n_neg)):
        d = root / dir_name / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"DJI_20260606_0001_r0_c{i}.jpg").write_bytes(b"x")


def test_manifest_counts_and_existence(tmp_path):
    specs = [
        DV.VariantSpec("present", dir_override="tiles_dataset_present"),
        DV.VariantSpec("absent", dir_override="tiles_dataset_absent"),
    ]
    _make_variant_dir(tmp_path, "tiles_dataset_present", n_cog=3, n_neg=5)
    m = DV.build_manifest(specs, root=tmp_path, manifest_path=tmp_path / "manifest.json")
    pres = m["variants"]["present"]
    assert pres["exists"] and pres["counts"] == {"cogongrass": 3, "not_cogongrass": 5, "total": 8}
    assert m["variants"]["absent"]["exists"] is False
    assert m["variants"]["absent"]["counts"] is None
    assert DV.available_variants(tmp_path / "manifest.json") == ["present"]


def test_manifest_is_idempotent(tmp_path):
    specs = [DV.VariantSpec("present", dir_override="tiles_dataset_present")]
    _make_variant_dir(tmp_path, "tiles_dataset_present", n_cog=2, n_neg=2)
    mp = tmp_path / "manifest.json"
    first = DV.build_manifest(specs, root=tmp_path, manifest_path=mp)
    second = DV.build_manifest(specs, root=tmp_path, manifest_path=mp)
    assert first == second   # re-scanning an unchanged tree is a no-op
