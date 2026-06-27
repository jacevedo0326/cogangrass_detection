"""U2 tests: variant naming + manifest (correct, idempotent) + ExG / label-area toggles.

No test for the heavy re-tiling wrapper itself (plan U2) — we assert the pure per-tile
rule and the manifest, which is where drift would silently corrupt results.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import data_variants as DV  # noqa: E402


# --- naming -------------------------------------------------------------------
def test_reference_variants_point_at_existing_dirs():
    assert DV.VARIANTS_BY_NAME["reference"].dir_name() == "tiles_dataset"
    assert DV.VARIANTS_BY_NAME["reference_clahe"].dir_name() == "tiles_dataset_clahe"
    assert DV.VARIANTS_BY_NAME["reference_0422clean"].dir_name() == "tiles_dataset_0422clean"


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


# --- U6: multi-scale / flip TTA views -----------------------------------------
def test_view_transforms_preserve_shape_and_label_geometry():
    arr = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    for name, fn in DV.VIEW_TRANSFORMS.items():
        out = fn(arr)
        assert out.shape == arr.shape, f"{name} changed the tile shape"
    # hflip is a real (non-identity) view and is its own inverse
    flipped = DV.hflip_view(arr)
    assert not np.array_equal(flipped, arr)
    assert np.array_equal(DV.hflip_view(flipped), arr)


def test_flip_view_averaging_preserves_shape():
    import common as C2  # noqa
    from models import ensemble as E
    probs_identity = np.array([0.2, 0.8, 0.6])
    probs_hflip = np.array([0.4, 0.7, 0.5])
    averaged = E.average_probs([probs_identity, probs_hflip])
    assert averaged.shape == probs_identity.shape
    assert np.allclose(averaged, [0.3, 0.75, 0.55])


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


# --- U3: suspect-negative ranking + cleaned 0422 variant ----------------------
def _scores(triples):
    """triples: list of (path, true_label, p_cog) -> list[ScoreRecord]."""
    return [C.ScoreRecord(path=p, frame=C.frame_of(p), true_label=lab, p_cogongrass=pc)
            for p, lab, pc in triples]


def test_rank_suspect_negatives_surfaces_high_confidence_negatives():
    recs = _scores([
        ("tiles/not_cogongrass/DJI_20260422_0001_r0_c0.jpg", "not_cogongrass", 0.97),  # suspect
        ("tiles/not_cogongrass/DJI_20260422_0001_r0_c1.jpg", "not_cogongrass", 0.10),  # fine
        ("tiles/cogongrass/DJI_20260422_0002_r0_c0.jpg", "cogongrass", 0.99),          # not a neg
        ("tiles/not_cogongrass/DJI_20260606_0003_r0_c0.jpg", "not_cogongrass", 0.95),  # wrong date
    ])
    sus = DV.rank_suspect_negatives(recs, min_p=0.5)
    assert [s["frame"] for s in sus] == ["DJI_20260422_0001"]   # only the 0422 high-conf negative
    assert sus[0]["p_cogongrass"] == 0.97


def test_merge_sidecar_scores_averages_and_checks_alignment():
    a = _scores([("t/not_cogongrass/DJI_20260422_0001_r0_c0.jpg", "not_cogongrass", 0.8)])
    b = _scores([("t/not_cogongrass/DJI_20260422_0001_r0_c0.jpg", "not_cogongrass", 0.6)])
    merged = DV.merge_sidecar_scores([a, b])
    assert merged[0].p_cogongrass == pytest.approx(0.7)
    # a sidecar with a different path set is rejected, not silently averaged
    c = _scores([("t/not_cogongrass/DJI_20260422_0009_r0_c0.jpg", "not_cogongrass", 0.6)])
    try:
        DV.merge_sidecar_scores([a, c])
        assert False, "expected a path-mismatch error"
    except ValueError:
        pass


def _make_unique_variant(root: Path, n_cog: int, n_neg: int):
    """Like _make_variant_dir but filenames are unique across classes (real tiles always are)."""
    base = root / "tiles_dataset"
    for i in range(n_cog):
        (base / "cogongrass").mkdir(parents=True, exist_ok=True)
        (base / "cogongrass" / f"DJI_20260422_0001_r0_c{i}.jpg").write_bytes(b"x")
    for i in range(n_neg):
        (base / "not_cogongrass").mkdir(parents=True, exist_ok=True)
        (base / "not_cogongrass" / f"DJI_20260422_0002_r1_c{i}.jpg").write_bytes(b"x")


def test_build_clean_variant_flips_relabeled_tiles_and_is_enumerable(tmp_path):
    _make_unique_variant(tmp_path, n_cog=1, n_neg=3)
    flip = "DJI_20260422_0002_r1_c0.jpg"   # a not_cogongrass tile that is really cogongrass
    before = DV.class_counts(C.enumerate_tiles(str(tmp_path / "tiles_dataset"))[0])
    out, n_flipped = DV.build_clean_variant({flip: "cogongrass"}, root=tmp_path)
    assert n_flipped == 1 and out.name == "tiles_dataset_0422clean"
    samples, _classes, _cog = C.enumerate_tiles(str(out))
    after = DV.class_counts(samples)
    assert after["total"] == before["total"]                 # no tiles lost
    assert after["cogongrass"] == before["cogongrass"] + 1   # exactly one negative flipped
    assert after["not_cogongrass"] == before["not_cogongrass"] - 1


def test_build_clean_variant_is_idempotent(tmp_path):
    _make_unique_variant(tmp_path, n_cog=1, n_neg=2)
    DV.build_clean_variant({}, root=tmp_path)
    out, n_flipped = DV.build_clean_variant({}, root=tmp_path)   # second call is a no-op
    assert n_flipped == 0
