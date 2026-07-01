"""Equivalence + behavior tests for tile_common (plan U1, R1/R2/R4/R5/R6).

The load-bearing tests are the LIVE-DATASET equivalence checks: tile_common,
the protected baseline ``train_tiles.py``, and ``train_tiles_collection.py`` are
deliberate copies of one contract, and these tests are the tripwire that catches
drift between them (plan KTD: "checked duplication replaces intentional
duplication"). Reading/importing the baseline is allowed; only modifying it is not.

Run:  .venv/bin/pytest tests/test_tile_common.py -x -q
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tile_common as tc

TILES_DIR = REPO / "tiles_dataset"
needs_live = pytest.mark.skipif(not TILES_DIR.is_dir(),
                                reason="live tiles_dataset/ not present")


# ---------------------------------------------------------------------------
# Live-dataset equivalence with the protected baseline + the DA-track split
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    if not TILES_DIR.is_dir():
        pytest.skip("live tiles_dataset/ not present")
    samples, classes, cog_idx = tc.enumerate_tiles(str(TILES_DIR))
    return samples, classes, cog_idx


@needs_live
def test_grouped_split_matches_train_tiles(live):
    """tile_common.grouped_split == train_tiles.grouped_split, same seed, live data."""
    samples, _, cog_idx = live
    import train_tiles as T   # read-only import of the protected baseline

    # The baseline shuffles with the module-global RNG (seeded at import); reseed so
    # its state is exactly the fresh random.Random(42) tile_common uses internally.
    random.seed(T.SEED)
    tr1, va1, te1, nf1 = T.grouped_split(samples, cog_idx)
    tr2, va2, te2, nf2 = tc.grouped_split(samples, cog_idx, seed=42)

    assert nf1 == nf2
    assert tr1 == tr2 and va1 == va2 and te1 == te2   # exact index lists, not just sets

    frames = lambda idx: {tc.frame_of(samples[i][0]) for i in idx}
    assert frames(tr1) == frames(tr2)
    assert frames(va1) == frames(va2)
    assert frames(te1) == frames(te2)
    # frame-grouped: no frame spans two splits
    assert frames(tr2).isdisjoint(frames(va2)) and frames(tr2).isdisjoint(frames(te2))


@needs_live
def test_collection_split_matches_train_tiles_collection(live):
    """tile_common.split_by_collection == train_tiles_collection.split_by_collection."""
    samples, _, cog_idx = live
    import train_tiles_collection as TC   # safe: main() is guarded

    tr1, va1, te1, nf1 = TC.split_by_collection(samples, cog_idx)
    tr2, va2, te2, nf2 = tc.split_by_collection(samples, cog_idx)   # default 20260422

    assert nf1 == nf2
    assert tr1 == tr2 and va1 == va2 and te1 == te2


@needs_live
def test_whitelist_frames_land_in_train_pool(live):
    """The 14 non-DJI frames route to the TRAIN pool (never held out), as today (R4)."""
    samples, _, cog_idx = live
    tr, va, te, _ = tc.split_by_collection(samples, cog_idx)
    frames = lambda idx: {tc.frame_of(samples[i][0]) for i in idx}
    present = {tc.frame_of(p) for p, _ in samples} & tc.NON_DJI_FRAME_WHITELIST
    assert len(present) == 14
    assert present & frames(te) == set()
    assert present <= frames(tr) | frames(va)


@needs_live
def test_balance_matches_train_tiles_collection(live):
    """tile_common.balance (ratio=1.0) == train_tiles_collection.balance, same rng."""
    samples, _, cog_idx = live
    import train_tiles_collection as TC

    tr, *_ = tc.split_by_collection(samples, cog_idx)
    assert TC.balance(tr, samples, cog_idx, random.Random(42)) == \
        tc.balance(tr, samples, cog_idx, random.Random(42))


# ---------------------------------------------------------------------------
# Mixed oblique + orthomosaic fixture: group keys and the divergence boundary
# ---------------------------------------------------------------------------
def mixed_samples():
    """(path, label) pairs mixing DJI oblique tiles and ortho flight+block tiles."""
    out = []
    for f in ["DJI_20260422_0001", "DJI_20260422_0002", "DJI_20260606_0001"]:
        for r in range(2):
            out.append((f"{f}_r{r}_c0.jpg", r % 2))
    for blk in ["bk0_0", "bk0_1"]:
        for r, c in [(0, 0), (0, 1)]:
            out.append((f"siteA-20270301_{blk}_r{r}_c{c}.jpg", 0))
    return out


def test_group_of_dji_and_ortho():
    assert tc.group_of("DJI_20260422_1234_r3_c7.jpg") == "DJI_20260422_1234"
    # ortho: group is flight + spatial block
    assert tc.group_of("siteA-20270301_bk2_5_r0_c1.jpg") == "siteA-20270301_bk2_5"
    # two tiles of the same block share a group; different blocks differ
    assert tc.group_of("siteA-20270301_bk2_5_r0_c0.jpg") == \
        tc.group_of("siteA-20270301_bk2_5_r4_c9.jpg")
    assert tc.group_of("siteA-20270301_bk2_5_r0_c0.jpg") != \
        tc.group_of("siteA-20270301_bk2_6_r0_c0.jpg")
    assert tc.has_block("siteA-20270301_bk2_5")
    assert not tc.has_block("DJI_20260422_1234")


def test_arch_sweep_divergence_boundary_documented():
    """arch_sweep/common.date_of returns "other" for ortho stems — the KNOWN divergence.

    arch_sweep/common.py stays permanently self-contained (plan KTD): its legacy
    date_of would route ortho tiles to the sweep's TRAIN pool, while tile_common
    parses the flight date from the block stem. This test makes that boundary
    explicit instead of latent — it only matters once ortho tiles exist.
    """
    sys.path.insert(0, str(REPO / "arch_sweep"))
    try:
        import common as sweep_common
    finally:
        sys.path.remove(str(REPO / "arch_sweep"))

    stem = "siteA-20270301_bk2_5"
    assert sweep_common.date_of(stem) == "other"       # legacy: silently trainable
    assert tc.date_of(stem) == "20270301"              # tile_common: a real collection
    # and on DJI stems the two agree exactly
    assert sweep_common.date_of("DJI_20260422_0001") == tc.date_of("DJI_20260422_0001")


def test_mixed_fixture_split_routes_each_stem_rule():
    samples = mixed_samples()
    tr, va, te, _ = tc.split_by_collection(samples, cog_idx=1)   # default heldout 0422
    frames = lambda idx: {tc.group_of(samples[i][0]) for i in idx}
    assert frames(te) == {"DJI_20260422_0001", "DJI_20260422_0002"}
    assert frames(tr) | frames(va) == {"DJI_20260606_0001",
                                       "siteA-20270301_bk0_0", "siteA-20270301_bk0_1"}


# ---------------------------------------------------------------------------
# date_of: DJI, ortho, whitelist, hard error
# ---------------------------------------------------------------------------
def test_date_of_dji():
    assert tc.date_of("DJI_20260606_1234") == "20260606"
    assert tc.date_of("DJI_20260422_0007_r3_c7") == "20260422"   # tile stem also fine


def test_date_of_ortho():
    assert tc.date_of("siteA-20270301_bk2_5") == "20270301"
    assert tc.date_of("siteA-20270301_bk2_5_r0_c1.jpg") == "20270301"


def test_date_of_whitelisted_real_frame():
    # BSEJ8944 is one of the 14 real non-DJI stems pre-seeded in the whitelist —
    # "other" preserves its current train-pool routing (R4).
    assert tc.date_of("BSEJ8944_r0_c0") == "other"
    assert tc.date_of("BSEJ8944") == "other"


def test_date_of_novel_garbage_raises():
    with pytest.raises(ValueError, match="IMG_9999"):
        tc.date_of("IMG_9999_r0_c0")
    with pytest.raises(ValueError, match="totally-new-scheme"):
        tc.date_of("totally-new-scheme")


# ---------------------------------------------------------------------------
# HELDOUT_DATES override moves collections between pools
# ---------------------------------------------------------------------------
def test_heldout_override_moves_collections():
    samples = mixed_samples()
    tr, va, te, _ = tc.split_by_collection(samples, cog_idx=1,
                                           heldout_dates=["20270301"])
    frames = lambda idx: {tc.group_of(samples[i][0]) for i in idx}
    # the ortho flight is now the held-out collection ...
    assert frames(te) == {"siteA-20270301_bk0_0", "siteA-20270301_bk0_1"}
    # ... and 20260422 returns to the train pool
    assert {"DJI_20260422_0001", "DJI_20260422_0002"} <= frames(tr) | frames(va)


def test_heldout_dates_from_env():
    assert tc.heldout_dates_from_env({}) == ["20260422"]
    assert tc.heldout_dates_from_env({"HELDOUT_DATES": ""}) == ["20260422"]
    assert tc.heldout_dates_from_env({"HELDOUT_DATES": "20270301"}) == ["20270301"]
    assert tc.heldout_dates_from_env(
        {"HELDOUT_DATES": "20270301, 20280101"}) == ["20270301", "20280101"]


# ---------------------------------------------------------------------------
# Tiling geometry parity with boxes_to_tiles.py (the oracle is its exact loop)
# ---------------------------------------------------------------------------
def oracle_tile_boxes(W, H, TILE):
    """boxes_to_tiles.py:67-72's exact arithmetic, copied verbatim as the oracle."""
    cols, rows = -(-W // TILE), -(-H // TILE)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * TILE, c * TILE
            y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
            out.append((r, c, (x0, y0, x1, y1)))
    return out


@pytest.mark.parametrize("W,H", [(4096, 3072), (4000, 3000), (300, 200)])
def test_tile_boxes_matches_boxes_to_tiles(W, H):
    assert tc.tile_boxes(W, H, 512) == oracle_tile_boxes(W, H, 512)
    assert tc.tile_grid(W, H, 512) == (-(-W // 512), -(-H // 512))


def test_image_smaller_than_tile_yields_one_clamped_box():
    assert tc.tile_boxes(300, 200, 512) == [(0, 0, (0, 0, 300, 200))]


def test_cut_tile_matches_save_behavior():
    from PIL import Image
    im = Image.new("RGB", (4000, 3000), (10, 200, 10))
    *_, box = tc.tile_boxes(4000, 3000, 512)[-1]
    # bottom-right clamped edge crop, resized UP to the square save size
    assert box == (3584, 2560, 4000, 3000)
    tile = tc.cut_tile(im, box, 512)
    assert tile.size == (512, 512)
    ref = im.crop(box).resize((512, 512))          # boxes_to_tiles.py:81 verbatim
    assert np.array_equal(np.asarray(tile), np.asarray(ref))


# ---------------------------------------------------------------------------
# ExG vegetation filter
# ---------------------------------------------------------------------------
def test_exg_green_passes_gray_fails():
    green = np.zeros((64, 64, 3), np.uint8)
    green[..., 0], green[..., 1], green[..., 2] = 30, 200, 30
    assert tc.tile_is_veg(tc.exg_map(green), (0, 0, 64, 64))

    gray = np.full((64, 64, 3), 128, np.uint8)     # ExG == 0 < 0.03 -> dropped
    assert not tc.tile_is_veg(tc.exg_map(gray), (0, 0, 64, 64))
    soil = np.zeros((64, 64, 3), np.uint8)         # brown-ish bare ground
    soil[..., 0], soil[..., 1], soil[..., 2] = 140, 110, 80
    assert not tc.tile_is_veg(tc.exg_map(soil), (0, 0, 64, 64))


# ---------------------------------------------------------------------------
# Provenance manifest: round-trip, hash stability, atomicity
# ---------------------------------------------------------------------------
def manifest_params():
    return {"tile_px": 512, "tile_save_px": 512, "prep_max": 4096,
            "jpeg_quality": 88, "veg_thresh": 0.03, "source_digest": "abc123",
            "created_at": "2026-07-01T00:00:00", "git": "test-fixed"}


def test_manifest_roundtrip(tmp_path):
    path = tc.write_provenance(tmp_path, manifest_params())
    assert path == tmp_path / tc.PROVENANCE_NAME
    assert tc.read_provenance(tmp_path) == manifest_params()


def test_manifest_missing_required_key_raises(tmp_path):
    bad = manifest_params()
    del bad["tile_px"]
    with pytest.raises(ValueError, match="tile_px"):
        tc.write_provenance(tmp_path, bad)
    assert not (tmp_path / tc.PROVENANCE_NAME).exists()


def test_provenance_hash_stable_across_key_order():
    a = manifest_params()
    b = dict(reversed(list(a.items())))
    assert list(a) != list(b)                       # genuinely different key order
    assert tc.provenance_hash(a) == tc.provenance_hash(b)
    changed = dict(a, tile_px=256)
    assert tc.provenance_hash(changed) != tc.provenance_hash(a)


def test_partial_write_leaves_old_manifest_intact(tmp_path):
    old = manifest_params()
    tc.write_provenance(tmp_path, old)
    # simulate a writer killed mid-write: temp file written but never os.replace'd
    (tmp_path / ".tmp-crashed.json").write_text('{"tile_px": 256, "trunc')
    assert tc.read_provenance(tmp_path) == old      # reader never sees the partial file


def test_source_digest_changes_on_content_change(tmp_path):
    a, b = tmp_path / "f1.jpg", tmp_path / "f2.jpg"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bb")
    d1 = tc.source_digest([a, b])
    assert d1 == tc.source_digest([b, a])           # order-independent (sorted)
    a.write_bytes(b"aaaaaa")                        # size change under the same name
    assert tc.source_digest([a, b]) != d1


# ---------------------------------------------------------------------------
# AdaBN (DA path only) smoke: cumulative stats over the given batches, eval after
# ---------------------------------------------------------------------------
def test_adapt_bn_recomputes_stats_and_restores_eval():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.BatchNorm2d(3))
    bn = model[0]
    bn.running_mean.fill_(9.0)                      # pretend "source" stats
    batches = [torch.randn(8, 3, 4, 4) + 2.0, torch.randn(8, 3, 4, 4) + 2.0]
    n = tc.adapt_bn(model, batches, device="cpu", verbose=False)
    assert n == 16
    assert not model.training                       # returned to eval
    x = torch.cat(batches)
    expected = x.mean(dim=(0, 2, 3))
    assert torch.allclose(bn.running_mean, expected, atol=1e-4)
