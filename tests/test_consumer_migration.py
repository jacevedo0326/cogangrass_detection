"""Consumer-migration tests (plan U2): shared-contract imports, full-res defaults,
and the data-safety guards (wipe guard/scope, label protection, stem collision,
manifest-aware feature signature).

The split-identity test compares against the characterization capture taken from
the PRE-migration train_tiles_collection.py on the live dataset (split_before.json);
it is skipped when either the live dataset or the capture is absent.

Run:  .venv/bin/pytest tests/test_consumer_migration.py -q
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "arch_sweep")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tile_common as tc

import boxes_to_tiles as btt
import label_tiles
import prep_images

TILES_DIR = REPO / "tiles_dataset"
SPLIT_BEFORE = Path("/tmp/claude-1000/-home-josh-dev-cogangrass-detection/"
                    "00714f72-22b1-4205-9f12-0934311118b3/scratchpad/split_before.json")
PY = sys.executable


# ---------------------------------------------------------------------------
# Characterization: post-migration split == pre-migration capture (live data)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (TILES_DIR.is_dir() and SPLIT_BEFORE.exists()),
                    reason="live tiles_dataset/ or split_before.json capture absent")
def test_split_identity_vs_pre_migration_capture():
    """The migrated train_tiles_collection split reproduces the captured membership."""
    import train_tiles_collection as TC

    before = json.loads(SPLIT_BEFORE.read_text())
    samples, _classes, cog_idx = tc.enumerate_tiles(str(TILES_DIR))
    tr, va, te, nf = TC.split_by_collection(samples, cog_idx)

    assert list(nf) == before["n_frames"]
    assert [len(tr), len(va), len(te)] == before["n_tiles"]
    assert tr == before["train_idx"] and va == before["val_idx"] and te == before["test_idx"]
    frames = lambda idx: sorted({tc.frame_of(samples[i][0]) for i in idx})
    assert frames(tr) == before["train_frames"]
    assert frames(va) == before["val_frames"]
    assert frames(te) == before["test_frames"]


# ---------------------------------------------------------------------------
# Defaults flip: full-res with no env, legacy layout still reachable via env
# ---------------------------------------------------------------------------
def _probe(code: str, extra_env: dict = None) -> str:
    env = {k: v for k, v in os.environ.items()
           if k not in ("TILE_PX", "TILE_SAVE_PX", "PREP_MAX")}
    env.update(extra_env or {})
    out = subprocess.run([PY, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_boxes_to_tiles_defaults_are_full_res():
    assert _probe("import boxes_to_tiles as b; print(b.TILE, b.CNN_SIZE)") == "512 512"


def test_boxes_to_tiles_legacy_env_still_honored():
    assert _probe("import boxes_to_tiles as b; print(b.TILE, b.CNN_SIZE)",
                  {"TILE_PX": "160", "TILE_SAVE_PX": "224"}) == "160 224"


def test_prep_images_default_is_full_res():
    assert _probe("import prep_images as p; print(p.MAX)") == "4096"
    assert _probe("import prep_images as p; print(p.MAX)", {"PREP_MAX": "1280"}) == "1280"


def test_label_tiles_default_grid_is_512():
    assert _probe("import label_tiles as l; print(l.DEFAULT_TILE)") == "512"


# ---------------------------------------------------------------------------
# Wipe guard (R7): provenance mismatch refuses, force or matching proceeds
# ---------------------------------------------------------------------------
def _manifest(**over):
    m = {"tile_px": 512, "tile_save_px": 512, "prep_max": 4096,
         "jpeg_quality": 88, "veg_thresh": 0.03, "source_digest": "abc",
         "created_at": "2026-07-01T00:00:00"}
    m.update(over)
    return m


def test_wipe_guard_mismatch_aborts_naming_params():
    existing = _manifest()
    requested = dict(_manifest(), tile_px=256)
    with pytest.raises(SystemExit) as e:
        btt.check_wipe_allowed(existing, requested, force=False)
    assert "tile_px" in str(e.value)
    assert "256" in str(e.value) and "512" in str(e.value)


def test_wipe_guard_names_every_differing_param():
    with pytest.raises(SystemExit) as e:
        btt.check_wipe_allowed(_manifest(), dict(_manifest(), tile_px=256,
                                                 tile_save_px=224), force=False)
    assert "tile_px" in str(e.value) and "tile_save_px" in str(e.value)


def test_wipe_guard_force_proceeds():
    assert btt.check_wipe_allowed(_manifest(), dict(_manifest(), tile_px=256),
                                  force=True) is True


def test_wipe_guard_matching_params_proceed():
    assert btt.check_wipe_allowed(_manifest(), _manifest(), force=False) is True
    assert btt.check_wipe_allowed(None, _manifest(), force=False) is True  # fresh dataset


def test_wipe_guard_prep_max_unknown_is_not_compared():
    # prep_max is env/prep-provenance derived; when unknown on either side it must
    # not spuriously block the wipe
    assert btt.check_wipe_allowed(_manifest(), dict(_manifest(), prep_max=None),
                                  force=False) is True


# ---------------------------------------------------------------------------
# Wipe scope: only the two class dirs + manifest, never collection subdirs
# ---------------------------------------------------------------------------
def test_wipe_scope_spares_collection_subdirs(tmp_path):
    ds = tmp_path / "tiles_dataset"
    for cls in ("cogongrass", "not_cogongrass"):
        (ds / cls).mkdir(parents=True)
        (ds / cls / "DJI_20260606_0001_r0_c0.jpg").write_bytes(b"x")
    ortho = ds / "col-20270301"
    ortho.mkdir()
    (ortho / "keep.jpg").write_bytes(b"precious")
    tc.write_provenance(ds, _manifest())

    btt.wipe_class_dirs(ds)

    assert not (ds / "cogongrass").exists()
    assert not (ds / "not_cogongrass").exists()
    assert not (ds / tc.PROVENANCE_NAME).exists()
    assert (ortho / "keep.jpg").read_bytes() == b"precious"   # untouched


# ---------------------------------------------------------------------------
# Label protection (R8): human-edited JSONs survive the bootstrap
# ---------------------------------------------------------------------------
def test_human_edited_label_survives_bootstrap(tmp_path):
    lp = tmp_path / "DJI_20260606_0001.json"
    human = {"image": "DJI_20260606_0001.jpg", "tile_px": 512, "rows": 6, "cols": 8,
             "tiles": {"0,0": "cogongrass"}, "human_edited": True}
    lp.write_text(json.dumps(human))
    written = btt.write_tile_label(lp, {"image": "DJI_20260606_0001.jpg", "tile_px": 512,
                                        "rows": 6, "cols": 8, "cogongrass": []})
    assert written is False
    assert json.loads(lp.read_text()) == human       # byte-for-byte survivor


def test_bootstrap_label_without_flag_is_overwritten(tmp_path):
    lp = tmp_path / "DJI_20260606_0001.json"
    lp.write_text(json.dumps({"image": "DJI_20260606_0001.jpg", "tile_px": 160,
                              "rows": 5, "cols": 8, "cogongrass": [[0, 0]]}))
    new = {"image": "DJI_20260606_0001.jpg", "tile_px": 512, "rows": 6, "cols": 8,
           "cogongrass": []}
    assert btt.write_tile_label(lp, new) is True
    assert json.loads(lp.read_text()) == new


# ---------------------------------------------------------------------------
# Label round-trip: the loader follows the JSON's recorded tile_px (R3)
# ---------------------------------------------------------------------------
def test_label_loader_follows_recorded_tile_px(tmp_path):
    lp = tmp_path / "f.json"
    lp.write_text(json.dumps({"image": "f.jpg", "tile_px": 512, "rows": 6, "cols": 8,
                              "tiles": {"2,3": "cogongrass", "0,1": "torpedograss"}}))
    tiles, tile = label_tiles.load_labels(lp)
    assert tile == 512
    assert tiles == {(2, 3): "cogongrass", (0, 1): "torpedograss"}


def test_label_loader_legacy_binary_format_and_default(tmp_path):
    lp = tmp_path / "g.json"
    lp.write_text(json.dumps({"image": "g.jpg", "tile_px": 160,
                              "cogongrass": [[1, 2]]}))
    tiles, tile = label_tiles.load_labels(lp)
    assert tile == 160                               # recorded size wins, even legacy
    assert tiles == {(1, 2): "cogongrass"}
    # no file -> empty labels at the caller's default
    tiles, tile = label_tiles.load_labels(tmp_path / "missing.json", default_tile=512)
    assert (tiles, tile) == ({}, 512)


# ---------------------------------------------------------------------------
# Stem collision (R9): same stem from two sources errors naming both
# ---------------------------------------------------------------------------
def test_stem_collision_across_source_folders(tmp_path):
    a = tmp_path / "fieldA" / "DJI_0001.JPG"
    b = tmp_path / "fieldB" / "DJI_0001.JPG"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"jpg")
    seen = {}
    prep_images.check_stem_collision("DJI_0001", a, seen)
    with pytest.raises(prep_images.StemCollision) as e:
        prep_images.check_stem_collision("DJI_0001", b, seen)
    assert "fieldA" in str(e.value) and "fieldB" in str(e.value)   # names BOTH sources
    # same source re-seen is not a collision; ALLOW_OVERRIDE bypasses
    prep_images.check_stem_collision("DJI_0001", a, seen)
    prep_images.check_stem_collision("DJI_0001", b, seen, allow_overwrite=True)


def test_output_overwrite_with_different_content_errors(tmp_path):
    out = tmp_path / "DJI_0001.jpg"
    out.write_bytes(b"old-bytes-here")
    with pytest.raises(prep_images.StemCollision) as e:
        prep_images.check_output_overwrite(out, b"new", tmp_path / "src.JPG")
    assert "DJI_0001.jpg" in str(e.value) and "src.JPG" in str(e.value)
    prep_images.check_output_overwrite(out, b"old-bytes-here", "src")   # same size ok
    prep_images.check_output_overwrite(out, b"new", "src", allow_overwrite=True)


# ---------------------------------------------------------------------------
# Feature signature (R6): legacy fallback exact, manifest-aware drift detection
# ---------------------------------------------------------------------------
import features as F  # noqa: E402  (arch_sweep on sys.path above)
import numpy as np  # noqa: E402


def legacy_signature_oracle(samples):
    """The OLD arch_sweep/features.feature_signature, copied verbatim as the oracle."""
    blob = "\n".join(f"{p}\t{lab}" for p, lab in samples)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


SAMPLES = [(f"tiles_dataset/cogongrass/DJI_20260606_000{i}_r0_c{i}.jpg", i % 2)
           for i in range(6)]


def test_signature_legacy_fallback_is_exact():
    """No manifest -> EXACTLY the old path-only hash, so existing caches stay valid."""
    assert F.feature_signature(SAMPLES) == legacy_signature_oracle(SAMPLES)
    assert F.feature_signature(SAMPLES, manifest=None) == legacy_signature_oracle(SAMPLES)
    assert F.signature_era(None) == "legacy"


def test_signature_mixes_in_manifest_and_tracks_param_changes():
    m512 = _manifest()
    m256 = _manifest(tile_px=256)
    legacy = F.feature_signature(SAMPLES)
    s512 = F.feature_signature(SAMPLES, manifest=m512)
    s256 = F.feature_signature(SAMPLES, manifest=m256)
    assert s512 != legacy                     # manifest presence changes the sig
    assert s512 != s256                       # ... and param drift changes it again
    assert F.signature_era(m512) == tc.provenance_hash(m512) != F.signature_era(m256)


def test_dataset_manifest_read(tmp_path):
    assert F.dataset_manifest(None) is None
    assert F.dataset_manifest(tmp_path) is None            # no manifest -> legacy era
    tc.write_provenance(tmp_path, _manifest())
    assert F.dataset_manifest(tmp_path)["tile_px"] == 512


def test_stale_cache_raised_on_signature_mismatch(tmp_path):
    """Retile under unchanged filenames (new manifest) -> StaleFeatureCache, not a hit."""
    feats = np.zeros((6, 4), dtype=np.float32)
    labels = np.array([s[1] for s in SAMPLES])
    paths = [s[0] for s in SAMPLES]
    old_sig = F.feature_signature(SAMPLES)                       # legacy-era cache
    F.save_features("stub", "reference", feats, labels, paths, sig=old_sig,
                    cache_dir=tmp_path)
    # same tile paths, but the dataset now carries a manifest -> different sig
    new_sig = F.feature_signature(SAMPLES, manifest=_manifest())
    with pytest.raises(F.StaleFeatureCache):
        F.load_features("stub", "reference", expected_sig=new_sig, cache_dir=tmp_path)
    # the legacy sig still loads (existing caches stay valid)
    assert F.load_features("stub", "reference", expected_sig=old_sig,
                           cache_dir=tmp_path) is not None


def test_save_features_records_era(tmp_path):
    feats = np.zeros((2, 3), dtype=np.float32)
    F.save_features("stub", "reference", feats, np.zeros(2), ["a", "b"],
                    sig="s", cache_dir=tmp_path)
    assert F.load_features("stub", "reference", cache_dir=tmp_path)["provenance"]["era"] == "legacy"
    m = _manifest()
    F.save_features("stub", "reference", feats, np.zeros(2), ["a", "b"],
                    sig="s2", era=F.signature_era(m), cache_dir=tmp_path)
    assert (F.load_features("stub", "reference", cache_dir=tmp_path)
            ["provenance"]["era"] == tc.provenance_hash(m))


# ---------------------------------------------------------------------------
# heatmap_infer tiling: ceil grid scores the edge tiles the floor grid dropped
# ---------------------------------------------------------------------------
def test_heatmap_infer_scores_edge_tiles(tmp_path):
    from PIL import Image
    import heatmap_infer as hi

    frame = tmp_path / "DJI_20260422_0001.jpg"
    Image.new("RGB", (4000, 3000), (30, 200, 30)).save(frame, quality=90)   # all-veg

    batch, coords, rows, cols, _im = hi.tiles_of(frame)
    assert (cols, rows) == tc.tile_grid(4000, 3000, hi.TILE) == (8, 6)   # ceil, not 7x5
    assert len(coords) == 48                          # every tile is vegetation
    assert (5, 7) in coords                           # the clamped bottom-right edge tile
    assert batch.shape == (48, 3, hi.CNN, hi.CNN)
