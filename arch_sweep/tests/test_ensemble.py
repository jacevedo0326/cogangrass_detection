"""U5 tests: ensemble aligns by paths, averages probs, scores like any cell, EATA runs.

Synthetic separable feature caches (no backbone/GPU) are injected into ``run_ensemble`` — the
same code path real ensemble cells use, minus extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
from models import ensemble as E  # noqa: E402

COG_IDX = 0


def _samples(n_train_frames=20, n_test_frames=8):
    samples = []
    for date, nf in [(C.TRAIN_DATE, n_train_frames), (C.TEST_DATE, n_test_frames)]:
        for f in range(nf):
            lab = f % 2                       # 0 = cogongrass, 1 = not_cogongrass
            cls = "cogongrass" if lab == 0 else "not_cogongrass"
            frame = f"DJI_{date}_{f:04d}"
            for t in range(3):
                samples.append((f"tiles/{cls}/{frame}_r0_c{t}.jpg", lab))
    return samples


def _cache(samples, seed, *, dim=16, noise=0.1, sep=2.0):
    """A member's feature cache: separable, decorrelated by seed noise."""
    rng = np.random.RandomState(seed)
    paths = [p for p, _ in samples]
    labels = np.asarray([lab for _, lab in samples])
    feats = np.asarray([np.full(dim, sep if lab == 0 else -sep) + rng.randn(dim) * noise
                        for _, lab in samples], dtype=np.float32)
    return {"features": feats, "labels": labels, "paths": paths}


def _cfg():
    return dict(max_epochs=20, patience=6, hidden=8, batch_size=8)


# --- alignment by paths -------------------------------------------------------
def test_align_reindexes_members_to_a_common_order():
    samples = _samples()
    c1 = _cache(samples, 0)
    # member 2: identical per-path features, but rows shuffled (and paths shuffled with them)
    perm = list(range(len(samples)))[::-1]
    c2 = {"features": c1["features"][perm].copy(),
          "labels": c1["labels"][perm].copy(),
          "paths": [c1["paths"][i] for i in perm]}
    paths, feats_list, labels = E.align_by_paths([c1, c2])
    assert paths == c1["paths"]
    # after alignment member 2's rows line up with member 1 again (per-path features identical)
    assert np.allclose(feats_list[0], feats_list[1])
    assert list(labels) == list(c1["labels"])


def test_align_detects_mismatched_tile_sets():
    samples = _samples()
    c1 = _cache(samples, 0)
    bad = _cache(samples, 1)
    bad["paths"] = list(bad["paths"])
    bad["paths"][0] = "tiles/cogongrass/DJI_20260606_9999_r0_c0.jpg"   # a tile c1 lacks
    with pytest.raises(ValueError):
        E.align_by_paths([c1, bad])


def test_average_probs_is_elementwise_mean():
    out = E.average_probs([[0.2, 0.8], [0.4, 0.6]])
    assert np.allclose(out, [0.3, 0.7])


# --- ensemble scores like any cell, >= best member ----------------------------
def test_ensemble_scores_at_least_best_member(tmp_path):
    samples = _samples()
    c1, c2 = _cache(samples, 0), _cache(samples, 99)
    a = E.run_ensemble(["m1"], caches=[c1], cog_idx=COG_IDX, results_dir=tmp_path, **_cfg())
    b = E.run_ensemble(["m2"], caches=[c2], cog_idx=COG_IDX, results_dir=tmp_path, **_cfg())
    ens = E.run_ensemble(["m1", "m2"], caches=[c1, c2], cog_idx=COG_IDX,
                         results_dir=tmp_path, **_cfg())
    assert ens.status == "ok"
    assert ens.balanced_accuracy >= max(a.balanced_accuracy, b.balanced_accuracy) - 1e-9


def test_ensemble_row_identity_is_distinct_and_cross_collection(tmp_path):
    samples = _samples()
    c1, c2 = _cache(samples, 0), _cache(samples, 99)
    ens = E.run_ensemble(["m1", "m2"], caches=[c1, c2], cog_idx=COG_IDX,
                         results_dir=tmp_path, **_cfg())
    assert ens.model == "ensemble" and ens.eval_setting == C.EVAL_CROSS
    assert ens.extra == "ensemble=m1+m2"
    # a different membership / soup flag yields a different job_id (KTD2)
    soup = E.run_ensemble(["m1", "m2"], caches=[c1, c2], cog_idx=COG_IDX, seed_soup=True,
                          seeds=(42, 1), results_dir=tmp_path, **_cfg())
    assert soup.job_id != ens.job_id and soup.status == "ok"


def test_ensemble_eata_tier_runs(tmp_path):
    samples = _samples()
    c1, c2 = _cache(samples, 0), _cache(samples, 99)
    ens = E.run_ensemble(["m1", "m2"], caches=[c1, c2], cog_idx=COG_IDX, adaptation="eata",
                         results_dir=tmp_path, **_cfg())
    assert ens.status == "ok" and ens.adaptation == "eata"


def test_ensemble_writes_scores_sidecar(tmp_path):
    samples = _samples()
    c1, c2 = _cache(samples, 0), _cache(samples, 99)
    ens = E.run_ensemble(["m1", "m2"], caches=[c1, c2], cog_idx=COG_IDX,
                         results_dir=tmp_path, write_scores=True, **_cfg())
    sidecars = list(tmp_path.glob("*.scores.jsonl"))
    assert len(sidecars) == 1
    recs = C.read_scores(sidecars[0])
    assert len(recs) == ens.n_test
    assert len(C.read_all_results(tmp_path)) == 1   # sidecar excluded from the merge
