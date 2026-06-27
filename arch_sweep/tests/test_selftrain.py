"""U11 tests: agreement gate, prior cap, pool/eval frame disjointness, non-decreasing rounds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
from models import selftrain as ST  # noqa: E402

COG_IDX = 0


def _samples(n_train_frames=24, n_test_frames=16):
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
    rng = np.random.RandomState(seed)
    paths = [p for p, _ in samples]
    labels = np.asarray([lab for _, lab in samples])
    feats = np.asarray([np.full(dim, sep if lab == 0 else -sep) + rng.randn(dim) * noise
                        for _, lab in samples], dtype=np.float32)
    return {"features": feats, "labels": labels, "paths": paths}


def _cfg():
    return dict(max_epochs=20, patience=6, hidden=8, batch_size=8)


# --- agreement gate -----------------------------------------------------------
def test_agreement_requires_k_of_n_high_confidence():
    # 4 members, 3 tiles: all-high, split (2 high / 2 low), all-low
    member_probs = np.array([
        [0.9, 0.9, 0.1],
        [0.85, 0.6, 0.05],
        [0.95, 0.1, 0.2],
        [0.88, 0.1, 0.15],
    ])
    pseudo = ST.ensemble_agreement(member_probs, hi=0.8, lo=0.2, agree_k=3)
    assert pseudo[0] == 1            # 4/4 high -> pseudo-positive
    assert pseudo[1] == ST.ABSTAIN  # only 1 high, 2 low -> not >=3 either way -> abstain
    assert pseudo[2] == 0           # 3/4 low -> pseudo-negative


def test_only_3_of_4_or_more_added():
    member_probs = np.array([[0.9], [0.9], [0.6], [0.6]])   # 2 high, 2 mid
    assert ST.ensemble_agreement(member_probs, agree_k=3)[0] == ST.ABSTAIN
    member_probs2 = np.array([[0.9], [0.9], [0.85], [0.6]])  # 3 high
    assert ST.ensemble_agreement(member_probs2, agree_k=3)[0] == 1


# --- prior cap ----------------------------------------------------------------
def test_cap_positives_to_prior_keeps_most_confident():
    pseudo = np.array([1, 1, 1, 1, 0])
    mean_p = np.array([0.99, 0.81, 0.95, 0.85, 0.05])
    capped = ST.cap_positives_to_prior(pseudo, mean_p, prior=0.4, n_reference=5)  # cap = 2 positives
    assert int((capped == 1).sum()) == 2
    kept = set(np.where(capped == 1)[0].tolist())
    assert kept == {0, 2}            # the two highest-confidence positives survive
    assert capped[4] == 0            # negatives untouched


# --- end-to-end loop ----------------------------------------------------------
def test_pool_and_eval_frames_strictly_disjoint(tmp_path):
    samples = _samples()
    c1, c2, c3, c4 = (_cache(samples, s) for s in (0, 1, 2, 3))
    row = ST.run_selftrain(["m1", "m2", "m3", "m4"], caches=[c1, c2, c3, c4], cog_idx=COG_IDX,
                           rounds=2, agree_k=3, results_dir=tmp_path, **_cfg())
    # the assertion inside run_selftrain guards disjointness; reaching ok status proves it held
    assert row.status == "ok" and row.eval_setting == C.EVAL_CROSS
    assert row.extra == "pseudo=agree3"


def test_rounds_are_non_decreasing_on_separable_data(tmp_path):
    samples = _samples()
    caches = [_cache(samples, s) for s in (0, 1, 2, 3)]
    row = ST.run_selftrain(["m1", "m2", "m3", "m4"], caches=caches, cog_idx=COG_IDX,
                           rounds=3, agree_k=3, results_dir=tmp_path, **_cfg())
    per_round = row.f2_sweep[-1]["round_baccs"]
    assert len(per_round) == 3
    # separable data: a self-training round must not hurt (tiny tolerance for head-init noise)
    assert all(per_round[i + 1] >= per_round[i] - 0.05 for i in range(len(per_round) - 1))
    assert row.balanced_accuracy >= 0.8


def test_selftrain_writes_scores_sidecar_and_merges_clean(tmp_path):
    samples = _samples()
    caches = [_cache(samples, s) for s in (0, 1, 2, 3)]
    row = ST.run_selftrain(["m1", "m2", "m3", "m4"], caches=caches, cog_idx=COG_IDX,
                           rounds=2, results_dir=tmp_path, write_scores=True, **_cfg())
    assert len(list(tmp_path.glob("*.scores.jsonl"))) == 1
    assert len(C.read_all_results(tmp_path)) == 1   # sidecar excluded from the merge
    assert row.n_test > 0
