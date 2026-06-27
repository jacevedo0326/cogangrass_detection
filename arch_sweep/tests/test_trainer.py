"""U4 tests: shared trainer trains a head, writes one honest row, threshold is 0606-only.

Uses a tiny synthetic, linearly-separable feature set (no backbone / GPU needed) injected
straight into ``train_and_eval`` — the same code path the real cells use, minus extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import heads as H  # noqa: E402
import trainer as T  # noqa: E402

COG_IDX = 0


def _synth(dim=16, n_train_frames=20, n_test_frames=8, seed=0):
    """Frame-clean synthetic ImageFolder: each frame is single-class, classes separable."""
    rng = np.random.RandomState(seed)
    samples, feats, labels = [], [], []

    def add(date, n_frames):
        for f in range(n_frames):
            lab = f % 2                       # 0 = cogongrass, 1 = not_cogongrass
            frame = f"DJI_{date}_{f:04d}"
            cls = "cogongrass" if lab == 0 else "not_cogongrass"
            for t in range(3):
                samples.append((f"tiles/{cls}/{frame}_r0_c{t}.jpg", lab))
                base = 2.0 if lab == 0 else -2.0
                feats.append(np.full(dim, base) + rng.randn(dim) * 0.1)
                labels.append(lab)

    add(C.TRAIN_DATE, n_train_frames)   # 0606 -> train/val
    add(C.TEST_DATE, n_test_frames)     # 0422 -> held-out test
    return samples, np.asarray(feats, dtype=np.float32), np.asarray(labels)


def _cfg(**kw):
    base = dict(model="stub", head="mlp_bn", max_epochs=20, patience=6, hidden=8, batch_size=8)
    base.update(kw)
    return T.TrainConfig(**base)


# --- head registry ------------------------------------------------------------
def test_mlp_head_exposes_batchnorm_linear_does_not():
    assert H.has_batchnorm(H.build_head("mlp_bn", in_dim=16))      # AdaBN/TTA precondition
    assert not H.has_batchnorm(H.build_head("linear", in_dim=16))
    with pytest.raises(ValueError):
        H.build_head("bogus", in_dim=16)


# --- smoke: trains, writes one valid row --------------------------------------
def test_train_and_eval_smoke_writes_one_row(tmp_path):
    samples, feats, labels = _synth()
    row = T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples,
                           features=feats, labels=labels, cog_idx=COG_IDX)
    assert row.status == "ok"
    assert row.eval_setting == C.EVAL_CROSS
    assert row.balanced_accuracy is not None and row.balanced_accuracy > 0.8   # separable
    assert row.trainable_params and row.trainable_params > 0
    # exactly one file, round-trips
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert C.read_result(files[0]).job_id == row.job_id


def test_scores_sidecar_off_by_default(tmp_path):
    samples, feats, labels = _synth()
    T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples,
                     features=feats, labels=labels, cog_idx=COG_IDX)
    assert not list(tmp_path.glob("*.scores.jsonl"))   # off unless explicitly requested


def test_scores_sidecar_written_one_row_per_0422_tile(tmp_path):
    samples, feats, labels = _synth()
    row = T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples, features=feats,
                           labels=labels, cog_idx=COG_IDX, write_scores=True)
    sidecars = list(tmp_path.glob("*.scores.jsonl"))
    assert len(sidecars) == 1
    recs = C.read_scores(sidecars[0])
    n_te = len(C.indices_for_date(samples, C.TEST_DATE))
    assert len(recs) == n_te == row.n_test
    assert all(0.0 <= r.p_cogongrass <= 1.0 for r in recs)
    assert all(r.frame and "20260422" in r.frame for r in recs)   # only held-out 0422 tiles
    # the sidecar does not pollute the result merge that feeds the report
    assert len(C.read_all_results(tmp_path)) == 1


def test_both_head_variants_train(tmp_path):
    samples, feats, labels = _synth()
    for head in ("linear", "mlp_bn"):
        row = T.train_and_eval(_cfg(head=head), results_dir=tmp_path, samples=samples,
                               features=feats, labels=labels, cog_idx=COG_IDX)
        assert row.status == "ok" and row.balanced_accuracy is not None


# --- threshold honesty: fit on 0606 ONLY --------------------------------------
def test_operating_threshold_is_independent_of_0422_labels(tmp_path):
    samples, feats, labels = _synth()
    base = T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples,
                            features=feats, labels=labels, cog_idx=COG_IDX)
    # Corrupt ONLY the 0422 slice (labels + features). If the threshold were selected on
    # 0422 it would move; the honest rule fixes it on 0606, so it must not change.
    te = C.indices_for_date(samples, C.TEST_DATE)
    feats2, labels2 = feats.copy(), labels.copy()
    for i in te:
        labels2[i] = 1 - labels2[i]
        feats2[i] = -feats2[i]
    other = T.train_and_eval(_cfg(extra="te-corrupted"), results_dir=tmp_path, samples=samples,
                             features=feats2, labels=labels2, cog_idx=COG_IDX)
    assert base.threshold == other.threshold


# --- determinism: same cfg + seed -> same metric ------------------------------
def test_determinism_same_cfg_same_balanced_accuracy(tmp_path):
    samples, feats, labels = _synth()
    a = T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples,
                         features=feats, labels=labels, cog_idx=COG_IDX)
    b = T.train_and_eval(_cfg(), results_dir=tmp_path, samples=samples,
                         features=feats, labels=labels, cog_idx=COG_IDX)
    assert a.balanced_accuracy == b.balanced_accuracy
    assert a.threshold == b.threshold


# --- failure isolation: an unloadable backbone records a row, never raises ----
def test_failure_is_recorded_not_raised(tmp_path):
    # a non-frozen cell on an unregistered backbone fails to build -> recorded, not raised
    samples, feats, labels = _synth()
    row = T.train_and_eval(_cfg(model="no-such-backbone", tuning_mode="full"),
                           results_dir=tmp_path, samples=samples, features=feats,
                           labels=labels, cog_idx=COG_IDX)
    assert row.status in ("failed", "oom")
    assert list(tmp_path.glob("*.jsonl"))   # row still written (honest accounting)
