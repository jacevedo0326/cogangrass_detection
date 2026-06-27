"""U10 tests: frame-disjoint budget/eval, few_shot tagging, active selection, monotonicity."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import fewshot as FS  # noqa: E402

COG = 0


def _target_set(dim=12, n_frames=20, seed=0):
    """0422-only synthetic: single-class frames, separable features. Returns samples,feats,labels."""
    rng = np.random.RandomState(seed)
    samples, feats, labels = [], [], []
    for f in range(n_frames):
        lab = f % 2
        cls = "cogongrass" if lab == 0 else "not_cogongrass"
        for t in range(4):
            samples.append((f"t/{cls}/DJI_20260422_{f:04d}_r0_c{t}.jpg", lab))
            base = 2.0 if lab == 0 else -2.0
            feats.append(np.full(dim, base) + rng.randn(dim) * 0.1)
            labels.append(lab)
    return samples, np.asarray(feats, np.float32), np.asarray(labels)


def test_fewshot_sweep_runs_all_cells_and_lands_in_report(tmp_path):
    import report as R
    samples, feats, labels = _target_set()
    rows = FS.run_fewshot_sweep("dinov2", feats, labels, samples, COG,
                                adapters=["prototype", "tip"], budgets=[8, 16],
                                results_dir=tmp_path)
    assert len(rows) == 4                                     # 2 adapters × 2 budgets
    assert all(r.eval_setting == C.EVAL_FEWSHOT and r.budget in (8, 16) for r in rows)
    assert len({r.job_id for r in rows}) == 4                # each cell is a distinct job
    # they merge into the report's SEPARATE few-shot table, not the cross-collection ranking
    merged = C.read_all_results(tmp_path)
    _cross, few, _bad = R.split_rows(merged)
    assert len(few) == 4 and not _cross
    report = R.render(merged, bar=0.817)
    assert "Few-shot target adaptation" in report


def test_budget_and_eval_frames_are_disjoint():
    samples, feats, labels = _target_set()
    pool, ev = FS.frame_holdout_split(samples, COG, eval_frac=0.5, seed=1)
    pool_frames = {C.frame_of(samples[i][0]) for i in pool}
    eval_frames = {C.frame_of(samples[i][0]) for i in ev}
    assert pool_frames and eval_frames
    assert pool_frames.isdisjoint(eval_frames)   # no frame leaks budget into eval


def test_row_is_tagged_few_shot_with_budget():
    samples, feats, labels = _target_set()
    row = FS.run_fewshot("dinov2", feats, labels, samples, COG, adapter="prototype",
                         budget=16, write=False)
    assert row.eval_setting == C.EVAL_FEWSHOT
    assert row.budget == 16 and "budget=16" in row.extra
    # excluded from the cross-collection ranking (report split keeps it out)
    import report as R
    cross, few, bad = R.split_rows([row])
    assert not cross and few == [row]


def test_active_selection_more_balanced_than_random_on_skewed_pool():
    # Skewed pool: 80% class A (confident, p far from 0.5), 20% class B (uncertain, p~0.5).
    rng = np.random.RandomState(0)
    nA, nB = 80, 20
    featsA = np.full((nA, 8), 1.0) + rng.randn(nA, 8) * 0.1
    featsB = np.full((nB, 8), -1.0) + rng.randn(nB, 8) * 0.1
    feats = np.vstack([featsA, featsB])
    labels = np.array([COG] * nA + [1 - COG] * nB)
    base_p = np.concatenate([np.full(nA, 0.95), np.full(nB, 0.5)])  # minority is boundary-uncertain
    sel = FS.select_budget(feats, base_p, budget=20, seed=0)
    frac_minority = np.mean(labels[sel] != COG)
    # random would give ~0.2 minority; uncertainty sampling should pull in far more
    assert frac_minority > 0.4


def test_larger_budget_non_decreasing_balanced_accuracy():
    samples, feats, labels = _target_set(n_frames=24, seed=2)
    accs = []
    for b in (4, 12, 24):
        r = FS.run_fewshot("m", feats, labels, samples, COG, adapter="prototype",
                           budget=b, seed=2, write=False)
        accs.append(r.balanced_accuracy)
    # separable data: more budget should not hurt (allow tiny noise tolerance)
    assert accs[1] >= accs[0] - 0.05 and accs[2] >= accs[1] - 0.05
    assert accs[-1] >= 0.9


def test_all_adapters_run_and_separate_well():
    samples, feats, labels = _target_set(seed=3)
    for ad in FS.ADAPTERS:
        r = FS.run_fewshot("m", feats, labels, samples, COG, adapter=ad, budget=16,
                           seed=3, write=False)
        assert r.status == "ok" and r.balanced_accuracy >= 0.8, ad


def test_unknown_adapter_raises():
    samples, feats, labels = _target_set()
    import pytest
    with pytest.raises(ValueError):
        FS.adapter_predict("bogus", feats[:4], labels[:4], feats[4:8], COG)


def test_writes_row_to_results_dir(tmp_path):
    samples, feats, labels = _target_set()
    row = FS.run_fewshot("dinov2", feats, labels, samples, COG, budget=12,
                         results_dir=tmp_path, write=True)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1 and C.read_result(files[0]).eval_setting == C.EVAL_FEWSHOT
