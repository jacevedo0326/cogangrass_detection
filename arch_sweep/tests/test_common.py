"""U1 tests: split is leakage-free, metrics honest, job-id deterministic, writes crash-safe."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402


# --- synthetic ImageFolder-style samples (path, label_idx); cog_idx = 0 -------
def _samples():
    return [
        ("tiles_dataset/cogongrass/DJI_20260422_0001_r0_c0.jpg", 0),
        ("tiles_dataset/cogongrass/DJI_20260422_0001_r0_c1.jpg", 0),
        ("tiles_dataset/not_cogongrass/DJI_20260422_0002_r1_c0.jpg", 1),
        ("tiles_dataset/cogongrass/DJI_20260606_0010_r0_c0.jpg", 0),
        ("tiles_dataset/not_cogongrass/DJI_20260606_0011_r2_c3.jpg", 1),
        ("tiles_dataset/not_cogongrass/DJI_20260606_0011_r2_c4.jpg", 1),
    ]


# --- split / leakage ----------------------------------------------------------
def test_frame_and_date_parsing():
    assert C.frame_of("tiles_dataset/cogongrass/DJI_20260422_0001_r0_c1.jpg") == "DJI_20260422_0001"
    assert C.date_of("DJI_20260422_0001") == "20260422"
    assert C.date_of("weird_name") == "other"


def test_no_leakage_between_collections():
    s = _samples()
    te = set(C.indices_for_date(s, C.TEST_DATE))
    tr = set(C.indices_for_date(s, C.TRAIN_DATE))
    assert te and tr
    assert te.isdisjoint(tr), "0422 and 0606 index sets must be disjoint"
    assert C.frames_for_date(s, C.TEST_DATE).isdisjoint(C.frames_for_date(s, C.TRAIN_DATE))
    assert all(C.date_of(C.frame_of(s[i][0])) == C.TEST_DATE for i in te)


def test_split_by_collection_is_frame_grouped_and_disjoint():
    s = _samples()
    tr, va, te, (nf_tr, nf_va, nf_te) = C.split_by_collection(s, cog_idx=0)
    # test slice is exactly the 0422 indices
    assert set(te) == set(C.indices_for_date(s, C.TEST_DATE))
    # train/val/test index sets are mutually disjoint (no tile in two slices)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    # no frame spans train and val (frame-grouped, no leakage)
    tr_frames = {C.frame_of(s[i][0]) for i in tr}
    va_frames = {C.frame_of(s[i][0]) for i in va}
    assert tr_frames.isdisjoint(va_frames)
    assert nf_te == 2  # two distinct 0422 frames


def test_balance_downsamples_majority():
    import random
    s = _samples()
    idx = list(range(len(s)))   # 2 cog (0,1) + ... cog_idx=0 -> pos = idx 0,1,3 ; neg = 2,4,5
    out = C.balance(idx, s, cog_idx=0, rng=random.Random(0))
    pos = [i for i in out if s[i][1] == 0]
    neg = [i for i in out if s[i][1] != 0]
    assert len(pos) == len(neg)


# --- metrics ------------------------------------------------------------------
def test_metric_perfect_separation():
    y = [1, 1, 0, 0]
    scores = [0.95, 0.80, 0.10, 0.05]
    pred = [1, 1, 0, 0]
    assert C.balanced_accuracy(y, pred) == 1.0
    assert C.auroc(y, scores) == 1.0
    assert C.average_precision(y, scores) == 1.0
    rec = C.per_class_recall(y, pred)
    assert rec["cogongrass"] == 1.0 and rec["not_cogongrass"] == 1.0


def test_balanced_accuracy_not_fooled_by_majority_class():
    # 8 positives, 2 negatives; predict everything positive -> raw acc 0.8 but bal-acc 0.5
    y = [1] * 8 + [0] * 2
    pred = [1] * 10
    assert C.balanced_accuracy(y, pred) == 0.5
    rec = C.per_class_recall(y, pred)
    assert rec["cogongrass"] == 1.0 and rec["not_cogongrass"] == 0.0


def test_f2_weights_recall():
    y = [1, 1, 1, 0]
    scores = [0.9, 0.4, 0.3, 0.1]
    rows = C.f2_sweep(y, scores, thresholds=[0.5, 0.2])
    assert rows[0]["fn"] == 2 and rows[0]["n_cog"] == 3
    assert rows[1]["fn"] == 0
    assert rows[1]["recall"] == pytest.approx(1.0)


def test_pick_threshold_is_pure_and_only_uses_given_scores():
    y = [1, 1, 1, 0, 0]
    scores_0606 = [0.8, 0.35, 0.30, 0.20, 0.05]
    thr = C.pick_threshold_on(y, scores_0606)
    assert 0.0 <= thr <= 1.0
    pred = [1 if sc >= thr else 0 for sc in scores_0606]
    assert pred[:3] == [1, 1, 1]   # F2 prioritizes recall -> recover all cogongrass
    assert C.pick_threshold_on(y, scores_0606) == thr   # deterministic, no hidden state


# --- job identity -------------------------------------------------------------
def test_job_id_deterministic_and_config_insensitive_to_metrics():
    cfg = {"model": "dinov2", "variant": "tile512_clahe", "tuning_mode": "frozen"}
    jid1 = C.job_id(cfg)
    jid2 = C.job_id(dict(cfg))
    assert jid1 == jid2
    # a minimal planned config and a full row for the same cell hash identically
    row = C.ResultRow(model="dinov2", variant="tile512_clahe", tuning_mode="frozen",
                      balanced_accuracy=0.83, status="ok")
    assert row.job_id == jid1


def test_job_id_distinguishes_cells():
    base = {"model": "dinov2", "variant": "reference"}
    assert C.job_id(base) != C.job_id({**base, "adaptation": "adabn"})
    assert C.job_id(base) != C.job_id({**base, "eval_setting": C.EVAL_FEWSHOT})
    assert C.job_id(base) != C.job_id({**base, "seed": 7})


def test_same_seed_same_metric_determinism():
    C.set_global_seed(123)
    import random
    a = [random.random() for _ in range(5)]
    C.set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


# --- crash-safe writer + resume -----------------------------------------------
def test_write_result_round_trip(tmp_path):
    row = C.ResultRow(model="resnet18", variant="reference", tuning_mode="frozen",
                      head="mlp_bn", adaptation="none", eval_setting=C.EVAL_CROSS, seed=42,
                      status="ok", balanced_accuracy=0.81, recall_cogongrass=0.9,
                      recall_not_cogongrass=0.72, auroc=0.88, threshold=0.3,
                      f2_sweep=[{"thr": 0.5, "recall": 0.9}], n_test=7006, n_cog_test=6000)
    assert not C.result_exists(row, tmp_path)
    path = C.write_result_atomic(row, tmp_path)
    assert C.result_exists(row, tmp_path)
    back = C.read_result(path)
    assert back == row
    assert back.eval_setting == C.EVAL_CROSS and back.seed == 42 and back.status == "ok"


def test_write_is_atomic_no_partial_final_on_interrupt(tmp_path, monkeypatch):
    row = C.ResultRow(model="dinov2", variant="reference")
    final = C.result_path(row, tmp_path)

    def boom(src, dst):
        raise KeyboardInterrupt("interrupted before rename")

    monkeypatch.setattr(C.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        C.write_result_atomic(row, tmp_path)
    # the final file must not exist (only an atomic rename creates it) and no temp litter
    assert not final.exists()
    assert not list(tmp_path.glob(".tmp-*")), "temp file must be cleaned up on interrupt"


def test_read_all_results_merges_per_job_files(tmp_path):
    rows = [
        C.ResultRow(model="resnet18", balanced_accuracy=0.80),
        C.ResultRow(model="dinov2", balanced_accuracy=0.83),
        C.ResultRow(model="dinov2", adaptation="adabn", balanced_accuracy=0.85),
    ]
    for r in rows:
        C.write_result_atomic(r, tmp_path)
    merged = C.read_all_results(tmp_path)
    assert len(merged) == 3
    assert {r.job_id for r in merged} == {r.job_id for r in rows}


def test_read_all_results_skips_empty_partial_file(tmp_path):
    C.write_result_atomic(C.ResultRow(model="resnet18"), tmp_path)
    (tmp_path / "deadbeef.jsonl").write_text("")   # a half-born/empty file
    merged = C.read_all_results(tmp_path)
    assert len(merged) == 1


def test_written_row_is_valid_json_line(tmp_path):
    path = C.write_result_atomic(C.ResultRow(model="siglip2"), tmp_path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    json.loads(lines[0])   # parses without error


# --- per-tile confidence sidecar (U1) ----------------------------------------
def _score_recs():
    paths = ["tiles/cogongrass/DJI_20260422_0001_r0_c0.jpg",
             "tiles/not_cogongrass/DJI_20260422_0001_r0_c1.jpg",
             "tiles/not_cogongrass/DJI_20260422_0002_r1_c0.jpg"]
    return C.build_score_records(paths, y_true_cog=[1, 0, 0], scores=[0.92, 0.04, 0.61])


def test_build_score_records_derives_frame_and_class():
    recs = _score_recs()
    assert len(recs) == 3
    assert recs[0].frame == "DJI_20260422_0001" and recs[0].true_label == "cogongrass"
    assert recs[1].true_label == "not_cogongrass"
    assert recs[2].frame == "DJI_20260422_0002" and recs[2].p_cogongrass == 0.61


def test_scores_sidecar_round_trip(tmp_path):
    cfg = {"model": "dinov2", "variant": "reference"}
    recs = _score_recs()
    path = C.write_scores_atomic(cfg, recs, tmp_path)
    assert path.name.endswith(C.SCORES_SUFFIX)
    assert path == C.scores_path(cfg, tmp_path)
    back = C.read_scores(path)
    assert back == recs            # identical round-trip
    assert all(0.0 <= r.p_cogongrass <= 1.0 for r in back)


def test_scores_sidecar_is_atomic_no_partial_on_interrupt(tmp_path, monkeypatch):
    cfg = {"model": "dinov2", "variant": "reference"}
    final = C.scores_path(cfg, tmp_path)

    def boom(src, dst):
        raise KeyboardInterrupt("interrupted before rename")

    monkeypatch.setattr(C.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        C.write_scores_atomic(cfg, _score_recs(), tmp_path)
    assert not final.exists()
    assert not list(tmp_path.glob(".tmp-*")), "temp sidecar must be cleaned up on interrupt"


def test_read_all_results_ignores_scores_sidecars(tmp_path):
    # a result row + its scores sidecar share the dir and both match *.jsonl; the merge that
    # feeds the report must return only the result row, never parse the sidecar as a row.
    row = C.ResultRow(model="dinov2", variant="reference", balanced_accuracy=0.83)
    C.write_result_atomic(row, tmp_path)
    C.write_scores_atomic(row, _score_recs(), tmp_path)
    merged = C.read_all_results(tmp_path)
    assert len(merged) == 1 and merged[0].job_id == row.job_id
