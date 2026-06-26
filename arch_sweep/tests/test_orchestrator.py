"""U11 tests: job enumeration + deterministic ids, resume skip, budget batching, finalize."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import run_all as O  # noqa: E402


SWEEP = {"jobs": [
    {"model": "resnet18"},
    {"model": "siglip2"},
    {"model": "dinov3", "size": "l"},
]}


def test_enumeration_and_deterministic_job_ids():
    jobs = O.enumerate_jobs(SWEEP)
    assert {j.identity["model"] for j in jobs} == {"resnet18", "siglip2", "dinov3_l"}
    # dinov3 size resolves to backbone name + extra, matching what the child writes
    d = next(j for j in jobs if j.identity["model"] == "dinov3_l")
    assert d.identity["extra"] == "size=l" and d.extra_args == ["--size", "l"]
    # job_id is the SAME hash trainer would compute for that identity (resume correctness)
    assert d.job_id == C.job_id(d.identity)
    # stable + deduplicated
    assert [j.job_id for j in O.enumerate_jobs(SWEEP)] == [j.job_id for j in jobs]


def test_grid_expansion_cross_product():
    sweep = {"jobs": [{"model": ["siglip2", "cradio"], "adaptation": ["none", "adabn"]}]}
    jobs = O.enumerate_jobs(sweep)
    assert len(jobs) == 4
    assert {(j.identity["model"], j.identity["adaptation"]) for j in jobs} == {
        ("siglip2", "none"), ("siglip2", "adabn"), ("cradio", "none"), ("cradio", "adabn")}


def test_resume_skips_completed_jobs(tmp_path):
    jobs = O.enumerate_jobs(SWEEP)
    # pretend resnet18 already finished
    done = next(j for j in jobs if j.identity["model"] == "resnet18")
    C.write_result_atomic(C.ResultRow(**done.identity, status="ok", balanced_accuracy=0.8), tmp_path)
    summary = O.run_all(SWEEP, dry_run=True, results_dir=tmp_path, log_dir=tmp_path / "logs")
    assert summary["skipped"] == 1 and summary["total_jobs"] == 3


def test_plan_batches_never_exceeds_budget():
    jobs = [O.Job("s", [], {"model": f"m{i}", "variant": "reference", "tuning_mode": "frozen",
                            "head": "mlp_bn", "adaptation": "none", "eval_setting": C.EVAL_CROSS,
                            "seed": 42, "extra": ""}, mb=mb)
            for i, mb in enumerate([9000, 9000, 5000, 3000, 3000, 2000])]
    budget = 12000
    batches = O.plan_batches(jobs, budget)
    for b in batches:
        assert sum(j.mb for j in b) <= budget        # peak concurrent VRAM never oversubscribes
    assert sum(len(b) for b in batches) == len(jobs)  # every job scheduled exactly once


def test_oversized_job_runs_alone():
    jobs = [O.Job("s", [], {"model": "big", "variant": "reference", "tuning_mode": "full",
                            "head": "mlp_bn", "adaptation": "none", "eval_setting": C.EVAL_CROSS,
                            "seed": 42, "extra": ""}, mb=200000)]
    batches = O.plan_batches(jobs, budget_mb=120000)
    assert len(batches) == 1 and len(batches[0]) == 1   # can't fit, but still scheduled (alone)


def test_finalize_writes_fallback_oom_when_child_left_no_row(tmp_path):
    job = O.enumerate_jobs(SWEEP)[0]
    log = tmp_path / f"{job.job_id}.log"
    log.write_text("...\nRuntimeError: CUDA out of memory. Tried to allocate ...\n")
    status = O.finalize(job, returncode=1, logpath=log, results_dir=tmp_path)
    assert status == "oom"
    row = C.read_result(C.result_path(job.identity, tmp_path))
    assert row.status == "oom"


def test_finalize_trusts_existing_child_row(tmp_path):
    job = O.enumerate_jobs(SWEEP)[0]
    C.write_result_atomic(C.ResultRow(**job.identity, status="ok", balanced_accuracy=0.79), tmp_path)
    log = tmp_path / f"{job.job_id}.log"
    log.write_text("out of memory")   # even if the log looks scary, the child's row wins
    status = O.finalize(job, returncode=0, logpath=log, results_dir=tmp_path)
    assert status == "ok-or-recorded"
    assert C.read_result(C.result_path(job.identity, tmp_path)).status == "ok"


def test_dry_run_runs_nothing_but_plans(tmp_path):
    summary = O.run_all(SWEEP, dry_run=True, results_dir=tmp_path, log_dir=tmp_path / "logs")
    assert summary["ran"] == 0 and summary["total_jobs"] == 3
    assert not list(tmp_path.glob("*.jsonl"))   # nothing executed/written
