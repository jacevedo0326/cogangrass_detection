"""Orchestrator (U11): the single script the user runs to sweep every cell.

Enumerates the job matrix from ``configs/sweep.yaml``, computes each cell's deterministic
``job_id``, **skips any whose result already exists** (resume after a shutdown), packs the
rest into VRAM-budgeted batches (KTD6 — cheap frozen jobs pack densely; heavy LoRA/full/SSL
jobs reserve a larger slice and never oversubscribe), runs each as an **isolated subprocess**
(a crash/OOM in one writes its own failed/oom row and never aborts the batch, KTD8), and ends
with the merged ranked report + a coverage/failure summary.

Each child is one of the per-model scripts, which writes its own crash-safe
``results/<job_id>.jsonl``. The orchestrator only writes a *fallback* row when a child dies
hard without leaving one, so coverage accounting stays honest.

Run on the Spark (stop other GPU tenants first; KTD6):
    python arch_sweep/run_all.py                       # full matrix, resumable
    python arch_sweep/run_all.py --budget-mb 60000     # tighter VRAM budget
    python arch_sweep/run_all.py --dry-run             # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backbones as B  # noqa: E402
import common as C  # noqa: E402
import report as REP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_MB = 120_000          # single-tenant GB10 (~120 GB) once other tenants are stopped
DEFAULT_SWEEP = Path(__file__).resolve().parent / "configs" / "sweep.yaml"

DEFAULTS = {"variant": "reference", "head": "mlp_bn", "tuning_mode": "frozen",
            "adaptation": "none", "seed": C.DEFAULT_SEED}

# Rough per-cell VRAM reservations (MB). Tuned later from the U3 fit gate (plan Open Question).
BASE_MB = {"resnet18": 2_000, "dinov2": 3_000, "plantclef": 4_000, "siglip2": 4_000,
           "dinov3_s": 3_000, "dinov3_b": 5_000, "dinov3_l": 9_000, "dinov3_sat": 9_000,
           "aimv2": 9_000, "cradio": 7_000}
MODE_MULT = {"frozen": 1.0, "lora": 2.5, "full": 4.0}


def footprint_mb(resolved_model: str, tuning_mode: str) -> int:
    return int(BASE_MB.get(resolved_model, 6_000) * MODE_MULT.get(tuning_mode, 1.0))


@dataclass
class Job:
    script: str                 # e.g. "models/train_dinov3.py"
    args: list                  # CLI ablation args
    identity: dict              # exactly what the child's ResultRow will carry
    mb: int
    extra_args: list = field(default_factory=list)   # e.g. ["--size", "l"]

    @property
    def job_id(self) -> str:
        return C.job_id(self.identity)

    @property
    def name(self) -> str:
        base = self.identity["model"]
        tags = [self.identity["variant"], self.identity["tuning_mode"], self.identity["adaptation"]]
        return base + " [" + ",".join(t for t in tags if t not in ("reference", "frozen", "none")) + "]" \
            if any(t not in ("reference", "frozen", "none") for t in tags) else base

    def command(self, python: str) -> list:
        return [python, str(REPO_ROOT / "arch_sweep" / self.script), *self.extra_args, *self.args]


def _make_job(model, *, size=None, variant=None, head=None, tuning_mode=None,
              adaptation=None, seed=None) -> Job:
    """Build one Job, resolving DINOv3 size to its backbone name (matching what the child writes)."""
    variant = variant or DEFAULTS["variant"]
    head = head or DEFAULTS["head"]
    tuning_mode = tuning_mode or DEFAULTS["tuning_mode"]
    adaptation = adaptation or DEFAULTS["adaptation"]
    seed = int(seed if seed is not None else DEFAULTS["seed"])

    extra_args, extra = [], ""
    resolved = model
    if model == "dinov3":
        if size is None:
            raise ValueError("dinov3 job requires a size (s/b/l/sat)")
        resolved = B.dinov3_name(size)
        extra = f"size={size}"
        extra_args = ["--size", str(size)]

    identity = {"model": resolved, "variant": variant, "tuning_mode": tuning_mode,
                "head": head, "adaptation": adaptation, "eval_setting": C.EVAL_CROSS,
                "seed": seed, "extra": extra}
    args = ["--variant", variant, "--head", head, "--mode", tuning_mode,
            "--adaptation", adaptation, "--seed", str(seed)]
    return Job(script=f"models/train_{model}.py", args=args, identity=identity,
               mb=footprint_mb(resolved, tuning_mode), extra_args=extra_args)


def _expand_spec(spec: dict) -> list[Job]:
    """Expand one sweep entry into Jobs. List-valued keys become a cross-product (a grid)."""
    import itertools

    axes = {k: (v if isinstance(v, list) else [v]) for k, v in spec.items()}
    keys = list(axes)
    jobs = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        d = dict(zip(keys, combo))
        jobs.append(_make_job(
            d["model"], size=d.get("size"), variant=d.get("variant"), head=d.get("head"),
            tuning_mode=d.get("tuning_mode"), adaptation=d.get("adaptation"), seed=d.get("seed")))
    return jobs


def enumerate_jobs(sweep: dict) -> list[Job]:
    """Flatten a sweep config into a deduplicated list of Jobs (stable order)."""
    seen, out = set(), []
    for spec in sweep.get("jobs", []):
        for job in _expand_spec(spec):
            if job.job_id not in seen:
                seen.add(job.job_id)
                out.append(job)
    return out


def load_sweep(path: str | Path) -> dict:
    import json

    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# VRAM-budgeted batching (pure — testable without a GPU). First-fit-decreasing.
# ---------------------------------------------------------------------------
def plan_batches(jobs: list[Job], budget_mb: int) -> list[list[Job]]:
    """Pack jobs into sequential batches each summing to ≤ budget (an oversized job runs alone).

    Within a batch every job runs concurrently, so the per-batch sum is the peak reserved VRAM
    — keeping it ≤ budget is the no-oversubscription guarantee (KTD6).
    """
    batches: list[list[Job]] = []
    sums: list[int] = []
    for job in sorted(jobs, key=lambda j: -j.mb):
        placed = False
        for i, s in enumerate(sums):
            if s + job.mb <= budget_mb:
                batches[i].append(job)
                sums[i] += job.mb
                placed = True
                break
        if not placed:
            batches.append([job])
            sums.append(job.mb)
    return batches


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _log_has_oom(logpath: Path) -> bool:
    try:
        return "out of memory" in logpath.read_text().lower()
    except OSError:
        return False


def finalize(job: Job, returncode: int, logpath: Path, results_dir=C.RESULTS_DIR) -> str:
    """After a child exits, ensure a row exists. Trust the child's row; else write a fallback.

    The per-model script writes its own ok/failed/oom row even on handled exceptions; only a
    hard crash (segfault / OOM-kill / timeout) leaves none — that's when we record a fallback
    so the cell is never silently missing from coverage (KTD8).
    """
    if C.result_exists(job.identity, results_dir):
        return "ok-or-recorded"
    status = "oom" if (returncode != 0 and _log_has_oom(logpath)) else "failed"
    row = C.ResultRow(**job.identity, status=status,
                      error=f"child exited {returncode} with no result row; see {logpath.name}")
    C.write_result_atomic(row, results_dir)
    return status


def run_batch(batch: list[Job], *, python: str, log_dir: Path, timeout: int,
              results_dir=C.RESULTS_DIR) -> None:
    """Launch every job in the batch as an isolated subprocess, then wait + finalize each."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME", "/home/josh/hf_cache")}
    procs = []
    for job in batch:
        logpath = log_dir / f"{job.job_id}.log"
        fh = open(logpath, "w")
        p = subprocess.Popen(job.command(python), cwd=REPO_ROOT, env=env, stdout=fh, stderr=fh)
        procs.append((job, p, fh, logpath))
        print(f"  launched {job.name}  (job {job.job_id}, ~{job.mb}MB) -> {logpath.name}", flush=True)
    for job, p, fh, logpath in procs:
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = 124
        fh.close()
        status = finalize(job, rc, logpath, results_dir)
        print(f"  finished {job.name}  rc={rc}  [{status}]", flush=True)


def run_all(sweep: dict, *, budget_mb=DEFAULT_BUDGET_MB, python=sys.executable, dry_run=False,
            timeout=14_400, results_dir=C.RESULTS_DIR, log_dir=None) -> dict:
    """Enumerate → resume-skip → batch → run → report. Returns a summary dict."""
    log_dir = Path(log_dir) if log_dir else (Path(results_dir) / "logs")
    all_jobs = enumerate_jobs(sweep)
    todo = [j for j in all_jobs if not C.result_exists(j.identity, results_dir)]
    skipped = len(all_jobs) - len(todo)
    batches = plan_batches(todo, budget_mb)
    print(f"jobs: {len(all_jobs)} total · {skipped} already done (resume) · {len(todo)} to run "
          f"in {len(batches)} batch(es) under {budget_mb}MB", flush=True)
    for i, batch in enumerate(batches, 1):
        names = ", ".join(j.name for j in batch)
        print(f"\n== batch {i}/{len(batches)} ({sum(j.mb for j in batch)}MB): {names} ==", flush=True)
        if not dry_run:
            run_batch(batch, python=python, log_dir=log_dir, timeout=timeout, results_dir=results_dir)

    rows = C.read_all_results(results_dir)
    summary = {"total_jobs": len(all_jobs), "skipped": skipped, "ran": 0 if dry_run else len(todo),
               "coverage": REP.coverage(rows, expected=len(all_jobs))}
    if not dry_run:
        report = REP.render(rows, bar=0.817, expected=len(all_jobs))
        out = Path(results_dir) / "sweep_report.md"
        out.write_text(report)
        print("\n" + report)
        print(f"\nwrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="arch_sweep orchestrator — parallel, resumable")
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    ap.add_argument("--budget-mb", type=int, default=DEFAULT_BUDGET_MB)
    ap.add_argument("--timeout", type=int, default=14_400, help="per-job wall-clock seconds")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = ap.parse_args()
    run_all(load_sweep(args.sweep), budget_mb=args.budget_mb, dry_run=args.dry_run,
            timeout=args.timeout)


if __name__ == "__main__":
    main()
