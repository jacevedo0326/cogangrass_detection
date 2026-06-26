"""Merge per-job results into one ranked read against the baselines (U12).

Globs ``results/*.jsonl`` (each the crash-safe output of one cell, KTD8), merges them, and
renders:

- the **cross-collection** ranking (the headline 0606->0422 numbers) with the **0.804**
  (ResNet18) and **0.817** (Stage-1 DA) baseline rows,
- a separate **few-shot** table (KTD4 — never blended into the cross-collection ranking),
- a **best-of** suggestion (winning variant × backbone × tuning × adaptation) that refuses
  to crown a cell whose cogongrass recall has collapsed,
- a **coverage + failure summary** (ok / failed / oom, and missing when a sweep config is
  given) so a partial sweep is never misread as complete.

Writes ``results/sweep_report.md`` and prints the same. Pattern: ``vlm_zeroshot/report.py``.

Run:  python arch_sweep/report.py [--results DIR] [--bar 0.817] [--sweep configs/sweep.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

# A cell with cogongrass recall below this is "collapsed" — never crowned best-of even if its
# balanced accuracy looks competitive (false negatives are the costly error; CLAUDE.md).
MIN_COG_RECALL = 0.5


def split_rows(rows):
    """Partition rows into (cross_collection ok, few_shot ok, non-ok) buckets."""
    cross, few, bad = [], [], []
    for r in rows:
        if r.status != "ok" or r.balanced_accuracy is None:
            bad.append(r)
        elif r.eval_setting == C.EVAL_FEWSHOT:
            few.append(r)
        else:
            cross.append(r)
    return cross, few, bad


def cell_name(r: C.ResultRow) -> str:
    """Human label for a cell: backbone[/extra] · variant · mode · head · adaptation."""
    base = r.model + (f"/{r.extra}" if r.extra else "")
    bits = [base, r.variant]
    if r.tuning_mode != "frozen":
        bits.append(r.tuning_mode)
    if r.head != "mlp_bn":
        bits.append(r.head)
    if r.adaptation != "none":
        bits.append(r.adaptation)
    return " · ".join(bits)


def rank(rows) -> list[C.ResultRow]:
    return sorted(rows, key=lambda r: (r.balanced_accuracy is None, -(r.balanced_accuracy or 0)))


def beats_baseline(r: C.ResultRow, bar: float) -> bool:
    """A real win: balanced accuracy strictly above the bar AND cog recall not collapsed."""
    return (r.balanced_accuracy is not None and r.balanced_accuracy > bar
            and (r.recall_cogongrass or 0) >= MIN_COG_RECALL)


def best_of(cross_rows, bar: float) -> C.ResultRow | None:
    """Highest balanced accuracy among non-collapsed cross-collection cells (recall guard)."""
    eligible = [r for r in cross_rows if (r.recall_cogongrass or 0) >= MIN_COG_RECALL]
    return rank(eligible)[0] if eligible else None


def coverage(rows, expected: int | None = None) -> dict:
    ok = [r for r in rows if r.status == "ok"]
    failed = [r for r in rows if r.status == "failed"]
    oom = [r for r in rows if r.status == "oom"]
    out = {"total": len(rows), "ok": len(ok), "failed": len(failed), "oom": len(oom)}
    if expected is not None:
        out["expected"] = expected
        out["missing"] = max(0, expected - len(rows))
    return out


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render(rows, bar: float, expected: int | None = None) -> str:
    cross, few, bad = split_rows(rows)
    cov = coverage(rows, expected)
    L = ["# arch_sweep — traditional-ML architecture sweep results", ""]

    # Coverage first so a partial run can't be misread as complete.
    cov_line = (f"**Coverage:** {cov['ok']} ok · {cov['failed']} failed · {cov['oom']} oom "
                f"· {cov['total']} rows")
    if "missing" in cov:
        cov_line += f" · {cov['missing']} missing of {cov['expected']} expected"
    L += [cov_line, ""]

    # Cross-collection ranking (headline), with the baseline rows interleaved by score.
    L += ["## Cross-collection (train 0606 → held-out 0422)", "",
          "| # | cell | bal acc | recall cog | recall not | AUROC | AP | op-thr | vs 0.817 |",
          "|---|------|:-------:|:----------:|:----------:|:-----:|:--:|:------:|:--------:|"]
    ranked = rank(cross)
    baseline_rows = [(name, b) for name, b in C.BASELINES]
    i = 0
    for r in ranked:
        i += 1
        flag = "✅" if beats_baseline(r, bar) else ""
        delta = "" if r.balanced_accuracy is None else f"{r.balanced_accuracy - bar:+.3f}"
        L.append(f"| {i} | {cell_name(r)} | **{_f(r.balanced_accuracy)}** | "
                 f"{_f(r.recall_cogongrass)} | {_f(r.recall_not_cogongrass)} | {_f(r.auroc)} | "
                 f"{_f(r.average_precision)} | {_f(r.threshold)} | {delta} {flag} |")
    for name, b in baseline_rows:
        L.append(f"| · | _{name}_ | _{_f(b)}_ | — | — | — | — | — | _baseline_ |")
    L.append("")

    # Best-of suggestion (recall-guarded).
    champ = best_of(cross, bar)
    if champ:
        verdict = ("**beats** the 0.817 bar" if beats_baseline(champ, bar)
                   else "does **not** beat 0.817 yet")
        L += [f"**Best-of:** `{cell_name(champ)}` at balanced accuracy "
              f"**{_f(champ.balanced_accuracy)}** (cog recall {_f(champ.recall_cogongrass)}) — {verdict}.", ""]
    else:
        L += ["**Best-of:** none — every cell collapsed cogongrass recall "
              f"(< {MIN_COG_RECALL}) or no cross-collection cells yet.", ""]

    # Few-shot table (separate; KTD4) — no baselines here.
    if few:
        L += ["## Few-shot target adaptation (separate eval setting — not comparable to above)", "",
              "| cell | budget | bal acc | recall cog | recall not | AUROC |",
              "|------|:------:|:-------:|:----------:|:----------:|:-----:|"]
        for r in rank(few):
            L.append(f"| {cell_name(r)} | {r.budget if r.budget is not None else '—'} | "
                     f"{_f(r.balanced_accuracy)} | {_f(r.recall_cogongrass)} | "
                     f"{_f(r.recall_not_cogongrass)} | {_f(r.auroc)} |")
        L.append("")

    # Failures, surfaced not dropped.
    if bad:
        L += ["## Failed / incomplete cells", "",
              "| cell | status | error |", "|------|:------:|-------|"]
        for r in bad:
            L.append(f"| {cell_name(r)} | {r.status} | {(r.error or '')[:100]} |")
        L.append("")
    return "\n".join(L)


def _expected_from_sweep(path: str | None) -> int | None:
    """Count enumerated jobs in a sweep config, if one is supplied (for the missing count)."""
    if not path or not Path(path).exists():
        return None
    try:
        import run_all  # local import; only needed when a sweep config is given
        return len(run_all.enumerate_jobs(run_all.load_sweep(path)))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Merge + rank arch_sweep results vs baselines")
    ap.add_argument("--results", default=str(C.RESULTS_DIR), help="dir of per-job *.jsonl")
    ap.add_argument("--bar", type=float, default=0.817, help="balanced-accuracy bar to flag wins")
    ap.add_argument("--sweep", default=None, help="optional sweep.yaml to compute a missing count")
    ap.add_argument("--out", default=None, help="markdown output (default results/sweep_report.md)")
    args = ap.parse_args()

    rows = C.read_all_results(args.results)
    report = render(rows, args.bar, _expected_from_sweep(args.sweep))
    print(report)
    out = args.out or str(Path(args.results) / "sweep_report.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
