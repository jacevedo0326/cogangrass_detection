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

# Bootstrap iterations for the frame-resampled CIs (U2). Lower in tests via the render arg.
N_BOOT = 1000


def _sidecar_scores(row: C.ResultRow, results_dir):
    """Load a row's per-tile (frames, y_true_cog, scores) from its U1 sidecar, or None.

    ``relabel`` (a {path: corrected_true_label} map) overrides the stored labels so the same
    scores can be re-scored against a cleaned ground truth (U3 clean-vs-noisy) — see callers.
    """
    if results_dir is None:
        return None
    path = C.scores_path(row.identity(), results_dir)
    if not path.exists():
        return None
    recs = C.read_scores(path)
    if not recs:
        return None
    return recs


def _y_p_frames(recs, relabel=None):
    """Turn ScoreRecords into aligned (frames, y_true_cog, scores), optional label override."""
    frames = [r.frame for r in recs]
    p = [r.p_cogongrass for r in recs]
    y = []
    for r in recs:
        lab = relabel.get(r.path, r.true_label) if relabel else r.true_label
        y.append(1 if lab == C.COG_CLASS else 0)
    return frames, y, p


def row_ci(row, results_dir, *, n_boot=N_BOOT, relabel=None):
    """Frame-resampled balanced-accuracy CI for a row from its sidecar, or None if absent."""
    recs = _sidecar_scores(row, results_dir)
    if recs is None:
        return None
    frames, y, p = _y_p_frames(recs, relabel)
    if len(set(y)) < 2:
        return None
    return C.balanced_accuracy_ci(frames, y, p, n_boot=n_boot)


def clean_balanced_accuracy(row, results_dir, relabel):
    """Re-score a row's stored scores against a cleaned label map (U3) -> balanced accuracy.

    Same per-tile scores, corrected ground truth: the clean column the report shows next to
    the raw number. Returns None if the row has no sidecar.
    """
    recs = _sidecar_scores(row, results_dir)
    if recs is None:
        return None
    _frames, y, p = _y_p_frames(recs, relabel)
    if len(set(y)) < 2:
        return None
    pred = [1 if pc >= 0.5 else 0 for pc in p]
    return C.balanced_accuracy(y, pred)


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


def ci_win(r: C.ResultRow, bar: float, ci) -> bool:
    """A *decisive* win (KTD1): recall not collapsed, and the CI lower bound clears the bar.

    When a frame-resampled CI is available the bar is the CI's lower bound (non-overlapping
    with the baseline point); without a sidecar/CI it falls back to the point-estimate rule so
    a partial run still flags candidates.
    """
    if (r.recall_cogongrass or 0) < MIN_COG_RECALL:
        return False
    if ci is None:
        return beats_baseline(r, bar)
    return C.ci_separated(ci, bar)


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


def _ci_str(ci):
    return "—" if ci is None else f"[{ci[0]:.3f}, {ci[2]:.3f}]"


def render(rows, bar: float, expected: int | None = None, *, results_dir=None,
           n_boot: int = N_BOOT, relabel=None) -> str:
    cross, few, bad = split_rows(rows)
    cov = coverage(rows, expected)
    L = ["# arch_sweep — traditional-ML architecture sweep results", ""]

    # Coverage first so a partial run can't be misread as complete.
    cov_line = (f"**Coverage:** {cov['ok']} ok · {cov['failed']} failed · {cov['oom']} oom "
                f"· {cov['total']} rows")
    if "missing" in cov:
        cov_line += f" · {cov['missing']} missing of {cov['expected']} expected"
    L += [cov_line, ""]

    # Frame-resampled CIs per cross row (U2) from the U1 sidecars, computed once and reused.
    cis = {r.job_id: row_ci(r, results_dir, n_boot=n_boot) for r in cross}
    has_ci = any(v is not None for v in cis.values())
    # A win is decisive only when the CI lower bound clears the bar (KTD1); fall back to the
    # point estimate when no sidecar/CI is available so partial runs still rank.
    def is_win(r):
        ci = cis.get(r.job_id)
        return ci_win(r, bar, ci)

    clean = relabel is not None
    clean_col = " clean bacc |" if clean else ""
    clean_sep = ":---------:|" if clean else ""

    # Cross-collection ranking (headline), with the baseline rows interleaved by score.
    L += ["## Cross-collection (train 0606 → held-out 0422)", "",
          f"| # | cell | bal acc | 95% CI (frame) |{clean_col} recall cog | recall not | AUROC | AP | op-thr | vs 0.817 |",
          f"|---|------|:-------:|:--------------:|{clean_sep}:----------:|:----------:|:-----:|:--:|:------:|:--------:|"]
    ranked = rank(cross)
    baseline_rows = [(name, b) for name, b in C.BASELINES]
    i = 0
    for r in ranked:
        i += 1
        flag = "✅" if is_win(r) else ""
        delta = "" if r.balanced_accuracy is None else f"{r.balanced_accuracy - bar:+.3f}"
        clean_cell = ""
        if clean:
            clean_cell = f" {_f(clean_balanced_accuracy(r, results_dir, relabel))} |"
        L.append(f"| {i} | {cell_name(r)} | **{_f(r.balanced_accuracy)}** | "
                 f"{_ci_str(cis.get(r.job_id))} |{clean_cell} "
                 f"{_f(r.recall_cogongrass)} | {_f(r.recall_not_cogongrass)} | {_f(r.auroc)} | "
                 f"{_f(r.average_precision)} | {_f(r.threshold)} | {delta} {flag} |")
    for name, b in baseline_rows:
        clean_b = " — |" if clean else ""
        L.append(f"| · | _{name}_ | _{_f(b)}_ | — |{clean_b} — | — | — | — | — | _baseline_ |")
    L.append("")
    if has_ci:
        L += ["_95% CI is frame-resampled (tiles within a frame move together); a ✅ requires "
              "the CI lower bound to clear the bar — not just the point estimate (KTD1)._", ""]

    # Best-of suggestion (recall-guarded), with its frame-resampled CI verdict.
    champ = best_of(cross, bar)
    if champ:
        champ_ci = cis.get(champ.job_id)
        verdict = ("**beats** the 0.817 bar with a separated CI" if is_win(champ)
                   else ("beats the point estimate but the CI overlaps the bar"
                         if (champ.balanced_accuracy or 0) > bar else "does **not** beat 0.817 yet"))
        L += [f"**Best-of:** `{cell_name(champ)}` at balanced accuracy "
              f"**{_f(champ.balanced_accuracy)}** (95% CI {_ci_str(champ_ci)}, "
              f"cog recall {_f(champ.recall_cogongrass)}) — {verdict}.", ""]
        # Per-frame view of the champion (is failure a few bad frames or systematic?).
        recs = _sidecar_scores(champ, results_dir)
        if recs is not None:
            frames, y, p = _y_p_frames(recs, relabel)
            pf = C.per_frame_metrics(frames, y, [1 if pc >= 0.5 else 0 for pc in p])
            L += [f"### Per-frame breakdown — best-of (`{cell_name(champ)}`)", "",
                  "| frame | tiles | cog tiles | recall cog | bal acc |",
                  "|-------|:-----:|:---------:|:----------:|:-------:|"]
            for fr in pf:
                L.append(f"| {fr['frame']} | {fr['n']} | {fr['n_cog']} | "
                         f"{_f(fr['recall_cog'])} | {_f(fr['bacc'])} |")
            L.append("")
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
    report = render(rows, args.bar, _expected_from_sweep(args.sweep), results_dir=args.results)
    print(report)
    out = args.out or str(Path(args.results) / "sweep_report.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
