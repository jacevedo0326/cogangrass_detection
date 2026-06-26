"""U12 tests: merge per-job rows, separate tables, win flag, recall-guarded best-of."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import report as R  # noqa: E402


def _row(model, bacc, *, status="ok", setting=C.EVAL_CROSS, cog=0.8, notr=0.8, budget=None, error=""):
    return C.ResultRow(model=model, status=status, eval_setting=setting,
                       balanced_accuracy=bacc, recall_cogongrass=cog, recall_not_cogongrass=notr,
                       budget=budget, error=error)


def test_merge_globs_per_job_files(tmp_path):
    for r in [_row("resnet18", 0.80), _row("dinov2", 0.83), _row("siglip2", 0.79)]:
        C.write_result_atomic(r, tmp_path)
    rows = C.read_all_results(tmp_path)
    assert len(rows) == 3
    cross, few, bad = R.split_rows(rows)
    assert len(cross) == 3 and not few and not bad


def test_failed_job_shows_in_coverage_not_dropped(tmp_path):
    C.write_result_atomic(_row("resnet18", 0.80), tmp_path)
    C.write_result_atomic(_row("aimv2", None, status="oom", error="CUDA out of memory"), tmp_path)
    C.write_result_atomic(_row("cradio", None, status="failed", error="boom"), tmp_path)
    rows = C.read_all_results(tmp_path)
    cov = R.coverage(rows)
    assert cov == {"total": 3, "ok": 1, "failed": 1, "oom": 1}
    report = R.render(rows, bar=0.817)
    assert "Failed / incomplete cells" in report and "CUDA out of memory"[:20] in report


def test_cross_and_fewshot_tables_are_separate_baselines_only_in_cross():
    rows = [_row("dinov2", 0.83), _row("dinov2", 0.88, setting=C.EVAL_FEWSHOT, budget=40)]
    report = R.render(rows, bar=0.817)
    assert "Cross-collection" in report and "Few-shot target adaptation" in report
    # baselines appear once, in the cross-collection table region only
    cross_region = report.split("Few-shot")[0]
    few_region = report.split("Few-shot")[1]
    assert "Stage-1 DA" in cross_region and "Stage-1 DA" not in few_region


def test_win_flag_only_above_bar():
    rows = [_row("winner", 0.83), _row("loser", 0.79)]
    report = R.render(rows, bar=0.817)
    assert R.beats_baseline(rows[0], 0.817) is True
    assert R.beats_baseline(rows[1], 0.817) is False
    # the winner's row carries the ✅ flag; the loser's does not
    win_line = [ln for ln in report.splitlines() if "winner" in ln][0]
    lose_line = [ln for ln in report.splitlines() if "loser" in ln][0]
    assert "✅" in win_line and "✅" not in lose_line


def test_best_of_recall_guard_does_not_crown_collapsed_cell():
    # higher balanced accuracy but collapsed cogongrass recall -> must not be crowned
    collapsed = _row("sneaky", 0.84, cog=0.20, notr=0.99)
    honest = _row("solid", 0.82, cog=0.70, notr=0.90)
    champ = R.best_of([collapsed, honest], bar=0.817)
    assert champ is not None and champ.model == "solid"
    # and a win flag is not awarded to the collapsed cell
    assert R.beats_baseline(collapsed, 0.817) is False
    assert R.beats_baseline(honest, 0.817) is True


def test_best_of_none_when_all_collapsed():
    rows = [_row("a", 0.84, cog=0.1), _row("b", 0.83, cog=0.2)]
    assert R.best_of(rows, bar=0.817) is None
    report = R.render(rows, bar=0.817)
    assert "Best-of:** none" in report


def test_render_writes_baselines_and_ranking_order():
    rows = [_row("low", 0.76), _row("high", 0.83), _row("mid", 0.80)]
    report = R.render(rows, bar=0.817)
    order = [ln for ln in report.splitlines() if ln.startswith("| 1 ") or ln.startswith("| 2 ")
             or ln.startswith("| 3 ")]
    assert "high" in order[0] and "mid" in order[1] and "low" in order[2]
    assert "0.804" in report and "0.817" in report   # both baselines present
