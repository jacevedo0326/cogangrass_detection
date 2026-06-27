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


# --- U2: CI column, per-frame view, CI-gated win flag, clean-vs-noisy ---------
def _write_sidecar(row, results_dir, *, frames, y, scores):
    recs = C.build_score_records(
        [f"tiles/x/{f}_r0_c{i}.jpg" for i, f in enumerate(frames)], y, scores)
    # overwrite the auto-derived frame with the supplied one (test fixtures use short ids)
    for rec, f in zip(recs, frames):
        rec.frame = f
    C.write_scores_atomic(row.identity(), recs, results_dir)


def test_ci_column_and_per_frame_view_rendered_from_sidecar(tmp_path):
    row = _row("dinov2", 1.0)
    C.write_result_atomic(row, tmp_path)
    _write_sidecar(row, tmp_path, frames=["F1", "F1", "F2", "F2"],
                   y=[1, 0, 1, 0], scores=[0.95, 0.05, 0.92, 0.08])
    report = R.render([row], bar=0.817, results_dir=tmp_path, n_boot=100)
    assert "95% CI (frame)" in report
    assert "Per-frame breakdown" in report
    assert "F1" in report and "F2" in report


def test_win_flag_requires_ci_separation(tmp_path):
    # point estimate clears the bar, but a wide (noisy) CI dips below it -> no decisive win.
    row = _row("borderline", 0.83)
    C.write_result_atomic(row, tmp_path)
    # F1/F2 classify correctly, F3 is fully wrong -> point bacc well below 1 and a wide CI
    _write_sidecar(row, tmp_path, frames=["F1", "F1", "F2", "F2", "F3", "F3"],
                   y=[1, 0, 1, 0, 1, 0], scores=[0.9, 0.1, 0.9, 0.1, 0.1, 0.9])
    ci = R.row_ci(row, tmp_path, n_boot=200)
    assert ci is not None and ci[0] <= 0.817            # lower bound does not clear the bar
    assert R.ci_win(row, 0.817, ci) is False
    report = R.render([row], bar=0.817, results_dir=tmp_path, n_boot=200)
    win_line = [ln for ln in report.splitlines() if "borderline" in ln][0]
    assert "✅" not in win_line


def test_win_flag_set_when_ci_clears_bar(tmp_path):
    row = _row("strong", 1.0)
    C.write_result_atomic(row, tmp_path)
    _write_sidecar(row, tmp_path, frames=["F1", "F1", "F2", "F2", "F3", "F3"],
                   y=[1, 0, 1, 0, 1, 0], scores=[0.97, 0.02, 0.95, 0.03, 0.96, 0.04])
    ci = R.row_ci(row, tmp_path, n_boot=200)
    assert ci is not None and ci[0] > 0.817
    assert R.ci_win(row, 0.817, ci) is True


def test_win_flag_falls_back_to_point_when_no_sidecar(tmp_path):
    # no sidecar -> CI is None -> ci_win falls back to the point-estimate rule
    row = _row("legacy", 0.83)
    assert R.row_ci(row, tmp_path) is None
    assert R.ci_win(row, 0.817, None) is True


def test_clean_vs_noisy_column(tmp_path):
    # one stored negative is actually cogongrass; the clean relabel flips it and the clean
    # balanced accuracy column reflects the corrected ground truth.
    row = _row("dinov2", 0.5)
    C.write_result_atomic(row, tmp_path)
    paths = ["tiles/x/F1_r0_c0.jpg", "tiles/x/F1_r0_c1.jpg",
             "tiles/x/F2_r0_c0.jpg", "tiles/x/F2_r0_c1.jpg"]
    recs = C.build_score_records(paths, y_true_cog=[1, 0, 1, 0], scores=[0.95, 0.92, 0.05, 0.08])
    for rec, f in zip(recs, ["F1", "F1", "F2", "F2"]):
        rec.frame = f
    C.write_scores_atomic(row.identity(), recs, tmp_path)
    # raw labels: tile 1 (p=0.92) is "not_cogongrass" -> a false negative in the answer key.
    relabel = {paths[1]: C.COG_CLASS}
    clean = R.clean_balanced_accuracy(row, tmp_path, relabel)
    assert clean is not None and clean > R.clean_balanced_accuracy(row, tmp_path, {})
    report = R.render([row], bar=0.817, results_dir=tmp_path, n_boot=50, relabel=relabel)
    assert "clean bacc" in report


def test_render_writes_baselines_and_ranking_order():
    rows = [_row("low", 0.76), _row("high", 0.83), _row("mid", 0.80)]
    report = R.render(rows, bar=0.817)
    order = [ln for ln in report.splitlines() if ln.startswith("| 1 ") or ln.startswith("| 2 ")
             or ln.startswith("| 3 ")]
    assert "high" in order[0] and "mid" in order[1] and "low" in order[2]
    assert "0.804" in report and "0.817" in report   # both baselines present
