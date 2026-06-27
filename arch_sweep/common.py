"""Shared library for the traditional-ML architecture sweep (U1).

Single source of the **0606 -> 0422 cross-collection split**, the **metric set**, a
**deterministic job identity**, the **per-cell seed policy**, and a **crash-safe
incremental result writer** that *every* model script imports. Keeping these here
(KTD1/KTD3/KTD7/KTD8) is what makes "separate scripts that train all the models" safe:
leakage prevention and comparability are single-sourced and cannot drift between scripts.

The split + metric helpers are ported from ``vlm_zeroshot/common.py`` and
``train_tiles_collection.py:39-61`` so the sweep matches the trained baselines without
importing or mutating any baseline script. The held-out set is every tile whose frame
``date_of(frame_of(path)) == "20260422"``, grouped by frame so no frame spans the
0422 / 0606 boundary. The 0606 slice is read ONLY to pick a fixed operating threshold
(``pick_threshold_on``) — never to train and never swept on 0422 itself.

Metrics follow the repo conventions (CLAUDE.md): report balanced accuracy + per-class
recall, never raw accuracy; the F2 sweep mirrors ``threshold_sweep.py:64-75``.

Self-check:  python arch_sweep/common.py --self-check tiles_dataset
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Collection split  (copied from train_tiles_collection.py:39-61 — see module docstring)
# ---------------------------------------------------------------------------
TEST_DATE = "20260422"        # held-out flight / "new field it has never seen"
TRAIN_DATE = "20260606"       # only read to pick a fixed operating threshold
CLASSES = ["cogongrass", "not_cogongrass"]   # ImageFolder order (alphabetical)
COG_CLASS = "cogongrass"
VAL_FRAC = 0.12               # 0606 validation slice for early stopping (matches baseline)
DEFAULT_SEED = 42             # comparability invariant — same split seed everywhere (KTD7)

# F2 sweep grid — same operating points as threshold_sweep.py:66
SWEEP_THRESHOLDS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

# Cross-collection baselines to beat (origin success bar).
BASELINES = [
    ("ResNet18 cross-collection", 0.804),
    ("Stage-1 DA cross-collection", 0.817),
]

# Where every job writes its own result file (KTD8). Git-ignored.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The two eval settings that must never be blended (KTD4).
EVAL_CROSS = "cross_collection"   # train 0606, test 0422, no target labels (headline)
EVAL_FEWSHOT = "few_shot"         # spends a labeled 0422 budget (reported separately)


def frame_of(path: str) -> str:
    """Frame stem a tile came from: ``<frame>_r<row>_c<col>.jpg`` -> ``<frame>``."""
    return re.match(r"(.+)_r\d+_c\d+$", Path(path).stem).group(1)


def date_of(frame: str) -> str:
    """Collection date encoded in a frame name (``DJI_20260422...`` -> ``20260422``)."""
    m = re.match(r"DJI_(\d{8})", frame)
    return m.group(1) if m else "other"


def indices_for_date(samples: Sequence[tuple], date: str) -> list[int]:
    """Indices into ``samples`` (``(path, label)`` pairs) whose frame date == ``date``.

    Selection is by frame date, and ``date_of`` is a pure function of the frame, so a
    frame can never land in two date slices — the no-leakage guarantee is structural.
    """
    return [i for i, (p, _) in enumerate(samples) if date_of(frame_of(p)) == date]


def frames_for_date(samples: Sequence[tuple], date: str) -> set[str]:
    """Distinct source frames present in the given date slice."""
    return {frame_of(p) for (p, _) in samples if date_of(frame_of(p)) == date}


def split_by_collection(samples: Sequence[tuple], cog_idx: int, seed: int = DEFAULT_SEED):
    """Frame-grouped 0606 -> 0422 split (ported from train_tiles_collection.py:45-61).

    Returns ``(train_idx, val_idx, test_idx, (n_train_frames, n_val_frames, n_test_frames))``.
    Train/val are carved from the non-0422 pool **by frame** (stratified on whether the
    frame contains any cogongrass) so no frame spans train and val; test is every 0422 tile.
    Identical recipe and seed for every model script — the comparability invariant (KTD7).
    """
    frames: dict[str, dict] = {}
    for i, (p, lab) in enumerate(samples):
        f = frame_of(p)
        d = frames.setdefault(f, {"idx": [], "pos": False, "date": date_of(f)})
        d["idx"].append(i)
        if lab == cog_idx:
            d["pos"] = True
    test_f = [f for f, d in frames.items() if d["date"] == TEST_DATE]
    pool = [f for f, d in frames.items() if d["date"] != TEST_DATE]
    rng = random.Random(seed)
    pos = [f for f in pool if frames[f]["pos"]]
    neg = [f for f in pool if not frames[f]["pos"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    nvp, nvn = int(len(pos) * VAL_FRAC), int(len(neg) * VAL_FRAC)
    val_f = pos[:nvp] + neg[:nvn]
    tr_f = pos[nvp:] + neg[nvn:]
    idx = lambda fl: [i for f in fl for i in frames[f]["idx"]]
    return idx(tr_f), idx(val_f), idx(test_f), (len(tr_f), len(val_f), len(test_f))


def balance(idx: Sequence[int], samples: Sequence[tuple], cog_idx: int, rng: random.Random) -> list[int]:
    """Down-sample the majority class to parity (ported from train_tiles_collection.py:64-70)."""
    pos = [i for i in idx if samples[i][1] == cog_idx]
    neg = [i for i in idx if samples[i][1] != cog_idx]
    if pos and neg:
        if len(neg) > len(pos):
            neg = rng.sample(neg, len(pos))
        elif len(pos) > len(neg):
            pos = rng.sample(pos, len(neg))
    out = pos + neg
    rng.shuffle(out)
    return out


def enumerate_tiles(data_dir: str = "tiles_dataset") -> tuple[list[tuple], list[str], int]:
    """Enumerate ``(path, label_idx)`` pairs from a torchvision ImageFolder.

    torchvision is imported lazily so importing this module (and unit-testing the pure
    split/metric helpers) does not require the full ML stack. Returns
    ``(samples, classes, cog_idx)``.
    """
    from torchvision import datasets  # lazy: only needed when reading the real dataset

    folder = datasets.ImageFolder(data_dir)
    cog_idx = folder.classes.index(COG_CLASS)
    return list(folder.samples), list(folder.classes), cog_idx


# ---------------------------------------------------------------------------
# Per-cell seed policy  (KTD7 — same seed everywhere, recorded in every row)
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python / numpy / torch deterministically for one cell.

    torch is imported lazily so the pure split/metric/writer helpers can be used (and
    tested) without the ML stack. cudnn is set deterministic so a re-run with the same
    seed reproduces the same metric (the comparability invariant, KTD7).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics  (CLAUDE.md: balanced accuracy + per-class recall, never raw accuracy)
# ---------------------------------------------------------------------------
def balanced_accuracy(y_true_cog: Sequence[int], y_pred_cog: Sequence[int]) -> float:
    return float(balanced_accuracy_score(y_true_cog, y_pred_cog))


def per_class_recall(y_true_cog: Sequence[int], y_pred_cog: Sequence[int]) -> dict[str, float]:
    """Recall for each class, keyed by class name. 1 == cogongrass, 0 == not_cogongrass."""
    rec = recall_score(y_true_cog, y_pred_cog, labels=[1, 0], average=None, zero_division=0.0)
    return {COG_CLASS: float(rec[0]), "not_cogongrass": float(rec[1])}


def auroc(y_true_cog: Sequence[int], scores: Sequence[float]) -> float:
    return float(roc_auc_score(y_true_cog, scores))


def average_precision(y_true_cog: Sequence[int], scores: Sequence[float]) -> float:
    return float(average_precision_score(y_true_cog, scores))


def _confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """recall / precision / F1 / F2 / FN at one threshold (mirrors threshold_sweep.py:67-74)."""
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / (tp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
    return {"thr": thr, "recall": rec, "prec": prec, "f1": f1, "f2": f2, "fn": fn}


def f2_sweep(y_true_cog: Sequence[int], scores: Sequence[float],
             thresholds: Sequence[float] = SWEEP_THRESHOLDS) -> list[dict]:
    """Recall / precision / F1 / F2 / FN at each threshold (mirrors threshold_sweep.py)."""
    y = np.asarray(y_true_cog, dtype=int)
    p = np.asarray(scores, dtype=float)
    n_cog = int(y.sum())
    return [{**_confusion_at(y, p, thr), "n_cog": n_cog} for thr in thresholds]


def pick_threshold_on(y_true_cog: Sequence[int], scores: Sequence[float]) -> float:
    """Pick the F2-optimal operating threshold from the GIVEN scores only.

    Honest-threshold rule (origin Key Decision #2): callers must only ever pass it
    0606-derived scores. The function is pure — it operates on whatever scores it is
    handed and has no way to see the 0422 slice.
    """
    y = np.asarray(y_true_cog, dtype=int)
    p = np.asarray(scores, dtype=float)
    uniq = np.unique(p)
    cands = (np.concatenate([[0.0], (uniq[:-1] + uniq[1:]) / 2.0, [1.0]])
             if uniq.size > 1 else np.array([0.5]))
    best_thr, best_f2 = 0.5, -1.0
    for thr in cands:
        f2 = _confusion_at(y, p, thr)["f2"]
        if f2 > best_f2:
            best_f2, best_thr = f2, float(thr)
    return best_thr


def metrics_at_threshold(y_true_cog: Sequence[int], scores: Sequence[float], thr: float) -> dict:
    """recall / precision / f1 / f2 / fn at one threshold (public wrapper over ``_confusion_at``)."""
    y = np.asarray(y_true_cog, dtype=int)
    p = np.asarray(scores, dtype=float)
    return _confusion_at(y, p, thr)


# ---------------------------------------------------------------------------
# Honest operating threshold + calibration  (U4 — report at the deployed point)
# ---------------------------------------------------------------------------
def prior_match_threshold(scores: Sequence[float], prior: float) -> float:
    """Label-free threshold whose predicted-positive fraction ≈ ``prior`` (known prevalence).

    Reads only ``scores`` and the supplied ``prior`` — **never the target labels** (KTD5
    honesty). The top ``round(prior*n)`` scored tiles are called positive; the returned
    threshold sits between the k-th and (k+1)-th largest score so exactly that many clear it.
    """
    p = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(p)
    if n == 0:
        return 0.5
    prior = min(max(float(prior), 0.0), 1.0)
    k = int(round(prior * n))
    if k <= 0:
        return float(p[0]) + 1e-9        # predict nothing positive
    if k >= n:
        return float(p[-1])              # predict everything positive
    return float((p[k - 1] + p[k]) / 2.0)


def fit_temperature(p_val: Sequence[float], y_val: Sequence[int], *, grid=None) -> float:
    """Fit a scalar temperature on 0606 val scores to minimize BCE (Platt-style calibration).

    Works in binary-logit space (``logit = log(p/(1-p))``); scaling by ``T`` is **monotonic**,
    so it preserves the score ranking (AUROC unchanged) and only recalibrates confidence.
    Fit on the source (0606) scores only — never the target. Returns the best ``T``.
    """
    p = np.clip(np.asarray(p_val, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_val, dtype=int)
    if len(set(y.tolist())) < 2:
        return 1.0
    logit = np.log(p / (1 - p))
    if grid is None:
        grid = np.linspace(0.05, 10.0, 200)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        q = np.clip(1.0 / (1.0 + np.exp(-logit / T)), 1e-6, 1 - 1e-6)
        nll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def apply_temperature(scores: Sequence[float], T: float) -> np.ndarray:
    """Apply a fitted temperature to probabilities (monotonic recalibration)."""
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / float(T)))


def conformal_threshold(cal_scores: Sequence[float], cal_labels: Sequence[int],
                        target_fnr: float = 0.1) -> float:
    """Label-DEPENDENT threshold controlling the false-negative rate to ``<= target_fnr``.

    Split-conformal style: with a labeled calibration set, threshold at the ``target_fnr``
    quantile of the positive scores so at most that fraction of positives fall below it. Uses
    **target labels**, so any cell using it MUST be tagged ``eval_setting=few_shot`` (KTD5).
    """
    p = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_labels, dtype=int)
    pos = np.sort(p[y == 1])
    if pos.size == 0:
        return 0.0
    k = int(np.floor(min(max(target_fnr, 0.0), 1.0) * pos.size))
    return float(pos[min(k, pos.size - 1)])


# ---------------------------------------------------------------------------
# Result row  (full cell config + all metrics) and deterministic job identity
# ---------------------------------------------------------------------------
# The fields that DEFINE a cell. ``job_id`` hashes exactly these, so two runs of the
# same cell collide (resume) and different cells never do. Outcome fields (status,
# metrics, timing) are deliberately excluded — they are results, not identity.
IDENTITY_FIELDS = ("model", "variant", "tuning_mode", "head", "adaptation",
                   "eval_setting", "seed", "extra")


@dataclass
class ResultRow:
    """One sweep cell: its full config (identity) plus every metric it produced.

    A row is written to ``results/<job_id>.jsonl`` the instant it is produced (KTD8).
    ``adaptation`` distinguishes e.g. ``none`` vs ``adabn`` so a TTA cell's adapted and
    un-adapted readings get distinct ``job_id``s and distinct files (no shared-file race).
    """

    # --- identity (feeds job_id) ---
    model: str                                  # backbone / model name (e.g. "dinov2")
    variant: str = "reference"                  # data variant (e.g. "tile512_clahe")
    tuning_mode: str = "frozen"                 # frozen | lora | full
    head: str = "mlp_bn"                         # head type (linear | mlp_bn)
    adaptation: str = "none"                    # none | adabn | tent | eata | rotta | ssl ...
    eval_setting: str = EVAL_CROSS              # cross_collection | few_shot (KTD4)
    seed: int = DEFAULT_SEED
    extra: str = ""                             # free identity discriminator (e.g. "size=l")

    # --- outcome / status ---
    status: str = "ok"                          # ok | failed | oom
    error: str = ""                             # message when status != ok

    # --- metrics (cross-collection or few-shot, per eval_setting) ---
    balanced_accuracy: float | None = None
    recall_cogongrass: float | None = None
    recall_not_cogongrass: float | None = None
    auroc: float | None = None
    average_precision: float | None = None
    threshold: float | None = None             # operating point fit on 0606 (cross) / budget
    val_balanced_accuracy: float | None = None  # best 0606 val bacc (early-stop signal)
    f2_sweep: list = field(default_factory=list)

    # --- honest deployed operating point + calibration (U4; outcome, not identity) ---
    temperature: float | None = None           # 0606-fit calibration temperature (monotonic)
    prior: float | None = None                 # known prevalence used for prior-matching
    threshold_prior: float | None = None       # label-free prior-matched threshold on 0422
    recall_cog_at_op: float | None = None       # cogongrass recall at the 0606 F2-threshold
    f2_at_op: float | None = None               # F2 at the 0606 F2-threshold
    recall_cog_at_prior: float | None = None    # cogongrass recall at the prior-matched threshold
    f2_at_prior: float | None = None            # F2 at the prior-matched threshold

    # --- provenance / budget ---
    n_train: int | None = None
    n_val: int | None = None
    n_test: int | None = None
    n_cog_test: int | None = None
    trainable_params: int | None = None
    budget: int | None = None                  # few-shot label budget (U10); None for cross
    created_at: str = ""                        # ISO timestamp, set by caller; optional

    def identity(self) -> dict:
        return {k: getattr(self, k) for k in IDENTITY_FIELDS}

    @property
    def job_id(self) -> str:
        return job_id(self.identity())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResultRow":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def job_id(config) -> str:
    """Deterministic 16-hex id for a cell from its identity fields only.

    Accepts a ``ResultRow`` or a plain dict (the orchestrator computes ids from planned
    configs that carry no metrics yet). Missing identity keys fall back to the
    ``ResultRow`` defaults, so a minimal ``{"model": "dinov2"}`` and a full row for the
    same cell hash identically — the property that makes resume (KTD8) correct.
    """
    if isinstance(config, ResultRow):
        ident = config.identity()
    else:
        defaults = {f.name: f.default for f in fields(ResultRow)}
        ident = {k: config.get(k, defaults[k]) for k in IDENTITY_FIELDS}
    blob = json.dumps(ident, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def result_path(config, results_dir: Path | str = RESULTS_DIR) -> Path:
    jid = config if isinstance(config, str) else job_id(config)
    return Path(results_dir) / f"{jid}.jsonl"


def result_exists(config, results_dir: Path | str = RESULTS_DIR) -> bool:
    """True iff this cell's result file already exists (resume skip, KTD8)."""
    return result_path(config, results_dir).exists()


def write_result_atomic(row: ResultRow, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Write one cell's result crash-safely: tmp -> flush -> fsync -> os.replace (KTD8).

    The final ``results/<job_id>.jsonl`` only ever appears via an atomic rename of a
    fully-written temp file, so an interrupt mid-write leaves either the previous file
    or nothing — never a truncated final file, and never a shared-file write race
    (each job owns its own path). Returns the final path.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    final = results_dir / f"{row.job_id}.jsonl"
    line = json.dumps(row.to_dict(), default=str) + "\n"
    fd, tmp = tempfile.mkstemp(dir=results_dir, prefix=".tmp-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)   # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return final


def read_result(path: str | Path) -> ResultRow:
    """Read one job file back into a ResultRow (round-trips ``write_result_atomic``)."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                return ResultRow.from_dict(json.loads(line))
    raise ValueError(f"no result row in {path}")


def read_all_results(results_dir: Path | str = RESULTS_DIR) -> list[ResultRow]:
    """Glob + merge every per-job result file (the report's input; KTD8).

    Per-tile score sidecars (``<job_id>.scores.jsonl``, U1) live in the same dir and also
    match ``*.jsonl`` — they are skipped here so they never get parsed as result rows.
    """
    out = []
    for p in sorted(Path(results_dir).glob("*.jsonl")):
        if p.name.endswith(SCORES_SUFFIX):
            continue   # a per-tile score sidecar, not a result row (U1)
        try:
            out.append(read_result(p))
        except ValueError:
            continue   # skip an empty/partial file rather than crash the report
    return out


# ---------------------------------------------------------------------------
# Per-tile confidence sidecar  (U1 — ScoreRecord ported from vlm_zeroshot)
# ---------------------------------------------------------------------------
# Persist per-tile ``P(cogongrass)`` once, alongside the result row, so label-cleaning (U3),
# ensembling (U5), and self-training (U11) consume stored scores instead of recomputing them
# (KTD3). The sidecar is ``results/<job_id>.scores.jsonl`` — same atomic temp->fsync->replace
# path as the result row, one JSONL record per evaluated 0422 tile.
SCORES_SUFFIX = ".scores.jsonl"


@dataclass
class ScoreRecord:
    """One evaluated 0422 tile's persisted confidence (the vlm_zeroshot ScoreRecord shape)."""

    path: str                    # tile path (frame-encoded; ``frame`` is derived from it)
    frame: str                   # source frame stem (``frame_of(path)``)
    true_label: str              # class name: "cogongrass" | "not_cogongrass"
    p_cogongrass: float          # model P(cogongrass) in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def build_score_records(paths, y_true_cog, scores) -> list[ScoreRecord]:
    """Zip aligned ``(path, true 0/1 label, P(cog))`` triples into ScoreRecords.

    ``y_true_cog`` is the 1==cogongrass / 0==not encoding used everywhere else; the class
    name is derived from it so the sidecar is self-describing without the label map.
    """
    out = []
    for p, y, pc in zip(paths, y_true_cog, scores):
        out.append(ScoreRecord(path=str(p), frame=frame_of(str(p)),
                               true_label=COG_CLASS if int(y) == 1 else "not_cogongrass",
                               p_cogongrass=float(pc)))
    return out


def scores_path(config, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Path of the per-tile score sidecar for a cell (mirrors ``result_path``)."""
    jid = config if isinstance(config, str) else job_id(config)
    return Path(results_dir) / f"{jid}{SCORES_SUFFIX}"


def write_scores_atomic(config, records: Sequence[ScoreRecord],
                        results_dir: Path | str = RESULTS_DIR) -> Path:
    """Write a cell's per-tile scores crash-safely: tmp -> flush -> fsync -> os.replace (U1).

    Same atomicity guarantee as ``write_result_atomic`` — the final sidecar only ever appears
    via an atomic rename of a fully-written temp file, so an interrupt leaves either the prior
    sidecar or nothing, never a truncated file. ``config`` may be a ResultRow/dict/job_id str.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jid = config if isinstance(config, str) else job_id(config)
    final = results_dir / f"{jid}{SCORES_SUFFIX}"
    payload = "".join(json.dumps(r.to_dict(), default=str) + "\n" for r in records)
    fd, tmp = tempfile.mkstemp(dir=results_dir, prefix=".tmp-", suffix=SCORES_SUFFIX)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)   # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return final


def read_scores(path: str | Path) -> list[ScoreRecord]:
    """Read a per-tile score sidecar back into ScoreRecords (round-trips the writer)."""
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ScoreRecord.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Frame-level bootstrap CIs + per-frame breakdown  (U2 — wins must be decidable)
# ---------------------------------------------------------------------------
# Tiles within a frame are correlated, so a tile-level bootstrap would understate the CI.
# We resample **frames** with replacement (tiles of a frame always move together, mirroring
# the frame-grouped 0606->0422 split) — the honest CI for "a new flight it has never seen".
def frame_groups(frames: Sequence[str]) -> dict[str, list[int]]:
    """Map each frame id to the tile indices belonging to it (preserves order)."""
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        groups.setdefault(f, []).append(i)
    return groups


def bootstrap_ci(frames: Sequence[str], metric_fn, *, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Frame-resampled ``(lo, point, hi)`` CI for a tile-level metric.

    ``metric_fn(tile_indices)`` computes the metric over the given tile indices and returns a
    float, or ``None`` for a degenerate resample (e.g. one class absent) which is skipped. Each
    bootstrap iteration resamples the **set of frames** with replacement and unions their tile
    indices, so the no-leakage frame grouping is preserved inside the CI. ``point`` is the
    metric on the full set; ``lo``/``hi`` are the ``alpha/2`` / ``1-alpha/2`` percentiles.
    """
    groups = frame_groups(frames)
    frame_ids = list(groups)
    point = metric_fn(list(range(len(frames))))
    if not frame_ids:
        return (point, point, point)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sampled = [frame_ids[rng.randrange(len(frame_ids))] for _ in range(len(frame_ids))]
        idx = [i for f in sampled for i in groups[f]]
        m = metric_fn(idx)
        if m is not None:
            stats.append(m)
    if not stats:
        return (point, point, point)
    stats.sort()
    lo = stats[min(len(stats) - 1, int((alpha / 2) * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return (float(lo), float(point), float(hi))


def balanced_accuracy_ci(frames, y_true_cog, scores, *, threshold: float = 0.5,
                         n_boot: int = 1000, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Frame-resampled CI for balanced accuracy at ``threshold`` (argmax by default).

    Convenience wrapper over ``bootstrap_ci`` for the headline metric, matching the argmax
    (``p >= 0.5``) rule ``trainer._make_ok_row`` uses for the point estimate.
    """
    y = list(y_true_cog)
    p = list(scores)

    def metric_fn(idx):
        yt = [y[i] for i in idx]
        if len(set(yt)) < 2:
            return None
        yp = [1 if p[i] >= threshold else 0 for i in idx]
        return balanced_accuracy(yt, yp)

    return bootstrap_ci(frames, metric_fn, n_boot=n_boot, seed=seed)


def ci_separated(ci: tuple[float, float, float], bar: float) -> bool:
    """True iff the CI's lower bound is strictly above ``bar`` (a decisive win over a point)."""
    return ci[0] > bar


def cis_overlap(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True iff two CIs overlap (the comparison is undecidable) — uses [lo, hi] of each."""
    return not (a[2] < b[0] or b[2] < a[0])


def per_frame_metrics(frames: Sequence[str], y_true_cog, y_pred_cog) -> list[dict]:
    """One summary row per frame: tile/cog counts, cogongrass recall, balanced accuracy.

    Surfaces whether failure is a few bad frames or systematic. ``recall_cog`` is ``None`` for
    a frame with no cogongrass tiles; ``bacc`` is ``None`` when a frame is single-class.
    """
    groups = frame_groups(frames)
    yt_all, yp_all = list(y_true_cog), list(y_pred_cog)
    out = []
    for f in sorted(groups):
        idx = groups[f]
        yt = [yt_all[i] for i in idx]
        yp = [yp_all[i] for i in idx]
        n_cog = int(sum(yt))
        rec = per_class_recall(yt, yp)[COG_CLASS] if n_cog else None
        bacc = balanced_accuracy(yt, yp) if len(set(yt)) == 2 else None
        out.append({"frame": f, "n": len(idx), "n_cog": n_cog,
                    "recall_cog": rec, "bacc": bacc})
    return out


# ---------------------------------------------------------------------------
# Self-check: prints 0422 / 0606 frame + tile counts from the real dataset.
# ---------------------------------------------------------------------------
def _self_check(data_dir: str = "tiles_dataset") -> None:
    samples, classes, cog_idx = enumerate_tiles(data_dir)
    print(f"classes: {classes}  (cog_idx={cog_idx})")
    for date, name in [(TEST_DATE, "HELD-OUT 0422"), (TRAIN_DATE, "0606 (threshold-only)")]:
        idx = indices_for_date(samples, date)
        frames = frames_for_date(samples, date)
        n_cog = sum(1 for i in idx if classes[samples[i][1]] == COG_CLASS)
        print(f"{name}: {len(frames)} frames | {len(idx)} tiles "
              f"({n_cog} cogongrass / {len(idx) - n_cog} not_cogongrass)")
    test_frames = frames_for_date(samples, TEST_DATE)
    train_frames = frames_for_date(samples, TRAIN_DATE)
    overlap = test_frames & train_frames
    print(f"frame overlap 0422 & 0606: {len(overlap)} (must be 0)")
    assert not overlap, "LEAKAGE: a frame appears in both 0422 and 0606 slices"

    # Split sanity: train/val/test index sets are mutually disjoint and test == 0422.
    tr, va, te, (nf_tr, nf_va, nf_te) = split_by_collection(samples, cog_idx)
    print(f"frame split -> train(0606) {nf_tr} | val(0606) {nf_va} | TEST(0422) {nf_te}")
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert set(te) == set(indices_for_date(samples, TEST_DATE)), "test slice must be exactly 0422"

    # Identity / writer smoke.
    row = ResultRow(model="selfcheck", variant="reference")
    print(f"sample job_id(selfcheck/reference) = {row.job_id}")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--self-check"]
    _self_check(args[0] if args else "tiles_dataset")
