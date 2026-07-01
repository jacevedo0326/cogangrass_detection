"""Shared pipeline contract for the cogongrass tile classifier (plan U1, R1/R2/R4/R5/R6).

Single source for the conventions that were copy-pasted across 6+ scripts:

* **Tile identity** — ``frame_of`` / ``group_of`` (the leakage-prevention group key).
* **Collection dates** — ``date_of`` with the pre-seeded non-DJI whitelist, plus the
  parameterized/env-overridable ``HELDOUT_DATES`` (default ``["20260422"]``).
* **Splits** — ``split_by_collection`` (cross-collection DA protocol) and
  ``grouped_split`` (random frame-grouped 80/10/10), plus ``balance``.
* **Tiling geometry** — ``tile_boxes`` / ``cut_tile``: ceil grid with ``min()``-clamped
  partial edge crops resized to the save size (training's actual rule — NO zero-padding).
* **Vegetation filter** — ``exg_map`` / ``tile_is_veg`` (ExG = 2G-R-B, threshold 0.03).
* **AdaBN** — ``adapt_bn``, input-space ``BatchNorm2d`` adaptation for the DA/ResNet
  path ONLY (the ensemble deploy path uses ``arch_sweep/tta.adapt_head`` instead).
* **Tiling-provenance manifest** — ``write_provenance`` / ``read_provenance`` /
  ``provenance_hash``: the anti-silent-drift primitive (feature-cache signatures and
  the wipe guard hash it in).

Every helper is ported from its canonical source and cited inline, mirroring
``arch_sweep/common.py``'s style. The protected baseline ``train_tiles.py`` keeps its
private copies (CLAUDE.md isolation rule) and ``arch_sweep/common.py`` stays
self-contained; ``tests/test_tile_common.py`` is the equivalence tripwire for all three.

Naming invariant: this module's name must stay disjoint from ``arch_sweep/``'s flat
module namespace (``common``, ``backbones``, ``features``, ``trainer``, ``heads``,
``tta``, ``report``, ``augment``, ``data_variants``, ``run_all``, ``continued_ssl``,
``fewshot``) because deploy scripts ``sys.path.insert`` that directory.

Heavy imports (torch / torchvision / PIL) are lazy — importing this module and testing
the pure helpers needs only the standard library and numpy.

Self-check:  python tile_common.py --self-check tiles_dataset
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants (values ported from their canonical scripts, cited per constant)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42            # every split everywhere (train_tiles.py:33, train_tiles_collection.py:23)
CLASSES = ["cogongrass", "not_cogongrass"]   # ImageFolder order (alphabetical)
COG_CLASS = "cogongrass"

# split_by_collection: 0606-pool validation slice (train_tiles_collection.py:24)
COLLECTION_VAL_FRAC = 0.12
# grouped_split: random frame-grouped 80/10/10 (train_tiles.py:34)
GROUPED_VAL_FRAC, GROUPED_TEST_FRAC = 0.10, 0.10

VEG_THRESH = 0.03            # drop tile if mean ExG below this (boxes_to_tiles.py:31)
COVER_THRESH = 0.30          # tile is cogongrass if >= this fraction inside a box (boxes_to_tiles.py:28)

# The 14 non-DJI frame stems present in the live tiles_dataset (close-up burst captures
# with camera-roll names, no encoded date). The legacy ``date_of`` returned "other" for
# them, silently routing them to the TRAIN pool of the cross-collection split; this
# whitelist preserves that exact train-pool membership (R4) while any NEW unparseable
# stem now fails loud instead of silently entering training. Discovered by listing
# tiles_dataset/{cogongrass,not_cogongrass} and keeping stems not matching DJI_########.
NON_DJI_FRAME_WHITELIST = frozenset({
    "BRMG5867", "BSEJ8944", "BTKI8738", "BXBY1903", "CIBA0087", "CUEX2657",
    "CWMA2987", "DCLM2115", "DCQJ5971", "DEAO2697", "DEZG0880", "DGKY5293",
    "DJXM9111", "DRVR8800",
})

# ---------------------------------------------------------------------------
# Held-out collections (R4). Default preserves every current number: the DA-track
# protocol trains on 2026-06-06 and holds out the entire 2026-04-22 flight
# (train_tiles_collection.py:25 ``TEST_DATE``). Override with the ``HELDOUT_DATES``
# env var (comma-separated) or per call via the ``heldout_dates=`` parameter.
# ---------------------------------------------------------------------------
DEFAULT_HELDOUT_DATES = ("20260422",)


def heldout_dates_from_env(env=None) -> list[str]:
    """Resolve the held-out collection dates from the environment.

    ``HELDOUT_DATES="20270301,20270315"`` -> ``["20270301", "20270315"]``; unset or
    blank -> the default ``["20260422"]``. Pure over the given mapping so it is testable
    without mutating ``os.environ``.
    """
    env = os.environ if env is None else env
    raw = env.get("HELDOUT_DATES", "")
    dates = [d.strip() for d in raw.split(",") if d.strip()]
    return dates if dates else list(DEFAULT_HELDOUT_DATES)


HELDOUT_DATES = heldout_dates_from_env()


# ---------------------------------------------------------------------------
# Tile identity  (ported from train_tiles.py:53-55 / train_tiles_collection.py:39)
# ---------------------------------------------------------------------------
_TILE_RE = re.compile(r"(.+)_r\d+_c\d+$")
_BLOCK_RE = re.compile(r".*_bk\d+_\d+$")
_DJI_DATE_RE = re.compile(r"DJI_(\d{8})")
_ORTHO_DATE_RE = re.compile(r".+-(\d{8})_bk\d+_\d+")


def frame_of(name: str) -> str:
    """Frame stem a tile came from: ``<frame>_r<row>_c<col>.jpg`` -> ``<frame>``.

    Ported from train_tiles.py:53-55 (identical regex in train_tiles_collection.py:39
    and arch_sweep/common.py:68-70). The regex is greedy, so only the LAST ``_r#_c#``
    segment is stripped. Raises ValueError on a non-tile name instead of the originals'
    bare AttributeError; behavior on valid tile names is identical.
    """
    stem = Path(name).stem
    m = _TILE_RE.match(stem)
    if not m:
        raise ValueError(f"not a tile name (expected <frame>_r<row>_c<col>): {stem!r}")
    return m.group(1)


def group_of(name: str) -> str:
    """The leakage-prevention group key for a tile (generalizes ``frame_of``, R4/KTD).

    * DJI oblique tiles (``DJI_20260422_..._r3_c7.jpg``): group == frame stem —
      unchanged behavior, so splitting/CIs/TTA keyed on this reproduce the baseline.
    * Orthomosaic tiles (``<site>-<YYYYMMDD>_bk<br>_<bc>_r#_c#.jpg``): the frame stem
      itself carries the spatial-block segment ``_bk<br>_<bc>``, so the group is
      flight+block — spatially adjacent ortho tiles share a group (block-level
      leakage guard) while different blocks of the same flight may split apart.

    Both cases reduce to the frame stem; the block is *part of* the stem by the ortho
    naming contract (``prep_ortho.py``, plan U6). ``_BLOCK_RE`` documents/detects the
    block form.
    """
    return frame_of(name)


def has_block(frame: str) -> bool:
    """True iff a frame/group stem carries an orthomosaic spatial-block segment."""
    return bool(_BLOCK_RE.match(frame))


def date_of(name: str, whitelist: frozenset = NON_DJI_FRAME_WHITELIST) -> str:
    """Collection date encoded in a frame (or tile) stem.

    Ported from train_tiles_collection.py:40-42 / arch_sweep/common.py:73-76, with the
    silent ``"other"`` fallback replaced by a whitelist + hard error (R4):

    * ``DJI_20260422_1234...`` -> ``"20260422"`` (oblique DJI frames).
    * ``siteA-20270301_bk2_5...`` -> ``"20270301"`` (orthomosaic flight+block stems).
    * A stem whose frame is in ``whitelist`` -> ``"other"``. The default whitelist is
      the exact 14 non-DJI stems in the live dataset (``NON_DJI_FRAME_WHITELIST``);
      "other" never matches a held-out date, so these route to the TRAIN pool exactly
      as the legacy ``date_of`` did — current numbers reproduce bit-for-bit.
    * Anything else raises ValueError naming the stem, so a new collection with an
      unrecognized naming scheme can never silently leak into the training pool.

    Accepts either a frame stem or a full tile name (the ``_r#_c#`` suffix is stripped
    before the whitelist check).
    """
    stem = Path(name).stem
    m = _DJI_DATE_RE.match(stem)
    if m:
        return m.group(1)
    m = _ORTHO_DATE_RE.match(stem)
    if m:
        return m.group(1)
    tile_m = _TILE_RE.match(stem)
    frame = tile_m.group(1) if tile_m else stem
    if frame in whitelist:
        return "other"
    raise ValueError(
        f"cannot parse a collection date from frame {frame!r} (stem {stem!r}); "
        f"expected DJI_<YYYYMMDD>... or <site>-<YYYYMMDD>_bk<r>_<c>..., or add the "
        f"frame to the whitelist to route it to the train pool explicitly")


# ---------------------------------------------------------------------------
# Splits  (ported from train_tiles_collection.py:45-70 and train_tiles.py:58-91)
# ---------------------------------------------------------------------------
def split_by_collection(samples: Sequence[tuple], cog_idx: int, heldout_dates=None,
                        seed: int = DEFAULT_SEED, val_frac: float = COLLECTION_VAL_FRAC,
                        whitelist: frozenset = NON_DJI_FRAME_WHITELIST):
    """Group-keyed cross-collection split (ported from train_tiles_collection.py:45-61).

    Test = every tile whose group date is in ``heldout_dates`` (default
    ``HELDOUT_DATES`` == ``["20260422"]``); train/val are carved from the remaining
    pool BY GROUP, stratified on whether the group contains any cogongrass tile, with
    ``rng = random.Random(seed)`` shuffling pos then neg — the exact original order of
    operations, so the default call reproduces the baseline split bit-for-bit.

    Returns ``(train_idx, val_idx, test_idx, (n_train_groups, n_val_groups, n_test_groups))``.
    """
    if heldout_dates is None:
        heldout_dates = HELDOUT_DATES
    heldout = set(heldout_dates)
    frames: dict[str, dict] = {}
    for i, (p, lab) in enumerate(samples):
        f = group_of(p)
        d = frames.setdefault(f, {"idx": [], "pos": False,
                                  "date": date_of(f, whitelist=whitelist)})
        d["idx"].append(i)
        if lab == cog_idx:
            d["pos"] = True
    test_f = [f for f, d in frames.items() if d["date"] in heldout]
    pool = [f for f, d in frames.items() if d["date"] not in heldout]
    rng = random.Random(seed)
    pos = [f for f in pool if frames[f]["pos"]]
    neg = [f for f in pool if not frames[f]["pos"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    nvp, nvn = int(len(pos) * val_frac), int(len(neg) * val_frac)
    val_f = pos[:nvp] + neg[:nvn]
    tr_f = pos[nvp:] + neg[nvn:]
    idx = lambda fl: [i for f in fl for i in frames[f]["idx"]]
    return idx(tr_f), idx(val_f), idx(test_f), (len(tr_f), len(val_f), len(test_f))


def grouped_split(samples: Sequence[tuple], cog_idx: int, seed: int = DEFAULT_SEED,
                  val_frac: float = GROUPED_VAL_FRAC, test_frac: float = GROUPED_TEST_FRAC):
    """Random frame-grouped 80/10/10 split (ported from train_tiles.py:58-77).

    Groups are stratified by whether they contain any cogongrass tile; one group's
    tiles never span splits. The original shuffles with the module-global RNG seeded
    ``random.seed(42)`` at import; a fresh ``random.Random(seed)`` consumed in the
    same order (positive group list, then negative — the ``{True: [], False: []}``
    dict order) produces the identical shuffle sequence, which
    ``tests/test_tile_common.py`` proves against the live dataset.

    Returns ``(train_idx, val_idx, test_idx, (n_train_groups, n_val_groups, n_test_groups))``.
    """
    frames: dict[str, dict] = {}
    for i, (p, lab) in enumerate(samples):
        d = frames.setdefault(group_of(p), {"idx": [], "pos": False})
        d["idx"].append(i)
        if lab == cog_idx:
            d["pos"] = True
    groups = {True: [], False: []}
    for f, d in frames.items():
        groups[d["pos"]].append(f)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for fl in groups.values():
        rng.shuffle(fl)
        n = len(fl)
        nt = int(n * test_frac)
        nv = int(n * val_frac)
        te += fl[:nt]
        va += fl[nt:nt + nv]
        tr += fl[nt + nv:]
    idx = lambda fl: [i for f in fl for i in frames[f]["idx"]]
    return idx(tr), idx(va), idx(te), (len(tr), len(va), len(te))


def balance(idx: Sequence[int], samples: Sequence[tuple], cog_idx: int,
            rng: random.Random, ratio: float = 1.0) -> list[int]:
    """Undersample the majority class (ported from train_tiles_collection.py:64-70).

    ``ratio`` generalizes train_tiles.py:80-91's ``balance_split`` — at the default
    1.0 the arithmetic (``int(1.0 * n) == n``, same comparison, same sample counts)
    reduces exactly to the collection script's parity balance, so one function covers
    both originals' behavior.
    """
    pos = [i for i in idx if samples[i][1] == cog_idx]
    neg = [i for i in idx if samples[i][1] != cog_idx]
    if pos and neg:
        if len(neg) > ratio * len(pos):
            neg = rng.sample(neg, int(ratio * len(pos)))
        elif len(pos) > ratio * len(neg):
            pos = rng.sample(pos, int(ratio * len(neg)))
    out = pos + neg
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Tiling geometry  (ported from boxes_to_tiles.py:67-82 — the TRAINING rule, R5)
# Ceil grid + min()-clamped partial edge crops resized up. NOT zero-padding, and NOT
# heatmap_infer.py's legacy floor grid (which silently dropped partial edge tiles).
# ---------------------------------------------------------------------------
def tile_grid(W: int, H: int, tile_px: int) -> tuple[int, int]:
    """``(cols, rows)`` of the ceil grid (ported from boxes_to_tiles.py:67)."""
    return -(-W // tile_px), -(-H // tile_px)


def tile_boxes(W: int, H: int, tile_px: int) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Row-major ``(r, c, (x0, y0, x1, y1))`` crop boxes for one frame.

    Ported from boxes_to_tiles.py:67-72: ``cols, rows = -(-W // T), -(-H // T)`` with
    ``y1, x1 = min(H, y0 + T), min(W, x0 + T)`` — edge boxes are smaller REAL crops
    (clamped), never padded; a frame smaller than one tile yields exactly one clamped
    box covering the whole frame.
    """
    cols, rows = tile_grid(W, H, tile_px)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            y1, x1 = min(H, y0 + tile_px), min(W, x0 + tile_px)
            out.append((r, c, (x0, y0, x1, y1)))
    return out


def cut_tile(im, box: tuple[int, int, int, int], save_px: int):
    """Crop one tile box from a PIL image and resize to ``save_px`` square.

    Matches boxes_to_tiles.py:81's save behavior exactly:
    ``im.crop((x0, y0, x1, y1)).resize((save_px, save_px))`` with PIL's default
    resample — clamped edge crops are resized UP to the square save size (the input
    distribution the models were trained on).
    """
    x0, y0, x1, y1 = box
    return im.crop((x0, y0, x1, y1)).resize((save_px, save_px))


# ---------------------------------------------------------------------------
# Vegetation filter  (ported from boxes_to_tiles.py:63-74 — ExG "green-ness")
# ---------------------------------------------------------------------------
def exg_map(arr) -> np.ndarray:
    """Per-pixel Excess-Green index for an HxWx3 RGB array (boxes_to_tiles.py:63-65).

    ``ExG = 2G/S - R/S - B/S`` with ``S = R+G+B+1e-6``. Vegetation is strongly
    positive; sky, bare ground, and gray surfaces sit near or below zero.
    """
    arr = np.asarray(arr).astype(np.float32)
    ssum = arr.sum(2) + 1e-6
    return 2 * arr[..., 1] / ssum - arr[..., 0] / ssum - arr[..., 2] / ssum


def tile_is_veg(exg: np.ndarray, box: tuple[int, int, int, int],
                thresh: float = VEG_THRESH) -> bool:
    """True iff a tile box passes the vegetation filter (boxes_to_tiles.py:73-74).

    ``exg`` is the full-frame ``exg_map``; the tile is KEPT when its mean ExG is
    ``>= thresh`` (the tiler skips it otherwise — sky / non-vegetation).
    """
    x0, y0, x1, y1 = box
    return float(exg[y0:y1, x0:x1].mean()) >= thresh


# ---------------------------------------------------------------------------
# AdaBN  (ported from heatmap_infer.py:62-74 — DA / ResNet path ONLY)
# ---------------------------------------------------------------------------
def adapt_bn(model, batches: Iterable, device=None, chunk: int = 256,
             verbose: bool = True) -> int:
    """AdaBN: recompute input-space ``BatchNorm2d`` running stats on target tiles.

    Ported from heatmap_infer.py:62-74. Puts every ``BatchNorm2d`` into cumulative
    train mode (``reset_running_stats(); momentum = None; train()``), forward-passes
    the target tiles with no grad, then returns the model to eval. Worth ~+2 pts on
    the held-out collection with zero target labels.

    This serves the **DA / ResNet path ONLY** (``heatmap_infer.py``, ``tta_eval.py``,
    ``train_tiles_da*.py`` checkpoints, which keep input-space BatchNorm precisely so
    this works). The ensemble deploy path adapts a *feature-space* ``BatchNorm1d``
    head instead — use ``arch_sweep/tta.adapt_head`` there, never this.

    ``batches`` is any iterable of image tensors (a per-frame tensor stack, a list of
    them, or a DataLoader — ``(x, y)`` tuples are unwrapped to ``x``); each is chunked
    to ``chunk`` for the 6 GB-class host. The caller does the tiling (deviation from
    the original's path-based signature so this stays decoupled from file I/O).
    Returns the number of tiles adapted on. torch is imported lazily.
    """
    import torch
    import torch.nn as nn

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None    # cumulative target stats
            mod.train()
    n = 0
    with torch.no_grad():
        for b in batches:
            if isinstance(b, (list, tuple)):   # DataLoader yielding (x, y)
                b = b[0]
            for i in range(0, len(b), chunk):
                model(b[i:i + chunk].to(device))
                n += min(chunk, len(b) - i)
    model.eval()
    if verbose:
        print(f"AdaBN: recomputed BatchNorm stats on {n} target tiles")
    return n


# ---------------------------------------------------------------------------
# Tiling-provenance manifest  (R6 — the anti-silent-drift primitive)
# Atomic-write pattern modeled on arch_sweep/common.py:443-465.
# ---------------------------------------------------------------------------
PROVENANCE_NAME = "_provenance.json"
REQUIRED_PROVENANCE_KEYS = ("tile_px", "tile_save_px", "prep_max", "jpeg_quality",
                            "veg_thresh", "source_digest", "created_at")


def source_digest(paths: Iterable) -> str:
    """Content digest of the tiler's SOURCE frames: sha1 over sorted (name, size, mtime).

    Recipe: for each path, one line ``<name>|<st_size>|<st_mtime_ns>``, sorted by file
    name, sha1 over the concatenation. Cheap (no pixel reads) yet catches the drift
    class that matters: re-prepping at a different ``PREP_MAX`` rewrites the frames
    (new size + mtime under the same names), so the digest — and therefore the
    provenance hash the feature cache signs — changes even though filenames don't.
    """
    h = hashlib.sha1()
    entries = []
    for p in paths:
        p = Path(p)
        st = p.stat()
        entries.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
    for line in sorted(entries):
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


def git_state(repo_dir=None) -> str:
    """``git describe --always --dirty`` if cheaply available, else ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo_dir or Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except Exception:
        return "unknown"


def write_provenance(dataset_dir, params: dict) -> Path:
    """Atomically write ``<dataset_dir>/_provenance.json`` (tmp -> fsync -> os.replace).

    ``params`` must carry at least ``REQUIRED_PROVENANCE_KEYS`` (``created_at`` is an
    ISO string supplied BY THE CALLER — this helper never reads the clock, keeping it
    pure and testable). ``git`` is filled from ``git_state()`` when absent. The write
    pattern mirrors arch_sweep/common.py:443-465: the final file only ever appears via
    an atomic rename of a fully-written, fsynced temp file, so an interrupt leaves
    either the previous manifest or nothing — never a truncated one. Returns the path.
    """
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in params]
    if missing:
        raise ValueError(f"provenance manifest missing required keys: {missing}")
    manifest = dict(params)
    manifest.setdefault("git", git_state())
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    final = dataset_dir / PROVENANCE_NAME
    fd, tmp = tempfile.mkstemp(dir=dataset_dir, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, sort_keys=True, indent=2, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)   # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return final


def read_provenance(dataset_dir) -> dict:
    """Read ``<dataset_dir>/_provenance.json`` back (round-trips ``write_provenance``)."""
    return json.loads((Path(dataset_dir) / PROVENANCE_NAME).read_text())


def provenance_hash(manifest: dict) -> str:
    """Stable sha1 hex over the canonical-JSON form of a manifest dict.

    ``sort_keys=True`` makes the hash independent of key order, so the feature-cache
    signature and the deploy bundle can both embed it and compare across writers.
    """
    return hashlib.sha1(
        json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dataset enumeration + self-check
# ---------------------------------------------------------------------------
def enumerate_tiles(data_dir: str = "tiles_dataset") -> tuple[list[tuple], list[str], int]:
    """``(samples, classes, cog_idx)`` from a torchvision ImageFolder (lazy import).

    Ported from arch_sweep/common.py:136-147 so the split helpers, the baseline, and
    this module all consume the identical ``(path, label)`` ordering.
    """
    from torchvision import datasets   # lazy: pure helpers stay stdlib+numpy-only

    folder = datasets.ImageFolder(data_dir)
    cog_idx = folder.classes.index(COG_CLASS)
    return list(folder.samples), list(folder.classes), cog_idx


def _self_check(data_dir: str = "tiles_dataset") -> None:
    """Print per-collection frame/tile counts and the split summary for a live dataset."""
    samples, classes, cog_idx = enumerate_tiles(data_dir)
    print(f"classes: {classes}  (cog_idx={cog_idx})")
    print(f"heldout_dates: {HELDOUT_DATES}")

    per_date: dict[str, dict] = {}
    for p, lab in samples:
        g = group_of(p)
        d = per_date.setdefault(date_of(g), {"frames": set(), "tiles": 0, "cog": 0})
        d["frames"].add(g)
        d["tiles"] += 1
        d["cog"] += int(lab == cog_idx)
    for date in sorted(per_date):
        d = per_date[date]
        held = "  [HELD OUT]" if date in HELDOUT_DATES else ""
        print(f"collection {date}: {len(d['frames'])} frames | {d['tiles']} tiles "
              f"({d['cog']} cogongrass / {d['tiles'] - d['cog']} not_cogongrass){held}")

    tr, va, te, (nf_tr, nf_va, nf_te) = split_by_collection(samples, cog_idx)
    print(f"split_by_collection -> train {nf_tr} frames/{len(tr)} tiles | "
          f"val {nf_va} frames/{len(va)} tiles | TEST(held-out) {nf_te} frames/{len(te)} tiles")
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    held_idx = {i for i, (p, _) in enumerate(samples)
                if date_of(group_of(p)) in HELDOUT_DATES}
    assert set(te) == held_idx, "test slice must be exactly the held-out collections"

    g_tr, g_va, g_te, (ng_tr, ng_va, ng_te) = grouped_split(samples, cog_idx)
    print(f"grouped_split       -> train {ng_tr} frames/{len(g_tr)} tiles | "
          f"val {ng_va} frames/{len(g_va)} tiles | test {ng_te} frames/{len(g_te)} tiles")
    assert set(g_tr).isdisjoint(g_va) and set(g_tr).isdisjoint(g_te) and set(g_va).isdisjoint(g_te)
    print("self-check OK")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--self-check"]
    _self_check(args[0] if args else "tiles_dataset")
