"""Shared train + eval loop every model script calls (U4).

``train_and_eval(cfg)`` is the single place a sweep cell is trained and scored, so every
model script is comparable by construction (KTD7): same frame-grouped 0606->0422 split,
same balance/early-stop recipe, same seed, same metric code, same honest-threshold rule,
same crash-safe result write. A model script supplies only *which backbone* (and ablation
knobs) — never its own split, metric, or writer.

Protocol enforced here:
- Train a head on **0606** features (majority class down-sampled to parity), early-stop on
  the 0606 validation balanced accuracy.
- **Fit the operating threshold on 0606 only** (``pick_threshold_on`` over the 0606 val
  scores) and record it; the headline ``balanced_accuracy`` is the argmax number on 0422,
  directly comparable to the 0.804 / 0.817 baselines. 0422 labels are **never** read to
  select anything.
- Evaluate on the held-out **0422** collection (natural distribution) and
  ``write_result_atomic`` immediately (eval_setting = cross_collection).

Frozen mode (cached features) is implemented here; ``lora`` / ``full`` tuning modes are
added in U6 and raise a clear error until then.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backbones as B  # noqa: E402
import common as C  # noqa: E402
import features as FEAT  # noqa: E402
import heads as H  # noqa: E402


@dataclass
class TrainConfig:
    """One sweep cell's full configuration (identity + hyperparameters)."""

    model: str                              # backbone name in backbones.REGISTRY
    variant: str = "reference"
    tuning_mode: str = "frozen"             # frozen | lora | full (lora/full -> U6)
    head: str = "mlp_bn"
    adaptation: str = "none"
    eval_setting: str = C.EVAL_CROSS
    seed: int = C.DEFAULT_SEED
    extra: str = ""
    # hyperparameters (fixed across models for comparability; KTD7)
    dropout: float = 0.4
    hidden: int = 512
    lr: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 256
    # non-frozen (lora/full) image-path hyperparameters (U6)
    lora_rank: int = 8
    lr_finetune: float = 1e-4
    ft_batch: int = 32
    ft_epochs: int = 8
    ft_patience: int = 3
    augment: bool = False        # domain-randomization + MixStyle on the train path (U7)

    def identity(self) -> dict:
        return {k: getattr(self, k) for k in C.IDENTITY_FIELDS}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_head(features: np.ndarray, labels: np.ndarray, tr_idx, va_idx, cog_idx, cfg: TrainConfig):
    """Train a head on cached features; early-stop on 0606 val balanced accuracy.

    Returns ``(head, best_val_bacc, n_trainable)``. Pure function of (features, split, cfg,
    seed) so the same inputs reproduce the same head (the determinism invariant).
    """
    import torch
    import torch.nn as nn

    C.set_global_seed(cfg.seed)
    device = _device()
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    in_dim = X.shape[1]
    head = H.build_head(cfg.head, in_dim, n_classes=2, dropout=cfg.dropout, hidden=cfg.hidden).to(device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)

    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=3)

    tr_idx = list(tr_idx)
    rng = np.random.RandomState(cfg.seed)
    best_bacc, best_state, since = -1.0, copy.deepcopy(head.state_dict()), 0
    for _epoch in range(cfg.max_epochs):
        head.train()
        order = rng.permutation(len(tr_idx))
        idx = [tr_idx[i] for i in order]
        for s in range(0, len(idx), cfg.batch_size):
            b = idx[s:s + cfg.batch_size]
            if len(b) < 2:    # BatchNorm needs >1 sample
                continue
            xb, yb = X[b].to(device), y[b].to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
        # val balanced accuracy on the 0606 val slice
        vb = _balanced_acc_on(head, X, y, va_idx, cog_idx, device)
        sched.step(vb)
        if vb > best_bacc + 1e-4:
            best_bacc, best_state, since = vb, copy.deepcopy(head.state_dict()), 0
        else:
            since += 1
            if since >= cfg.patience:
                break
    head.load_state_dict(best_state)
    return head, float(best_bacc), int(n_trainable)


def _probs(head, X, idx, cog_idx, device):
    import torch
    head.eval()
    with torch.no_grad():
        logits = head(X[list(idx)].to(device))
        return logits.softmax(1)[:, cog_idx].float().cpu().numpy()


def _balanced_acc_on(head, X, y, idx, cog_idx, device):
    p = _probs(head, X, idx, cog_idx, device)
    y_true = [1 if int(y[i]) == cog_idx else 0 for i in idx]
    y_pred = [1 if pc >= 0.5 else 0 for pc in p]
    if len(set(y_true)) < 2:
        return 0.0
    return C.balanced_accuracy(y_true, y_pred)


def _make_ok_row(cfg, *, p_val, y_val, p_te, y_te, val_bacc, n_train, n_val, n_test,
                 n_trainable, identity_overrides=None) -> C.ResultRow:
    """Shared scoring tail: 0606-fit threshold + 0422 metrics -> an ok ResultRow.

    Used by the frozen, fine-tune (U6), and TTA (U8) paths so every cell is scored
    identically (KTD7). ``identity_overrides`` lets a TTA cell stamp e.g. adaptation=adabn.
    """
    threshold = C.pick_threshold_on(y_val, p_val) if len(set(y_val)) == 2 else 0.5
    y_pred = [1 if pc >= 0.5 else 0 for pc in p_te]   # argmax — comparable to baselines
    rec = C.per_class_recall(y_te, y_pred)
    both = len(set(y_te)) == 2
    identity = {**cfg.identity(), **(identity_overrides or {})}
    return C.ResultRow(
        **identity, status="ok",
        balanced_accuracy=C.balanced_accuracy(y_te, y_pred),
        recall_cogongrass=rec[C.COG_CLASS], recall_not_cogongrass=rec["not_cogongrass"],
        auroc=C.auroc(y_te, p_te) if both else None,
        average_precision=C.average_precision(y_te, p_te) if both else None,
        threshold=threshold, val_balanced_accuracy=val_bacc, f2_sweep=C.f2_sweep(y_te, p_te),
        n_train=n_train, n_val=n_val, n_test=n_test, n_cog_test=int(sum(y_te)),
        trainable_params=n_trainable)


def _train_frozen(cfg, samples, features, labels, cog_idx, tr_bal, va_idx, te_idx):
    """Frozen path: train a head on cached features, optionally adapt at test time (U8).

    The operating threshold is fit on the 0606 val scores of the **source** head (before any
    TTA); then, if ``cfg.adaptation`` names a TTA method, the head's BatchNorm is adapted to
    the unlabeled 0422 features and the 0422 scores are taken from the adapted head.
    Returns (p_val, p_te, val_bacc, n_trainable).
    """
    import torch
    head, val_bacc, n_trainable = train_head(features, labels, tr_bal, va_idx, cog_idx, cfg)
    device = _device()
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    p_val = _probs(head, X, va_idx, cog_idx, device)            # source head -> honest threshold
    if cfg.adaptation != "none":
        import tta
        tta.adapt_head(head, X[list(te_idx)].to(device), cfg.adaptation)   # label-free target adapt
    return p_val, _probs(head, X, te_idx, cog_idx, device), val_bacc, n_trainable


def _train_finetune(cfg, cog_idx, tr_bal, va_idx, te_idx):
    """Non-frozen path (U6): fine-tune the backbone (lora/full) end-to-end through the images.

    Bypasses the feature cache (KTD2) — the backbone is updated — with a bounded batch for the
    VRAM budget (KTD6). Records the trainable-param count. Returns (p_val, p_te, val_bacc, n).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    C.set_global_seed(cfg.seed)
    device = _device()
    ext = B.get(cfg.model).build()
    model = ext.model
    if cfg.tuning_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif cfg.tuning_mode == "lora":
        B.inject_lora(model, rank=cfg.lora_rank)   # freezes base, adds low-rank adapters
    else:
        raise ValueError(f"_train_finetune got tuning_mode={cfg.tuning_mode!r}")
    model = model.to(device)        # ensure any newly-added params (LoRA) land on the GPU
    model.train()
    head = H.build_head(cfg.head, ext.feature_dim, 2, dropout=cfg.dropout, hidden=cfg.hidden).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    n_trainable = sum(p.numel() for p in params)

    variant_dir = FEAT._variant_dir(cfg.variant)
    ds_eval = datasets.ImageFolder(variant_dir, transform=ext.preprocess)   # pure: no augmentation
    mixstyle = None
    if cfg.augment:                                # train-path only (0606); eval stays pure (U7)
        import augment
        ds_train = datasets.ImageFolder(variant_dir, transform=augment.domain_randomization())
        mixstyle = augment.make_mixstyle().to(device)
    else:
        ds_train = ds_eval
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(params, lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

    def forward(x):
        feats = ext.forward_features(x)
        if mixstyle is not None:
            feats = mixstyle(feats)               # no-op in eval mode (U7)
        return head(feats)

    def predict(idx):
        model.eval(); head.eval()
        if mixstyle is not None:
            mixstyle.eval()
        ps = []
        loader = DataLoader(Subset(ds_eval, list(idx)), cfg.ft_batch, shuffle=False, num_workers=2)
        with torch.no_grad():
            for x, _ in loader:
                ps.append(forward(x.to(device)).softmax(1)[:, cog_idx].float().cpu().numpy())
        return np.concatenate(ps) if ps else np.zeros(0)

    y_val = [1 if int(ds_eval.samples[i][1]) == cog_idx else 0 for i in va_idx]
    tr_loader = DataLoader(Subset(ds_train, tr_bal), cfg.ft_batch, shuffle=True, num_workers=2)
    best_bacc, best, since = -1.0, None, 0
    import copy
    for _epoch in range(cfg.ft_epochs):
        model.train(); head.train()
        if mixstyle is not None:
            mixstyle.train()
        for x, y in tr_loader:
            if x.size(0) < 2:
                continue
            opt.zero_grad()
            crit(forward(x.to(device)), y.to(device)).backward()
            opt.step()
        p_val = predict(va_idx)
        vb = (C.balanced_accuracy(y_val, [1 if p >= 0.5 else 0 for p in p_val])
              if len(set(y_val)) == 2 else 0.0)
        if vb > best_bacc + 1e-4:
            best_bacc, since = vb, 0
            best = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            since += 1
            if since >= cfg.ft_patience:
                break
    if best is not None:
        model.load_state_dict(best[0]); head.load_state_dict(best[1])
    return predict(va_idx), predict(te_idx), float(max(best_bacc, 0.0)), n_trainable


def train_and_eval(cfg: TrainConfig, *, results_dir=C.RESULTS_DIR, samples=None,
                   features=None, labels=None, cog_idx=None,
                   write_scores=False) -> C.ResultRow:
    """Train one cell on 0606 and evaluate on held-out 0422; write the row.

    Dispatches on ``tuning_mode``: ``frozen`` trains a head on cached features (U4);
    ``lora`` / ``full`` fine-tune the backbone through the images (U6). ``samples`` /
    ``features`` / ``labels`` / ``cog_idx`` are injectable so the frozen loop is testable on a
    tiny synthetic feature set. Always writes a result row (even on failure) so the
    orchestrator's coverage accounting is honest (KTD8).

    ``write_scores`` (U1) additionally persists a ``<job_id>.scores.jsonl`` sidecar with one
    per-tile ``P(cogongrass)`` for every 0422 tile, so label-cleaning (U3) / ensembling (U5) /
    self-training (U11) consume stored scores instead of recomputing (KTD3). Off by default so
    tiny unit tests don't require it; ``run_cli`` turns it on for real runs.
    """
    import torch  # noqa: F401 — ensures the ML stack is present before we start

    score_payload = None
    try:
        if features is None and cfg.tuning_mode == "frozen":
            data_dir = FEAT._variant_dir(cfg.variant)
            samples, _classes, cog_idx = C.enumerate_tiles(data_dir)
            cache = FEAT.extract_and_cache(cfg.model, cfg.variant, batch_size=64)
            features, labels = cache["features"], cache["labels"]
        elif samples is None:
            samples, _classes, cog_idx = C.enumerate_tiles(FEAT._variant_dir(cfg.variant))
        if cog_idx is None:
            cog_idx = 0
        if labels is not None:
            labels = np.asarray(labels)

        tr_idx, va_idx, te_idx, _ = C.split_by_collection(samples, cog_idx, cfg.seed)
        import random
        tr_bal = C.balance(tr_idx, samples, cog_idx, random.Random(cfg.seed))

        if cfg.tuning_mode == "frozen":
            p_val, p_te, val_bacc, n_trainable = _train_frozen(
                cfg, samples, features, labels, cog_idx, tr_bal, va_idx, te_idx)
            lab = labels
        else:
            p_val, p_te, val_bacc, n_trainable = _train_finetune(cfg, cog_idx, tr_bal, va_idx, te_idx)
            lab = np.asarray([s[1] for s in samples])

        y_val = [1 if int(lab[i]) == cog_idx else 0 for i in va_idx]
        y_te = [1 if int(lab[i]) == cog_idx else 0 for i in te_idx]
        row = _make_ok_row(cfg, p_val=p_val, y_val=y_val, p_te=p_te, y_te=y_te,
                           val_bacc=val_bacc, n_train=len(tr_bal), n_val=len(va_idx),
                           n_test=len(te_idx), n_trainable=n_trainable)
        if write_scores:   # U1: persist per-tile P(cog) for every 0422 tile (KTD3)
            te_paths = [samples[i][0] for i in te_idx]
            score_payload = C.build_score_records(te_paths, y_te, p_te)
    except Exception as e:  # noqa: BLE001 — record the failure, never lose the cell (KTD8)
        status = "oom" if _is_oom(e) else "failed"
        row = C.ResultRow(**cfg.identity(), status=status, error=f"{type(e).__name__}: {e}"[:500])

    C.write_result_atomic(row, results_dir)
    if score_payload is not None:   # written after the row, only on success; never fails a cell
        C.write_scores_atomic(cfg.identity(), score_payload, results_dir)   # dict -> row's job_id
    return row


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or type(e).__name__ == "OutOfMemoryError"


def build_cli_parser(model: str | None, add_size: bool) -> argparse.ArgumentParser:
    """Shared ablation-arg parser for the per-model scripts (U5). One source, no drift."""
    import backbones as B

    ap = argparse.ArgumentParser(description=f"Train + test {model or 'a model'} on 0606->0422")
    ap.add_argument("--variant", default="reference", help="data variant (see data_variants.py)")
    ap.add_argument("--head", default="mlp_bn", choices=H.HEAD_TYPES)
    ap.add_argument("--mode", dest="tuning_mode", default="frozen",
                    choices=["frozen", "lora", "full"], help="tuning mode (lora/full -> U6)")
    ap.add_argument("--adaptation", default="none", help="test-time adaptation (-> U8)")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--augment", action="store_true",
                    help="domain-randomization + MixStyle on the train path (U7; lora/full)")
    ap.add_argument("--smoke", action="store_true",
                    help="fit gate: load the backbone + train/test on a tiny stratified subset; "
                         "writes under results/smoke (never the real results)")
    ap.add_argument("--smoke-frames", type=int, default=10, help="frames/class/collection in --smoke")
    ap.add_argument("--smoke-tiles", type=int, default=6, help="tiles/frame in --smoke")
    if add_size:
        ap.add_argument("--size", default="l", choices=list(B.DINOV3_SIZES),
                        help="DINOv3 backbone size")
    return ap


def config_from_args(model: str | None, args) -> TrainConfig:
    """Turn parsed CLI args into a TrainConfig (resolving DINOv3 size to a backbone name)."""
    import backbones as B

    parts = []
    resolved = model
    if getattr(args, "size", None) is not None:
        resolved = B.dinov3_name(args.size)
        parts.append(f"size={args.size}")
    augment = getattr(args, "augment", False)
    if augment:
        parts.append("aug")        # in `extra` so an augmented cell gets a distinct job_id
    return TrainConfig(model=resolved, variant=args.variant, head=args.head,
                       tuning_mode=args.tuning_mode, adaptation=args.adaptation,
                       seed=args.seed, extra=",".join(parts), augment=augment)


SMOKE_DIR = C.RESULTS_DIR / "smoke"
SMOKE_FEAT = SMOKE_DIR / "features"


def smoke_subset(samples, cog_idx, frames_per_class=10, tiles_per_frame=6,
                 seed=C.DEFAULT_SEED) -> list[int]:
    """Pick a tiny frame-grouped subset spanning both collections + both classes.

    Frames (not tiles) are sampled so the 0606/0422 split stays leakage-free at smoke scale;
    each chosen frame contributes a class-balanced handful of tiles. Enough 0606 frames are
    kept that the 0606 val slice is non-empty, so the real train/threshold path is exercised.
    """
    import random

    rng = random.Random(seed)
    by_frame: dict[str, dict] = {}
    for i, (p, lab) in enumerate(samples):
        f = C.frame_of(p)
        d = by_frame.setdefault(f, {"date": C.date_of(f), "tiles": [], "has_cog": False})
        d["tiles"].append((i, lab))
        if lab == cog_idx:
            d["has_cog"] = True
    chosen: list[int] = []
    for date in (C.TRAIN_DATE, C.TEST_DATE):
        frames = [f for f, d in by_frame.items() if d["date"] == date]
        pos = [f for f in frames if by_frame[f]["has_cog"]]
        neg = [f for f in frames if not by_frame[f]["has_cog"]]
        rng.shuffle(pos)
        rng.shuffle(neg)
        for f in pos[:frames_per_class] + neg[:frames_per_class]:
            tiles = by_frame[f]["tiles"]
            cog = [i for i, lab in tiles if lab == cog_idx]
            non = [i for i, lab in tiles if lab != cog_idx]
            half = max(1, tiles_per_frame // 2)
            take = (cog[:half] + non[:half]) or [i for i, _ in tiles[:tiles_per_frame]]
            chosen.extend(take[:tiles_per_frame])
    return chosen


def run_smoke(cfg: TrainConfig, frames_per_class=10, tiles_per_frame=6) -> C.ResultRow:
    """Fit gate for one cell: build the real backbone, extract a tiny subset, train+test.

    Writes under ``results/smoke`` so a fit-gate pass never pollutes the real sweep results.
    Returns the row (``status == "ok"`` means the whole real path works for this backbone).
    """
    import features as FEAT

    data_dir = FEAT._variant_dir(cfg.variant)
    samples, _classes, cog_idx = C.enumerate_tiles(data_dir)
    sub_idx = smoke_subset(samples, cog_idx, frames_per_class, tiles_per_frame, cfg.seed)
    sub = [samples[i] for i in sub_idx]
    print(f"[smoke] {cfg.model}: extracting {len(sub)} tiles "
          f"({sum(1 for _, l in sub if l == cog_idx)} cogongrass) ...", flush=True)
    cache = FEAT.extract_and_cache(cfg.model, cfg.variant, samples=sub, overwrite=True,
                                   batch_size=32, cache_dir=SMOKE_FEAT)
    return train_and_eval(cfg, results_dir=SMOKE_DIR, samples=sub,
                          features=cache["features"], labels=cache["labels"], cog_idx=cog_idx)


def run_cli(model: str | None = None, *, add_size: bool = False, argv=None) -> C.ResultRow:
    """Entry point every per-model script calls: parse ablation args, train+eval, report.

    The script supplies only the backbone name (and ``add_size`` for DINOv3). All split,
    metric, threshold, and result-writing logic lives in ``common`` / ``trainer`` — the
    script defines none of it (KTD1, enforced by tests/test_model_scripts.py).
    """
    args = build_cli_parser(model, add_size).parse_args(argv)
    cfg = config_from_args(model, args)
    smoke = getattr(args, "smoke", False)
    print(f"== {'[smoke] ' if smoke else ''}{cfg.model}  variant={cfg.variant} head={cfg.head} "
          f"mode={cfg.tuning_mode} adaptation={cfg.adaptation} seed={cfg.seed} ==", flush=True)
    if smoke:
        row = run_smoke(cfg, args.smoke_frames, args.smoke_tiles)
    else:
        row = train_and_eval(cfg, write_scores=True)   # U1: real runs persist per-tile scores
    if row.status == "ok":
        print(f"\n0422 balanced accuracy: {row.balanced_accuracy:.3f}  "
              f"(recall cog {row.recall_cogongrass:.3f} / not {row.recall_not_cogongrass:.3f})  "
              f"AUROC {row.auroc if row.auroc is None else round(row.auroc, 3)}  "
              f"op-threshold(0606) {row.threshold:.3f}")
        for name, bacc in C.BASELINES:
            print(f"  baseline {name}: {bacc:.3f}")
    else:
        print(f"\n[{row.status}] {row.error}")
    print(f"wrote result -> {C.result_path(row, SMOKE_DIR if smoke else C.RESULTS_DIR)}")
    return row
