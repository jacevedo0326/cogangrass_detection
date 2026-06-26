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

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def train_and_eval(cfg: TrainConfig, *, results_dir=C.RESULTS_DIR, samples=None,
                   features=None, labels=None, cog_idx=None) -> C.ResultRow:
    """Train one cell on 0606 frozen features and evaluate on held-out 0422; write the row.

    ``samples`` / ``features`` / ``labels`` / ``cog_idx`` are injectable so the loop is
    testable on a tiny synthetic feature set; in production they are loaded from the
    per-(backbone, variant) feature cache (U3). Always writes a result row (even on failure)
    so the orchestrator's coverage/failure accounting is honest (KTD8).
    """
    import torch  # noqa: F401 — ensures the ML stack is present before we start

    try:
        if cfg.tuning_mode != "frozen":
            raise NotImplementedError(
                f"tuning_mode={cfg.tuning_mode!r} is added in U6; U4 implements frozen cells")
        if features is None:
            data_dir = FEAT._variant_dir(cfg.variant)
            samples, classes, cog_idx = C.enumerate_tiles(data_dir)
            cache = FEAT.extract_and_cache(cfg.model, cfg.variant, batch_size=64)
            features, labels = cache["features"], cache["labels"]
        if cog_idx is None:
            cog_idx = 0
        labels = np.asarray(labels)

        tr_idx, va_idx, te_idx, (nf_tr, nf_va, nf_te) = C.split_by_collection(samples, cog_idx, cfg.seed)
        import random
        rng = random.Random(cfg.seed)
        tr_bal = C.balance(tr_idx, samples, cog_idx, rng)

        head, val_bacc, n_trainable = train_head(features, labels, tr_bal, va_idx, cog_idx, cfg)

        device = _device()
        X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        # operating threshold: fit on 0606 val scores ONLY (never the 0422 slice).
        p_val = _probs(head, X, va_idx, cog_idx, device)
        y_val = [1 if int(labels[i]) == cog_idx else 0 for i in va_idx]
        threshold = C.pick_threshold_on(y_val, p_val) if len(set(y_val)) == 2 else 0.5

        # evaluate on held-out 0422 (natural distribution).
        p_te = _probs(head, X, te_idx, cog_idx, device)
        y_te = [1 if int(labels[i]) == cog_idx else 0 for i in te_idx]
        y_pred = [1 if pc >= 0.5 else 0 for pc in p_te]   # argmax — comparable to baselines
        rec = C.per_class_recall(y_te, y_pred)
        both = len(set(y_te)) == 2

        row = C.ResultRow(
            **cfg.identity(), status="ok",
            balanced_accuracy=C.balanced_accuracy(y_te, y_pred),
            recall_cogongrass=rec[C.COG_CLASS], recall_not_cogongrass=rec["not_cogongrass"],
            auroc=C.auroc(y_te, p_te) if both else None,
            average_precision=C.average_precision(y_te, p_te) if both else None,
            threshold=threshold, val_balanced_accuracy=val_bacc,
            f2_sweep=C.f2_sweep(y_te, p_te),
            n_train=len(tr_bal), n_val=len(va_idx), n_test=len(te_idx),
            n_cog_test=int(sum(y_te)), trainable_params=n_trainable)
    except Exception as e:  # noqa: BLE001 — record the failure, never lose the cell (KTD8)
        status = "oom" if _is_oom(e) else "failed"
        row = C.ResultRow(**cfg.identity(), status=status, error=f"{type(e).__name__}: {e}"[:500])

    C.write_result_atomic(row, results_dir)
    return row


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or type(e).__name__ == "OutOfMemoryError"
