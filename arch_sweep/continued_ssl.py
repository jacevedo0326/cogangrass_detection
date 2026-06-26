"""Continued self-supervised pretraining on unlabeled tiles (U9, ExPLoRA-style).

Adapts a foundation backbone to the cogongrass *domain* using **labels-free** tiles, then
exposes the adapted checkpoint as a normal schedulable backbone so its cross-collection
number is measured like any other model (via models/train_dinov3_ssl.py).

ExPLoRA discipline (avoid catastrophic forgetting): freeze the backbone, inject LoRA on the
attention/projection layers, and additionally unfreeze only the **last 1-2 transformer
blocks**. The SSL objective is a self-distillation (SimSiam) negative-cosine loss on two
augmented views — no labels are ever read. Checkpoints are written **atomically**
(temp -> os.replace) with periodic checkpointing so a multi-hour run resumes.

Real SSL needs the GPU; the freeze discipline, atomic checkpoint round-trip, and unlabeled
loop are unit-tested with a stub on CPU (plan U9 execution note: run a tiny-step smoke first).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backbones as B  # noqa: E402


# ---------------------------------------------------------------------------
# Freeze discipline: LoRA everywhere + last-N blocks fully trainable
# ---------------------------------------------------------------------------
def find_blocks(model):
    """Locate the transformer block list across ViT families (dino/timm/HF)."""
    for attr in ("blocks", "layers"):
        b = getattr(model, attr, None)
        if b is not None:
            return b
    enc = getattr(model, "encoder", None)
    if enc is not None:
        for attr in ("layer", "layers", "blocks"):
            b = getattr(enc, attr, None)
            if b is not None:
                return b
    return None


def freeze_for_ssl(model, n_blocks: int = 2, lora_rank: int = 8) -> dict:
    """Freeze the backbone, add LoRA adapters, and unfreeze the last ``n_blocks`` blocks.

    Returns a dict with the wrapped-LoRA count and the trainable-param count. After this only
    the LoRA adapters and the trailing blocks carry gradients (ExPLoRA), so the bulk of the
    pretrained backbone is preserved.
    """
    n_lora = B.inject_lora(model, rank=lora_rank)   # freezes all params, adds trainable adapters
    blocks = find_blocks(model)
    unfrozen_blocks = 0
    if blocks is not None and len(blocks):
        for blk in list(blocks)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
            unfrozen_blocks += 1
    return {"lora_layers": n_lora, "unfrozen_blocks": unfrozen_blocks,
            "trainable_params": B.count_trainable(model)}


# ---------------------------------------------------------------------------
# SimSiam self-distillation objective (no labels)
# ---------------------------------------------------------------------------
def make_ssl_head(in_dim: int, proj_dim: int = 256, hidden: int = 512):
    """Projector + predictor MLPs for SimSiam (the only newly-initialized SSL params)."""
    import torch.nn as nn

    projector = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                              nn.Linear(hidden, proj_dim))
    predictor = nn.Sequential(nn.Linear(proj_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                              nn.Linear(hidden, proj_dim))
    return projector, predictor


def simsiam_loss(p1, z1, p2, z2):
    """Symmetric negative-cosine SimSiam loss with stop-grad on the targets."""
    import torch.nn.functional as F

    def d(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=1).mean()

    return 0.5 * d(p1, z2) + 0.5 * d(p2, z1)


# ---------------------------------------------------------------------------
# Atomic checkpointing (crash-safe, resumable)
# ---------------------------------------------------------------------------
def save_checkpoint_atomic(state: dict, path: str | Path, meta: dict | None = None) -> Path:
    """Write a torch checkpoint via temp -> fsync -> os.replace (no partial final on interrupt)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "meta": meta or {}}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-ckpt-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)       # atomic
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_checkpoint(path: str | Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# SSL loop (loader yields image batches; labels never read — unlabeled path)
# ---------------------------------------------------------------------------
def run_ssl(forward_features, two_views, loader, *, in_dim, steps, out_path,
            lr=1e-4, device="cpu", checkpoint_every=200, state_fn=None):
    """Run the SimSiam loop over an unlabeled loader; checkpoint atomically.

    ``forward_features(x)`` -> pooled features (grad-capable); ``two_views(x)`` -> (v1, v2).
    The loader may yield ``(x, label)`` or ``x`` — the label is **never** read (unlabeled).
    ``state_fn()`` returns the state dict to checkpoint (the adapted backbone). Returns the
    number of optimizer steps taken.
    """
    import torch

    projector, predictor = make_ssl_head(in_dim)
    projector, predictor = projector.to(device), predictor.to(device)
    params = [p for p in projector.parameters()] + [p for p in predictor.parameters()]
    if state_fn is not None:
        params += [p for p in _params_of(state_fn)]
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)

    done = 0
    while done < steps:
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch   # ignore any label
            x = x.to(device)
            v1, v2 = two_views(x)
            z1, z2 = projector(forward_features(v1)), projector(forward_features(v2))
            p1, p2 = predictor(z1), predictor(z2)
            loss = simsiam_loss(p1, z1, p2, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            done += 1
            if state_fn is not None and done % checkpoint_every == 0:
                save_checkpoint_atomic(state_fn(), out_path, {"steps": done})
            if done >= steps:
                break
    if state_fn is not None:
        save_checkpoint_atomic(state_fn(), out_path, {"steps": done})
    return done


def _params_of(state_fn):
    """Trainable params behind a state_fn that also exposes ``.parameters`` (a module closure)."""
    mod = getattr(state_fn, "module", None)
    return [p for p in mod.parameters() if p.requires_grad] if mod is not None else []


def load_ssl_backbone(base_name: str, ckpt_path: str | Path):
    """Build the base backbone extractor and load the continued-SSL adapted weights into it."""
    spec = B.get(base_name)
    ext = spec.build()
    ckpt = load_checkpoint(ckpt_path)
    # LoRA was injected during SSL, so rebuild the same structure before loading.
    B.inject_lora(ext.model, rank=ckpt.get("meta", {}).get("lora_rank", 8))
    ext.model.load_state_dict(ckpt["state"], strict=False)
    return ext
