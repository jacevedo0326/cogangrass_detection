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


# ---------------------------------------------------------------------------
# Runnable pretrain entry (U7): wire a real backbone into the unlabeled SSL loop
# ---------------------------------------------------------------------------
def _two_views_fn():
    """Two independent stochastic views of a preprocessed batch (flip + jitter; label-free)."""
    import torch

    def two_views(x):
        def view():
            v = x
            if bool(torch.rand(1).item() < 0.5):
                v = torch.flip(v, dims=[-1])               # random horizontal flip
            return v + 0.02 * torch.randn_like(v)          # light feature jitter
        return view(), view()

    return two_views


def pretrain(base_name: str, *, size: str = "l", variant: str = "reference", steps: int = 2000,
             lora_rank: int = 8, n_blocks: int = 2, batch_size: int = 32, lr: float = 1e-4,
             out_path: str | Path | None = None, smoke: bool = False,
             smoke_tiles: int = 64) -> Path:
    """Continued-SSL pretrain a DINOv3 backbone on unlabeled tiles; write the adapted checkpoint.

    Builds the base backbone, applies the ExPLoRA freeze discipline, and runs the already-tested
    ``run_ssl`` SimSiam loop over the variant's tiles (labels never read). ``smoke`` runs a few
    steps on a tiny subset (the U7/U9 fit gate) to confirm the atomic-checkpoint round-trip
    before the multi-hour run. Returns the checkpoint path.
    """
    import sys as _sys

    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C
    import features as FEAT

    name = B.dinov3_name(size) if base_name in ("dinov3", "dinov3_ssl") else base_name
    ext = B.get(name).build()
    info = freeze_for_ssl(ext.model, n_blocks=n_blocks, lora_rank=lora_rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext.model.to(device).train()
    out_path = Path(out_path) if out_path else (C.RESULTS_DIR / "ssl" / f"dinov3_{size}_ssl.pt")

    ds = datasets.ImageFolder(FEAT._variant_dir(variant), transform=ext.preprocess)
    if smoke:
        steps = min(steps, 5)
        ds = Subset(ds, list(range(min(len(ds), smoke_tiles))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    def forward_features(x):
        return ext.forward_features(x)

    class _StateFn:
        module = ext.model

        def __call__(self):
            return ext.model.state_dict()

    state_fn = _StateFn()
    print(f"[ssl] {name}: {info['trainable_params']} trainable params "
          f"({info['lora_layers']} LoRA layers, {info['unfrozen_blocks']} unfrozen blocks); "
          f"{'SMOKE ' if smoke else ''}steps={steps} -> {out_path}", flush=True)
    done = run_ssl(forward_features, _two_views_fn(), loader, in_dim=ext.feature_dim,
                   steps=steps, out_path=out_path, lr=lr, device=device, state_fn=state_fn)
    # stamp the lora_rank so load_ssl_backbone rebuilds the matching structure
    ckpt = load_checkpoint(out_path)
    save_checkpoint_atomic(ckpt["state"], out_path, {"steps": done, "lora_rank": lora_rank})
    print(f"[ssl] done: {done} steps -> {out_path}")
    return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Continued-SSL pretrain on unlabeled tiles (U7)")
    ap.add_argument("--size", default="l", choices=list(B.DINOV3_SIZES))
    ap.add_argument("--variant", default="reference")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="fit gate: a few steps on a tiny subset, confirm the checkpoint loads")
    args = ap.parse_args()
    path = pretrain("dinov3", size=args.size, variant=args.variant, steps=args.steps,
                    lora_rank=args.lora_rank, n_blocks=args.n_blocks, out_path=args.out,
                    smoke=args.smoke)
    # confirm the checkpoint is loadable as a backbone (the round-trip the smoke gates)
    ext = load_ssl_backbone(B.dinov3_name(args.size), path)
    print(f"[ssl] checkpoint loads as a backbone: feature_dim={ext.feature_dim}")


if __name__ == "__main__":
    main()
