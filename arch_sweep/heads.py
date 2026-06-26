"""Classifier head registry for the sweep (U4).

Two heads sit on top of the frozen (or fine-tuned) backbone features:

- ``linear``  — dropout + a single linear layer (the lean baseline, like train_tiles_dino).
- ``mlp_bn``  — Linear -> **BatchNorm1d** -> ReLU -> Dropout -> Linear.

The MLP head **carries a BatchNorm** on purpose (KTD5): ViT backbones have none, so this is
what lets the AdaBN / TENT / EATA / RoTTA test-time-adaptation cells (U8) recompute or adapt
normalization stats on the 0422 target. Keep a BatchNorm in any head meant for a deployable
(TTA-able) cell.

Dropout is a head knob here; label-smoothing and weight-decay are training knobs the
trainer (U4) applies, kept together conceptually so a cell's regularization is one config.
"""
from __future__ import annotations

HEAD_TYPES = ["linear", "mlp_bn"]


def build_head(head_type: str, in_dim: int, n_classes: int = 2,
               dropout: float = 0.4, hidden: int = 512):
    """Build a head module. Lazy torch import so this module is import-safe without the stack."""
    import torch.nn as nn

    if head_type == "linear":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
    if head_type == "mlp_bn":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),   # AdaBN/TTA precondition (KTD5)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    raise ValueError(f"unknown head {head_type!r}; choices: {HEAD_TYPES}")


def has_batchnorm(module) -> bool:
    """True if the head contains a BatchNorm (the AdaBN/TTA precondition)."""
    import torch.nn as nn
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in module.modules())
