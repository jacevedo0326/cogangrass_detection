"""Test-time adaptation at inference (U8): AdaBN, TENT, EATA, RoTTA.

Adapts the **BatchNorm head** (KTD5) to the unlabeled 0422 target with no labels — the
realistic deployment setting. Operates in feature space on the frozen cells' cached
features, so it is fully CPU-testable.

- **AdaBN** — recompute BN running stats on the target (the proven +2-pt move; CLAUDE.md).
- **TENT** — entropy-minimize the BN **affine** params on the target (fragile on imbalanced
  single-frame streams — kept for comparison, not preferred).
- **EATA** — TENT but only on **reliable** (low-entropy) samples, so noisy/ambiguous tiles
  can't drag the affine params into a one-class collapse.
- **RoTTA** — robust: AdaBN first (stable stats), then a gentle filtered entropy step.

Only BN affine / running stats are ever touched — the backbone (here, the head's Linear
weights) is never updated. Methods that would collapse to one class are guarded by the
reliable-sample filter (EATA/RoTTA).
"""
from __future__ import annotations

import math

METHODS = ["none", "adabn", "tent", "eata", "rotta"]
_LN2 = math.log(2.0)   # max entropy of a 2-class prediction


def _bn_layers(module):
    import torch.nn as nn
    return [m for m in module.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]


def _entropy(logits):
    import torch
    p = logits.softmax(1)
    return -(p * (p + 1e-9).log()).sum(1)


def adabn(head, feat):
    """Recompute the head's BN running stats on the target features (label-free)."""
    import torch
    bns = _bn_layers(head)
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None      # cumulative moving average over the target pass
    head.train()
    with torch.no_grad():
        head(feat)
    head.eval()
    return head


def tent(head, feat, *, steps=10, lr=1e-3, entropy_filter=None):
    """Entropy-minimize BN affine on the target. ``entropy_filter`` keeps only reliable rows.

    Backbone/Linear weights are frozen — only BN ``weight``/``bias`` receive gradients.
    ``entropy_filter`` (in nats) restricts the loss to low-entropy (confident) samples — the
    EATA/RoTTA guard against one-class collapse on imbalanced streams.
    """
    import torch
    for p in head.parameters():
        p.requires_grad_(False)
    bns = _bn_layers(head)
    params = []
    for bn in bns:
        bn.train()
        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
        params += [bn.weight, bn.bias]
    if not params:
        head.eval()
        return head
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        ent = _entropy(head(feat))
        if entropy_filter is not None:
            mask = ent < entropy_filter
            loss = ent[mask].mean() if int(mask.sum()) >= 2 else ent.mean()
        else:
            loss = ent.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


def adapt_head(head, feat_target, method, *, steps=10, lr=1e-3):
    """Adapt ``head`` to ``feat_target`` by ``method`` (in place). Returns the head.

    ``feat_target`` is a tensor of the unlabeled target (0422) features. ``none`` is a no-op.
    """
    if method == "none":
        return head
    if method == "adabn":
        return adabn(head, feat_target)
    if method == "tent":
        return tent(head, feat_target, steps=steps, lr=lr)
    if method == "eata":
        return tent(head, feat_target, steps=steps, lr=lr, entropy_filter=0.4 * _LN2)
    if method == "rotta":
        adabn(head, feat_target)
        return tent(head, feat_target, steps=max(3, steps // 2), lr=lr / 2, entropy_filter=0.4 * _LN2)
    raise ValueError(f"unknown TTA method {method!r}; choices: {METHODS}")
