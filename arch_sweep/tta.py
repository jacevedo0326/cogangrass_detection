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

import numpy as np

METHODS = ["none", "adabn", "tent", "eata", "rotta", "sar", "cotta"]
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


def sar(head, feat, *, steps=10, lr=1e-3, entropy_filter=0.4 * _LN2, rho=0.05):
    """SAR — Sharpness-Aware + Reliable-sample TTA (BN affine only).

    Like EATA (entropy-min on low-entropy *reliable* samples), but each step is **sharpness
    aware**: the BN affine params are perturbed along the gradient to the local sharp point
    before the descent step, so the adaptation lands in a flatter, collapse-resistant minimum.
    The backbone / Linear weights never receive gradients.
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
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(steps):
        ent = _entropy(head(feat))
        mask = ent < entropy_filter
        sel = feat[mask] if int(mask.sum()) >= 2 else feat
        opt.zero_grad()
        _entropy(head(sel)).mean().backward()
        with torch.no_grad():
            grads = [p.grad.detach().clone() for p in params]
            norm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
            for p, g in zip(params, grads):
                p.add_(rho * g / norm)            # ascend to the sharp point
        opt.zero_grad()
        _entropy(head(sel)).mean().backward()     # gradient at the perturbed point
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.sub_(rho * g / norm)            # restore, then SGD steps with the sharp gradient
        opt.step()
    head.eval()
    return head


def cotta(head, feat, *, steps=10, lr=1e-3, alpha=0.99):
    """CoTTA — weight-averaged (EMA) teacher provides stable targets; student BN affine adapts.

    AdaBN-warm-starts the student, snapshots an EMA teacher, then minimizes the student's
    cross-entropy to the (frozen-per-step) teacher's soft pseudo-labels while EMA-updating the
    teacher toward the student. The slow teacher resists the error accumulation plain TENT shows
    on long streams. Only BN affine updates.
    """
    import copy

    import torch

    adabn(head, feat)                              # stable normalization start
    teacher = copy.deepcopy(head)
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
        with torch.no_grad():
            target = teacher(feat).softmax(1)      # teacher pseudo-labels
        loss = -(target * head(feat).log_softmax(1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                      # EMA the teacher toward the student
            tsd, ssd = teacher.state_dict(), head.state_dict()
            for k in tsd:
                if tsd[k].dtype.is_floating_point:
                    tsd[k].mul_(alpha).add_((1 - alpha) * ssd[k])
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
    if method == "sar":
        return sar(head, feat_target, steps=steps, lr=lr)
    if method == "cotta":
        return cotta(head, feat_target, steps=steps, lr=lr)
    raise ValueError(f"unknown TTA method {method!r}; choices: {METHODS}")


def predict_episodic(head, feat_target, frames, cog_idx, method, *, base_state=None, **kw):
    """Episodic per-FRAME adaptation (matches ``heatmap_infer.py``'s per-frame AdaBN).

    For each source frame independently: reset the head to ``base_state``, adapt BN on **only
    that frame's** target tiles, and predict them — so a single imbalanced frame can never drag
    the global BN stats into a one-class collapse, and frames don't contaminate each other.
    Returns per-row ``P(cogongrass)`` aligned with ``feat_target`` / ``frames``. Restores the
    head to ``base_state`` on exit.
    """
    import copy

    import torch

    base_state = base_state or copy.deepcopy(head.state_dict())
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        groups.setdefault(f, []).append(i)
    probs = np.zeros(len(frames), dtype=float)
    for idx in groups.values():
        head.load_state_dict(base_state)           # episodic reset per frame
        sub = feat_target[idx]
        adapt_head(head, sub, method, **kw)
        head.eval()
        with torch.no_grad():
            p = head(sub).softmax(1)[:, cog_idx].float().cpu().numpy()
        for j, i in enumerate(idx):
            probs[i] = p[j]
    head.load_state_dict(base_state)
    return probs
