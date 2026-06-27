"""Decorrelated backbone ensemble (U5).

Averages per-tile ``P(cogongrass)`` across the top backbones (siglip2 / aimv2 / cradio /
dinov3_sat), each a cheap head trained off that backbone's cached features (KTD2), optionally
EATA-adapted to the 0422 target per member (U8/`tta.py`). The averaged probabilities are then
scored through ``trainer._make_ok_row`` so an ensemble cell is measured *identically* to every
single-backbone cell (KTD7) and merges into the same report (KTD2): one ``ResultRow`` with
``model="ensemble"`` and ``extra="ensemble=<members>"`` (a distinct ``job_id``).

Two free variance-reducers:
- **ensembling** — average probabilities across decorrelated backbones.
- **seed-soup** — average a single backbone's head *weights* over seeds before predicting.

The math (alignment by ``paths`` + probability averaging) is pure and CPU-testable with
injected synthetic feature caches; the real backbones only matter at extract time.

Run:
    python arch_sweep/models/ensemble.py --members aimv2,cradio,siglip2 --adaptation eata
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import trainer as T  # noqa: E402

DEFAULT_MEMBERS = ["aimv2", "cradio", "siglip2", "dinov3_sat"]


def align_by_paths(caches):
    """Align ensemble members to one canonical tile order, keyed by ``paths``.

    Each cache is ``{features, labels, paths}`` (the ``features.load_features`` shape). Members
    must cover the **same tile set**; a mismatch is raised, never silently averaged — averaging
    misaligned rows would mix different tiles' scores. Returns ``(paths, [feats...], labels)``
    with every member's features reindexed to the first member's path order.
    """
    if not caches:
        raise ValueError("no ensemble members")
    ref_paths = list(caches[0]["paths"])
    ref_set = set(ref_paths)
    if len(ref_set) != len(ref_paths):
        raise ValueError("duplicate tile paths in a member cache — cannot align")
    feats_list, labels = [], None
    for c in caches:
        if set(c["paths"]) != ref_set:
            raise ValueError("ensemble members cover different tile sets — cannot align by paths")
        pos = {p: i for i, p in enumerate(c["paths"])}
        order = [pos[p] for p in ref_paths]
        feats_list.append(np.asarray(c["features"])[order])
        if labels is None:
            labels = np.asarray(c["labels"])[order]
    return ref_paths, feats_list, labels


def average_probs(member_probs) -> np.ndarray:
    """Element-wise mean of aligned per-tile probability vectors (the ensemble rule)."""
    return np.mean(np.stack([np.asarray(p, dtype=float) for p in member_probs], axis=0), axis=0)


def _soup(heads):
    """Average head *weights* over seeds (a model soup) -> one head. Cheap variance reduction."""
    import copy

    import torch

    base = copy.deepcopy(heads[0])
    sd = base.state_dict()
    for k in sd:
        stacked = torch.stack([h.state_dict()[k].float() for h in heads], dim=0)
        sd[k] = stacked.mean(0).to(sd[k].dtype)
    base.load_state_dict(sd)
    base.eval()
    return base


def _member_probs(features, labels, samples, cog_idx, cfg, *, adaptation, seeds, seed_soup):
    """Train one member's head(s) on its features and return ``(p_val, p_te, va_idx, te_idx)``.

    With ``seed_soup`` the head is trained at each seed and the weights are souped before
    predicting; otherwise a single head is trained at ``cfg.seed``. If ``adaptation`` names a
    TTA method, the (souped) head's BN is adapted to the unlabeled 0422 features first (KTD5).
    """
    import torch

    tr_idx, va_idx, te_idx, _ = C.split_by_collection(samples, cog_idx, cfg.seed)
    tr_bal = C.balance(tr_idx, samples, cog_idx, random.Random(cfg.seed))
    device = T._device()
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    seed_list = list(seeds) if seed_soup else [cfg.seed]
    heads = []
    for s in seed_list:
        head, _vb, _n = T.train_head(features, labels, tr_bal, va_idx, cog_idx, replace(cfg, seed=s))
        heads.append(head)
    head = _soup(heads) if len(heads) > 1 else heads[0]
    p_val = T._probs(head, X, va_idx, cog_idx, device)
    if adaptation != "none":
        import tta
        tta.adapt_head(head, X[list(te_idx)].to(device), adaptation)   # label-free per-member adapt
    p_te = T._probs(head, X, te_idx, cog_idx, device)
    return p_val, p_te, va_idx, te_idx


def ensemble_extra(members, seed_soup: bool) -> str:
    """Identity discriminator so each ensemble gets a distinct ``job_id`` (KTD2)."""
    tag = "ensemble=" + "+".join(members)
    return tag + (",soup" if seed_soup else "")


def run_ensemble(members=DEFAULT_MEMBERS, *, variant="reference", adaptation="none",
                 seeds=(C.DEFAULT_SEED,), seed_soup=False, results_dir=C.RESULTS_DIR,
                 caches=None, cog_idx=None, write_scores=False, **cfg_kwargs) -> C.ResultRow:
    """Train per-member heads, average per-tile probs, score the ensemble as one cell.

    ``caches`` (one ``{features, labels, paths}`` per member) is injectable so the whole path
    is CPU-testable on synthetic features; left ``None`` it loads each member's real feature
    cache. Always writes a ``ResultRow`` (failure recorded, never lost — KTD8).
    """
    cfg = T.TrainConfig(model="ensemble", variant=variant, adaptation=adaptation,
                        extra=ensemble_extra(members, seed_soup), **cfg_kwargs)
    try:
        if caches is None:
            import features as FEAT
            caches = []
            for b in members:
                c = FEAT.load_features(b, variant)
                if c is None:
                    raise FileNotFoundError(f"no cached features for {b}×{variant} — extract first")
                caches.append(c)
        ref_paths, feats_list, labels = align_by_paths(caches)
        samples = list(zip(ref_paths, [int(x) for x in labels]))
        if cog_idx is None:
            cog_idx = C.CLASSES.index(C.COG_CLASS)
        pvals, ptes, va_ref, te_ref = [], [], None, None
        for feats in feats_list:
            pv, pt, va_idx, te_idx = _member_probs(
                feats, labels, samples, cog_idx, cfg,
                adaptation=adaptation, seeds=seeds, seed_soup=seed_soup)
            pvals.append(pv)
            ptes.append(pt)
            va_ref, te_ref = va_idx, te_idx
        p_val, p_te = average_probs(pvals), average_probs(ptes)
        y_val = [1 if int(labels[i]) == cog_idx else 0 for i in va_ref]
        y_te = [1 if int(labels[i]) == cog_idx else 0 for i in te_ref]
        tr_idx, _va, _te, _ = C.split_by_collection(samples, cog_idx, cfg.seed)
        n_train = len(C.balance(tr_idx, samples, cog_idx, random.Random(cfg.seed)))
        row = T._make_ok_row(cfg, p_val=p_val, y_val=y_val, p_te=p_te, y_te=y_te,
                             val_bacc=None, n_train=n_train, n_val=len(va_ref),
                             n_test=len(te_ref), n_trainable=None, prior=cfg.prior)
        score_payload = (C.build_score_records([samples[i][0] for i in te_ref], y_te, p_te)
                         if write_scores else None)
    except Exception as e:  # noqa: BLE001 — record, never lose the cell (KTD8)
        status = "oom" if T._is_oom(e) else "failed"
        row = C.ResultRow(**cfg.identity(), status=status, error=f"{type(e).__name__}: {e}"[:500])
        score_payload = None

    C.write_result_atomic(row, results_dir)
    if score_payload is not None:
        C.write_scores_atomic(cfg.identity(), score_payload, results_dir)
    return row


def main():
    ap = argparse.ArgumentParser(description="Decorrelated backbone ensemble (U5)")
    ap.add_argument("--members", default=",".join(DEFAULT_MEMBERS),
                    help="comma-separated backbones (must have cached features)")
    ap.add_argument("--variant", default="reference")
    ap.add_argument("--adaptation", default="none", help="per-member TTA (none|adabn|eata|...)")
    ap.add_argument("--seed-soup", action="store_true", help="average head weights over seeds")
    ap.add_argument("--seeds", default="42,1,2,3", help="seeds for the soup (comma-separated)")
    args = ap.parse_args()
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seed_soup else (C.DEFAULT_SEED,)
    print(f"== ensemble {members}  variant={args.variant} adaptation={args.adaptation} "
          f"seed_soup={args.seed_soup} ==", flush=True)
    row = run_ensemble(members, variant=args.variant, adaptation=args.adaptation,
                       seeds=seeds, seed_soup=args.seed_soup, write_scores=True)
    if row.status == "ok":
        print(f"\n0422 balanced accuracy: {row.balanced_accuracy:.3f}  "
              f"(recall cog {row.recall_cogongrass:.3f})  AUROC "
              f"{row.auroc if row.auroc is None else round(row.auroc, 3)}")
    else:
        print(f"\n[{row.status}] {row.error}")
    print(f"wrote -> {C.result_path(row, results_dir=C.RESULTS_DIR)}")


if __name__ == "__main__":
    main()
