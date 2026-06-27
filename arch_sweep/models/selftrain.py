"""Ensemble-agreement pseudo-label self-training (U11).

The only self-training form that survives the origin's confirmation-bias objection: a tile is
pseudo-labeled **only when >=K of the ensemble's backbones agree at high confidence** (KTD3,
builds on U5). Agreed-confident 0422 tiles are added to the 0606 training pool with their
*own* predicted labels, the heads are retrained off cache, and 2-3 rounds compound.

Protocol preservation (the part that makes this honest):
- The 0422 set is frame-split into a **pseudo-label pool** and a **held-out eval** set; eval
  frames are *strictly* never pseudo-labeled or trained on (no train-on-test leakage).
- Target data enters training only via its own ensemble predictions — never ground-truth labels
  — so the cell stays ``cross_collection`` (tagged ``extra="pseudo=agreeK"`` so the report
  visually separates it from the frozen cells).
- Pseudo-positives are capped to the known prevalence (``prior``) so a round can't drift toward
  the majority class.

Pure agreement/cap logic is CPU-tested; the loop runs on injected synthetic caches like U5.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import fewshot as FS  # noqa: E402
import trainer as T  # noqa: E402
from models import ensemble as E  # noqa: E402

ABSTAIN = -1


def ensemble_agreement(member_probs, *, hi: float = 0.8, lo: float = 0.2,
                       agree_k: int = 3) -> np.ndarray:
    """Per-tile pseudo-label from member agreement: 1 / 0 / ABSTAIN(-1).

    ``member_probs`` is ``(n_members, n_tiles)``. A tile is pseudo-positive when ``>= agree_k``
    members score ``>= hi``, pseudo-negative when ``>= agree_k`` score ``<= lo``, else abstain.
    Disagreement (some high, some low) abstains — confirmation bias is filtered out (KTD3).
    """
    P = np.asarray(member_probs, dtype=float)
    n_hi = (P >= hi).sum(0)
    n_lo = (P <= lo).sum(0)
    out = np.full(P.shape[1], ABSTAIN, dtype=int)
    out[n_hi >= agree_k] = 1
    out[(n_lo >= agree_k) & (n_hi < agree_k)] = 0    # high agreement wins ties (favor recall)
    return out


def cap_positives_to_prior(pseudo: np.ndarray, mean_p, prior: float, n_reference: int) -> np.ndarray:
    """Cap the pseudo-POSITIVES to ``round(prior * n_reference)``, keeping the most confident.

    Prevents a self-training round from drifting toward the majority class: only the
    highest-``mean_p`` agreed positives survive the cap; the rest revert to ABSTAIN. Negatives
    and abstains are untouched.
    """
    pseudo = np.asarray(pseudo).copy()
    mean_p = np.asarray(mean_p, dtype=float)
    pos_idx = np.where(pseudo == 1)[0]
    cap = int(round(max(0.0, min(1.0, prior)) * n_reference))
    if len(pos_idx) > cap:
        keep = pos_idx[np.argsort(-mean_p[pos_idx])[:cap]]
        drop = set(pos_idx.tolist()) - set(keep.tolist())
        for i in drop:
            pseudo[i] = ABSTAIN
    return pseudo


def _member_heads_probs(feats_list, eff_labels, train_idx, va_idx, score_idx, cog_idx, cfg):
    """Train one head per member on ``train_idx`` and return per-member probs on ``score_idx``."""
    import torch

    member_probs = []
    for feats in feats_list:
        head, _vb, _n = T.train_head(feats, eff_labels, train_idx, va_idx, cog_idx, cfg)
        X = torch.as_tensor(np.asarray(feats), dtype=torch.float32)
        member_probs.append(T._probs(head, X, score_idx, cog_idx, T._device()))
    return np.asarray(member_probs)


def run_selftrain(members=E.DEFAULT_MEMBERS, *, variant="reference", rounds=3, hi=0.8, lo=0.2,
                  agree_k=3, prior=None, eval_frac=0.5, seed=C.DEFAULT_SEED,
                  results_dir=C.RESULTS_DIR, caches=None, cog_idx=None, write_scores=False,
                  **cfg_kwargs) -> C.ResultRow:
    """Iterate ensemble-agreement self-training; evaluate on the held-out 0422 eval frames.

    ``caches`` (one ``{features, labels, paths}`` per member) is injectable for CPU testing.
    Returns one ``cross_collection`` ResultRow tagged ``extra="pseudo=agreeK"``. Records its
    per-round balanced accuracy in ``f2_sweep``-adjacent fields via the standard scoring tail.
    """
    cfg = T.TrainConfig(model="selftrain", variant=variant,
                        extra=f"pseudo=agree{agree_k}", seed=seed, **cfg_kwargs)
    try:
        if caches is None:
            import features as FEAT
            caches = []
            for b in members:
                c = FEAT.load_features(b, variant)
                if c is None:
                    raise FileNotFoundError(f"no cached features for {b}×{variant} — extract first")
                caches.append(c)
        ref_paths, feats_list, labels = E.align_by_paths(caches)
        labels = np.asarray(labels)
        samples = list(zip(ref_paths, [int(x) for x in labels]))
        if cog_idx is None:
            cog_idx = C.CLASSES.index(C.COG_CLASS)

        tr_idx, va_idx, te_idx, _ = C.split_by_collection(samples, cog_idx, seed)
        tr_bal = C.balance(tr_idx, samples, cog_idx, random.Random(seed))
        # Frame-split the 0422 target: pool frames may be pseudo-labeled; eval frames never are.
        te_samples = [samples[i] for i in te_idx]
        pool_local, eval_local = FS.frame_holdout_split(te_samples, cog_idx, eval_frac, seed)
        pool_idx = [te_idx[i] for i in pool_local]
        eval_idx = [te_idx[i] for i in eval_local]
        # Protocol guard: pseudo-pool and eval frames must be strictly disjoint.
        pool_frames = {C.frame_of(samples[i][0]) for i in pool_idx}
        eval_frames = {C.frame_of(samples[i][0]) for i in eval_idx}
        assert pool_frames.isdisjoint(eval_frames), "self-train pool leaked into eval frames"

        prior_val = float(prior) if prior is not None else float(np.mean(labels[tr_idx] == cog_idx))
        accepted: dict[int, int] = {}
        per_round = []
        for _r in range(max(1, rounds)):
            eff_labels = _eff(labels, accepted, cog_idx)
            train_idx = list(tr_bal) + list(accepted)
            # 1) score the held-out eval set with the current ensemble (the honest number)
            eval_probs = _member_heads_probs(feats_list, eff_labels, train_idx, va_idx,
                                             eval_idx, cog_idx, cfg)
            p_eval = E.average_probs(eval_probs)
            y_eval = [1 if int(labels[i]) == cog_idx else 0 for i in eval_idx]
            yp = [1 if p >= 0.5 else 0 for p in p_eval]
            per_round.append(C.balanced_accuracy(y_eval, yp) if len(set(y_eval)) == 2 else 0.0)
            # 2) score the pool, accept newly agreed-confident tiles (capped to the prior)
            pool_probs = _member_heads_probs(feats_list, eff_labels, train_idx, va_idx,
                                             pool_idx, cog_idx, cfg)
            pseudo = ensemble_agreement(pool_probs, hi=hi, lo=lo, agree_k=agree_k)
            pseudo = cap_positives_to_prior(pseudo, E.average_probs(pool_probs), prior_val,
                                            n_reference=len(pool_idx))
            for j, lab in enumerate(pseudo):
                if lab != ABSTAIN:
                    accepted[pool_idx[j]] = int(lab)

        # Final scoring through the shared tail on the held-out eval set (val from 0606).
        val_probs = _member_heads_probs(feats_list, labels, list(tr_bal), va_idx, va_idx,
                                        cog_idx, cfg)
        p_val = E.average_probs(val_probs)
        y_val = [1 if int(labels[i]) == cog_idx else 0 for i in va_idx]
        final_eval = _member_heads_probs(
            feats_list, _eff(labels, accepted, cog_idx), list(tr_bal) + list(accepted), va_idx,
            eval_idx, cog_idx, cfg)
        p_te = E.average_probs(final_eval)
        y_te = [1 if int(labels[i]) == cog_idx else 0 for i in eval_idx]
        row = T._make_ok_row(cfg, p_val=p_val, y_val=y_val, p_te=p_te, y_te=y_te,
                             val_bacc=per_round[-1], n_train=len(tr_bal) + len(accepted),
                             n_val=len(va_idx), n_test=len(eval_idx), n_trainable=None,
                             prior=prior_val)
        score_payload = (C.build_score_records([samples[i][0] for i in eval_idx], y_te, p_te)
                         if write_scores else None)
        row.f2_sweep = (row.f2_sweep or []) + [{"round_baccs": per_round, "n_pseudo": len(accepted)}]
    except Exception as e:  # noqa: BLE001 — record, never lose the cell (KTD8)
        status = "oom" if T._is_oom(e) else "failed"
        row = C.ResultRow(**cfg.identity(), status=status, error=f"{type(e).__name__}: {e}"[:500])
        score_payload = None

    C.write_result_atomic(row, results_dir)
    if score_payload is not None:
        C.write_scores_atomic(cfg.identity(), score_payload, results_dir)
    return row


def _eff(labels, accepted, cog_idx):
    """Apply accepted pseudo-labels (1=cogongrass / 0=not) as ImageFolder label indices."""
    eff = np.asarray(labels).copy()
    for i, lab in accepted.items():
        eff[i] = cog_idx if lab == 1 else (1 - cog_idx)
    return eff


def main():
    ap = argparse.ArgumentParser(description="Ensemble-agreement self-training (U11)")
    ap.add_argument("--members", default=",".join(E.DEFAULT_MEMBERS))
    ap.add_argument("--variant", default="reference")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--agree-k", type=int, default=3)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--lo", type=float, default=0.2)
    args = ap.parse_args()
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    print(f"== self-train {members}  rounds={args.rounds} agree_k={args.agree_k} ==", flush=True)
    row = run_selftrain(members, variant=args.variant, rounds=args.rounds, agree_k=args.agree_k,
                        hi=args.hi, lo=args.lo, write_scores=True)
    if row.status == "ok":
        print(f"\n0422(held-out) balanced accuracy: {row.balanced_accuracy:.3f}  "
              f"(recall cog {row.recall_cogongrass:.3f})")
    else:
        print(f"\n[{row.status}] {row.error}")


if __name__ == "__main__":
    main()
