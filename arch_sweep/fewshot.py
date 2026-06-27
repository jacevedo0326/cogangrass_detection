"""Few-shot target adaptation on the 0422 field (U10) — a SEPARATE eval setting (KTD4).

Measures how far a tiny **labeled 0422** budget goes, reported on its own and never blended
into the zero-target cross-collection ranking. Protocol:

1. Split the 0422 tiles **by frame** into a candidate pool and a held-out eval set — budget
   frames can never leak into eval (reuses the frame-grouped discipline from ``common``).
2. From the pool, **active learning** picks the budget by *uncertainty × diversity*: the most
   boundary-uncertain tiles, spread out by feature distance (greedy k-center) so the budget
   isn't redundant.
3. Fit a lightweight adapter on the labeled budget — ``prototype`` (nearest class mean),
   ``tip`` (training-free Tip-Adapter cache), or ``soup`` (averaged bootstrap logistic heads).
4. Evaluate on the held-out 0422 eval frames; write a row tagged ``eval_setting = few_shot``
   with the budget size recorded.

All operations are on cached features (U3), so this is pure-CPU testable with synthetic data.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ADAPTERS = ["prototype", "tip", "soup"]


# ---------------------------------------------------------------------------
# Frame-grouped budget/eval split of the 0422 set (no frame spans both)
# ---------------------------------------------------------------------------
def frame_holdout_split(samples, cog_idx, eval_frac=0.5, seed=C.DEFAULT_SEED):
    """Split sample indices into (pool_idx, eval_idx) by frame — disjoint frames.

    Frames are stratified on whether they contain any cogongrass so both the pool and the
    eval set see both classes. Budget tiles are later drawn only from ``pool_idx``.
    """
    frames: dict[str, dict] = {}
    for i, (p, lab) in enumerate(samples):
        f = C.frame_of(p)
        d = frames.setdefault(f, {"idx": [], "pos": False})
        d["idx"].append(i)
        if lab == cog_idx:
            d["pos"] = True
    rng = random.Random(seed)
    pos = [f for f, d in frames.items() if d["pos"]]
    neg = [f for f, d in frames.items() if not d["pos"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_ep, n_en = max(1, int(len(pos) * eval_frac)), max(1, int(len(neg) * eval_frac))
    eval_f = set(pos[:n_ep] + neg[:n_en])
    pool_idx = [i for f, d in frames.items() if f not in eval_f for i in d["idx"]]
    eval_idx = [i for f in eval_f for i in frames[f]["idx"]]
    return pool_idx, eval_idx


# ---------------------------------------------------------------------------
# Active learning: uncertainty × diversity (greedy)
# ---------------------------------------------------------------------------
def _l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def select_budget(features, base_p, budget, seed=C.DEFAULT_SEED):
    """Pick ``budget`` row indices by uncertainty × diversity (greedy k-center on cos-dist).

    ``base_p`` is P(cogongrass) from a base scorer (e.g. the 0606 model applied to 0422);
    uncertainty = 1 - |p - 0.5|·2. Each step picks the candidate maximizing
    ``uncertainty + min cosine-distance to the already-selected`` — boundary tiles, spread out.
    """
    n = len(features)
    budget = min(budget, n)
    if budget <= 0:
        return []
    Xn = _l2norm(np.asarray(features, dtype=np.float64))
    unc = 1.0 - np.abs(np.asarray(base_p, dtype=np.float64) - 0.5) * 2.0  # in [0,1], 1 = at boundary
    chosen = [int(np.argmax(unc))]                                         # seed: most uncertain
    # min cosine distance from each candidate to the chosen set (1 - max cos-sim)
    min_dist = 1.0 - (Xn @ Xn[chosen[0]])
    while len(chosen) < budget:
        score = unc + min_dist
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - (Xn @ Xn[nxt]))
    return chosen


# ---------------------------------------------------------------------------
# Adapters (all return P(cogongrass) per eval row)
# ---------------------------------------------------------------------------
def _prototype_p(feat_budget, lab_budget, feat_eval, cog_idx):
    Xb, Xe = _l2norm(np.asarray(feat_budget)), _l2norm(np.asarray(feat_eval))
    lab = np.asarray(lab_budget)
    protos = {}
    for c in (cog_idx, 1 - cog_idx):
        m = Xb[lab == c]
        protos[c] = m.mean(0) if len(m) else np.zeros(Xb.shape[1])
    d_cog = Xe @ _l2norm(protos[cog_idx][None])[0]
    d_not = Xe @ _l2norm(protos[1 - cog_idx][None])[0]
    z = np.stack([d_cog, d_not], 1) * 10.0       # temperature -> sharper softmax
    e = np.exp(z - z.max(1, keepdims=True))
    return e[:, 0] / e.sum(1)


def _tip_p(feat_budget, lab_budget, feat_eval, cog_idx, beta=5.5):
    Xb, Xe = _l2norm(np.asarray(feat_budget)), _l2norm(np.asarray(feat_eval))
    lab = np.asarray(lab_budget)
    onehot = np.zeros((len(lab), 2))
    onehot[np.arange(len(lab)), (lab != cog_idx).astype(int)] = 1.0   # col0 = cog, col1 = not
    affinity = np.exp(beta * (Xe @ Xb.T - 1.0))
    logits = affinity @ onehot
    p = logits[:, 0] / (logits.sum(1) + 1e-9)
    return p


def _soup_p(feat_budget, lab_budget, feat_eval, cog_idx, n_heads=8, seed=C.DEFAULT_SEED):
    from sklearn.linear_model import LogisticRegression
    Xb, Xe = np.asarray(feat_budget, dtype=np.float64), np.asarray(feat_eval, dtype=np.float64)
    y = (np.asarray(lab_budget) == cog_idx).astype(int)
    if len(set(y)) < 2:                          # can't train a 2-class head -> fall back
        return _prototype_p(feat_budget, lab_budget, feat_eval, cog_idx)
    rng = np.random.RandomState(seed)
    probs = []
    for _ in range(n_heads):
        idx = rng.randint(0, len(y), len(y))     # bootstrap resample
        if len(set(y[idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xb[idx], y[idx])
        probs.append(clf.predict_proba(Xe)[:, list(clf.classes_).index(1)])
    return np.mean(probs, axis=0) if probs else _prototype_p(feat_budget, lab_budget, feat_eval, cog_idx)


_ADAPTER_FN = {"prototype": _prototype_p, "tip": _tip_p, "soup": _soup_p}


def adapter_predict(adapter, feat_budget, lab_budget, feat_eval, cog_idx):
    if adapter not in _ADAPTER_FN:
        raise ValueError(f"unknown adapter {adapter!r}; choices: {ADAPTERS}")
    return _ADAPTER_FN[adapter](feat_budget, lab_budget, feat_eval, cog_idx)


# ---------------------------------------------------------------------------
# End-to-end few-shot run -> a few_shot ResultRow
# ---------------------------------------------------------------------------
def run_fewshot(model, features, labels, samples, cog_idx, *, adapter="prototype", budget=40,
                eval_frac=0.5, seed=C.DEFAULT_SEED, base_p=None, variant="reference",
                results_dir=C.RESULTS_DIR, write=True) -> C.ResultRow:
    """Fit ``adapter`` on an actively-selected 0422 budget, eval on held-out 0422 frames."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    pool_idx, eval_idx = frame_holdout_split(samples, cog_idx, eval_frac, seed)

    if base_p is None:   # cold-start uncertainty: prototype from a tiny random seed of the pool
        rng = random.Random(seed)
        seed_idx = rng.sample(pool_idx, min(len(pool_idx), max(4, budget // 4)))
        base_p_pool = _prototype_p(features[seed_idx], labels[seed_idx], features[pool_idx], cog_idx)
    else:
        base_p_pool = np.asarray(base_p)[pool_idx]

    sel_local = select_budget(features[pool_idx], base_p_pool, budget, seed)
    budget_idx = [pool_idx[i] for i in sel_local]

    p_eval = adapter_predict(adapter, features[budget_idx], labels[budget_idx],
                             features[eval_idx], cog_idx)
    # operating threshold on the labeled budget only (never the eval frames)
    p_budget = adapter_predict(adapter, features[budget_idx], labels[budget_idx],
                               features[budget_idx], cog_idx)
    y_budget = [1 if int(labels[i]) == cog_idx else 0 for i in budget_idx]
    threshold = C.pick_threshold_on(y_budget, p_budget) if len(set(y_budget)) == 2 else 0.5

    y_eval = [1 if int(labels[i]) == cog_idx else 0 for i in eval_idx]
    y_pred = [1 if p >= 0.5 else 0 for p in p_eval]
    rec = C.per_class_recall(y_eval, y_pred)
    both = len(set(y_eval)) == 2
    row = C.ResultRow(
        model=model, variant=variant, tuning_mode="frozen", head="adapter",
        adaptation=f"fewshot:{adapter}", eval_setting=C.EVAL_FEWSHOT, seed=seed,
        extra=f"budget={budget}", status="ok",
        balanced_accuracy=C.balanced_accuracy(y_eval, y_pred),
        recall_cogongrass=rec[C.COG_CLASS], recall_not_cogongrass=rec["not_cogongrass"],
        auroc=C.auroc(y_eval, p_eval) if both else None,
        average_precision=C.average_precision(y_eval, p_eval) if both else None,
        threshold=threshold, f2_sweep=C.f2_sweep(y_eval, p_eval),
        n_train=len(budget_idx), n_test=len(eval_idx), n_cog_test=int(sum(y_eval)),
        budget=budget)
    if write:
        C.write_result_atomic(row, results_dir)
    return row


# ---------------------------------------------------------------------------
# Runnable sweep entry (U7): run every adapter × budget on a backbone's cached features
# ---------------------------------------------------------------------------
DEFAULT_BUDGETS = [8, 16, 40, 80]


def run_fewshot_sweep(model, features, labels, samples, cog_idx, *, adapters=ADAPTERS,
                      budgets=DEFAULT_BUDGETS, seed=C.DEFAULT_SEED, variant="reference",
                      results_dir=C.RESULTS_DIR, write=True) -> list[C.ResultRow]:
    """Run every ``adapter × budget`` cell on one backbone's cached 0422 features.

    Each cell is its own ``few_shot`` ResultRow (distinct ``adaptation``/``extra`` -> distinct
    ``job_id``), so they merge into the report's separate few-shot table (KTD4) and never touch
    the cross-collection ranking. Returns the rows.
    """
    rows = []
    for adapter in adapters:
        for budget in budgets:
            rows.append(run_fewshot(model, features, labels, samples, cog_idx, adapter=adapter,
                                    budget=budget, seed=seed, variant=variant,
                                    results_dir=results_dir, write=write))
    return rows


def main():
    import argparse

    import features as FEAT

    ap = argparse.ArgumentParser(description="Few-shot 0422 adaptation sweep (U7; few_shot table)")
    ap.add_argument("--model", required=True, help="backbone with cached features (features.py)")
    ap.add_argument("--variant", default="reference")
    ap.add_argument("--adapters", default=",".join(ADAPTERS))
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    args = ap.parse_args()

    cache = FEAT.load_features(args.model, args.variant)
    if cache is None:
        raise SystemExit(f"no cached features for {args.model}×{args.variant} — run features.py first")
    samples = list(zip(cache["paths"], [int(x) for x in cache["labels"]]))
    cog_idx = C.CLASSES.index(C.COG_CLASS)
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    print(f"== few-shot sweep {args.model}  adapters={adapters} budgets={budgets} ==", flush=True)
    rows = run_fewshot_sweep(args.model, cache["features"], cache["labels"], samples, cog_idx,
                             adapters=adapters, budgets=budgets, seed=args.seed,
                             variant=args.variant)
    for r in rows:
        print(f"  {r.adaptation:18s} budget={r.budget:<3} -> bacc {r.balanced_accuracy:.3f} "
              f"(recall cog {r.recall_cogongrass:.3f})")


if __name__ == "__main__":
    main()
