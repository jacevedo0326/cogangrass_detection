---
date: 2026-06-26
type: feat
status: requirements
origin: docs/ideation/2026-06-26-002-whats-next-cogongrass-cross-field.html
---

# Stage-2 program: push cross-field cogongrass detection past the baseline — on a trustworthy ruler

## Summary

The `arch_sweep` Stage-1 work answered "which backbone generalizes best frozen" and showed test-time
adaptation already ties the original deployable model (aimv2 + EATA = **0.839**, vs baselines 0.804 / 0.817 and
the original ~0.84). This program covers **everything worth trying next** — the seven survivors from the
2026-06-26 ideation — organized into one sequenced effort with a single governing principle:

> **Fix what we measure against before optimizing against it.** AUROC ~0.90 with argmax recall ~0.6 is the
> fingerprint of label noise, not a weak model. Until the 0422 ground truth is cleaned and every score carries a
> confidence interval, "best architecture" is undecidable.

The program is therefore staged: a **foundation** that makes results trustworthy, a tier of **cheap cached-feature
gains**, two **new generalization mechanisms**, and a **compounded-adaptation** tier. Each item is comparable to
the existing 0.804 / 0.817 / 0.84 tile numbers (or states the new metric it needs).

## Problem Frame

The real task is generalizing to a *new flight it has never seen* (train collection 0606 → test entirely
held-out field 0422, 7,006 tiles, ~28% positive). False negatives (missed cogongrass) are the costly error. Two
structural weaknesses limit progress: (1) the decision boundary is learned once on 0606 and frozen, so it can't
adapt to a new field's color/exposure/phenology; (2) labels are coarse — a tile is positive only if ≥30% inside a
YOLO box, and a frame with no box file is all-negative — so missed annotations become false negatives in the
answer key. The deliverable is a coverage heatmap, not tile labels.

## Goals

- **G1.** Establish a *trustworthy* evaluation: cleaned 0422 ground truth + frame-level bootstrap confidence
  intervals, so architecture comparisons are decidable.
- **G2.** Beat the **0.817** baseline — and ideally match/exceed the original **~0.84** — on the cleaned ruler,
  **without collapsing cogongrass recall** (the recall-guard already in the report).
- **G3.** Reduce false negatives at the deployed operating point (report recall / F2 at the chosen threshold, not
  just argmax).
- **G4.** Determine whether a *different problem formulation* (promptable detection / segmentation) generalizes
  across fields better than tile classification — the only path that re-anchors per field.
- **G5.** Keep every result comparable and crash-safe within the existing `arch_sweep` harness (shared split,
  metrics, job-id, report).

## Non-Goals

- Real-time / on-drone / edge constraints; multi-species beyond cogongrass.
- Re-collecting data beyond a small active-learning / relabeling budget.
- Shipping / deployment packaging — this program measures and selects; it does not productionize.
- Approaches already falsified or blocked (see Scope Boundaries).

## Work Items (sequenced; IDs for traceability)

Ordering reflects leverage and dependency, not human-time. Each item states **intent** and **success/acceptance**;
implementation is for `/ce-plan`.

### Foundation — make the ruler trustworthy (do first)

- **R1 — Clean the 0422 test labels.** Surface tiles the models confidently call cogongrass but ground truth marks
  negative (the existing `suspect_negatives.py` / `fp_audit.py`, repointed from the legacy 160/1280 config to
  512/4096), human-review the flagged frames, and freeze a `tiles_dataset_0422clean/` variant.
  *Success:* a cleaned eval variant enumerable by `common.py`; the count and nature of corrected tiles recorded;
  every backbone re-scored against it.
- **R2 — Evaluation hygiene.** Add **frame-level** bootstrap confidence intervals on balanced accuracy / F2 (resample
  frames, not tiles, since tiles are correlated within a frame), a per-frame metric breakdown, and dual reporting
  against raw vs cleaned ground truth.
  *Success:* `report.py` shows CI columns and a per-frame view; a win is only claimed when CIs separate it from the
  baseline.

### Cheap cached-feature gains (hours each; run off the existing feature cache)

- **R3 — Decorrelated backbone ensemble.** Average per-tile cogongrass probabilities across the top backbones
  (siglip2, aimv2, cradio, dinov3_sat), in both frozen and EATA-adapted tiers.
  *Success:* an ensemble `ResultRow` (distinct job-id) scored identically; reported against the best single member.
- **R4 — Apply the operating threshold we already fit (calibration).** The headline is computed at argmax 0.5 while
  the F2-optimal 0606 threshold is recorded but unused. Apply it; add temperature scaling (0606), AdaBN score
  alignment (0422), and label-free prior-matching; report recall / F2 at the chosen point. Label-dependent conformal
  FN-rate control routes through the existing `few_shot` track, reported separately.
  *Success:* cogongrass recall at the deployed threshold rises materially toward the AdaBN/RoTTA regime; the headline
  stays an honest cross-collection number.
- **R5 — Cheap domain-alignment stack.** CORAL feature re-coloring (0606→0422 second-order stats, label-free),
  early-fusion feature concatenation, and multi-scale + flip test-time augmentation. All stack with R3/R4.
  *Success:* each emits a comparable row; the best stack is identified with CIs.

### New generalization mechanisms (bigger bets)

- **R6 — Deploy-time promptable detection with visual exemplars.** Evaluate an open-vocab / visual-prompt detector
  (e.g. T-Rex2, with negative exemplars for look-alike grasses) that re-anchors per field from a few example boxes,
  rasterized onto the 0422 tile grid.
  *Success:* per-tile predictions on the same 0422 grid → identical balanced-accuracy / recall / AUROC; verdict on
  whether per-field re-anchoring beats frozen+TTA, with the diffuse-texture/oblique-view risk assessed.
- **R7 — SAM label-repair + segmentation/coverage reframe.** (a) Offline: intersect YOLO boxes with SAM masks to
  replace the crude ≥30%-of-box rule, lifting every track's ceiling. (b) Deploy: segment → classify region → pixel
  coverage (the true deliverable). Builds on existing `sam_explore.py`; the GB10 unlocks SAM2-large / SAM 3.
  *Success:* (a) a cleaner-label variant and the lift it gives the 0.817 baseline when retrained; (b) tiles collapsed
  for the standard metric, plus pixel IoU / coverage-MAE as the richer native score.

### Compounded adaptation

- **R8 — Robust, deployment-real, compounded TTA.** Add SAR and CoTTA to `tta.py` and an **episodic per-frame** mode
  (matching `heatmap_infer.py`, which adapts one frame's tiles at a time); enable TTA on the fine-tuned (LoRA/full)
  path (stack fine-tune + EATA); and ensemble-agreement pseudo-label self-training on 0422 (add tiles where ≥3/4
  backbones agree at high confidence, target data only via its own predictions, test frames strictly held out).
  *Success:* per-frame heatmap stability demonstrated (no one-class collapse); a clear verdict on whether
  fine-tune+TTA and self-training compound beyond R3–R5.

## Success Criteria (program-level)

- The headline ranking is read on the **cleaned 0422 set with bootstrap CIs** (R1+R2). No win is claimed inside the
  noise band.
- At least one configuration **beats 0.817 with non-overlapping CIs** and does not drop cogongrass recall below the
  report's recall guard (G2). Stretch: match/exceed ~0.84.
- The deployed operating point reports **recall / F2**, with a measurable false-negative reduction vs argmax (G3).
- A documented verdict on R6/R7 (does a reframe generalize better than tile-classification+TTA?), even if negative.
- Few-shot results stay in their own table, never blended into the cross-collection ranking (KTD4 from the plan).

## Scope Boundaries

### Deferred for later (out of this program, not wrong)
- **Bi-temporal change detection** — blocked by data: 0606 and 0422 are two *different fields*, not repeat flights.
  Becomes viable only with a deliberate same-field revisit; a data-collection roadmap item.
- A weed-specific foundation backbone (e.g. WeedNet) — at most one extra tournament slot, not a program.

### Rejected (explicit "no", with reason)
- **VLM as the classifier** — already falsified (Qwen2-VL-7B ≈ chance, 0.572 / 0.535). VLMs are retained only as
  offline label-cleaners under R1, never as detectors.
- **One-class / novelty detection** — cogongrass is ~28% of held-out tiles, not rare; the outlier assumption breaks.
  Allowed only as a re-ranking/abstention layer, not a standalone detector.
- **Naïve single-model self-training** — confirmation bias; only the ensemble-agreement-gated form (R8) survives.

## Dependencies & Assumptions

- **Single-tenant GB10 (~120 GB)**; stop other GPU tenants before heavy runs. Features are cached per
  (backbone, variant), so R3–R5 are CPU-cheap.
- **Label-audit tooling** (`suspect_negatives.py`, `fp_audit.py`, `neighbor_analysis.py`) currently hardcodes the
  legacy 160px/1280-prep config and must be repointed to 512/4096 before R1 (assumption, verified: these scripts
  predate the full-res pipeline).
- **R1 needs a bounded human relabeling pass** on the flagged 0422 frames (carried from the origin assumption of a
  small labeling budget).
- **0606 and 0422 are genuinely different physical fields** (carried from origin); R8 self-training must keep the
  0422 test frames strictly out of any training/pseudo-label pool to preserve the protocol.
- Current state: Stage-1 frozen tournament + TTA arm complete (aimv2+EATA 0.839); aimv2 LoRA/full fine-tune running;
  continued-SSL and few-shot harnesses built but not fully run.

## Outstanding Questions (resolve in planning / at run time)

- Relabeling budget for R1 — how many flagged frames to review, and the confidence cutoff for `suspect_negatives.py`.
- The "material win" margin over 0.817 once Stage-1-with-CIs spread is visible (e.g. CIs must clear by ≥ X).
- Whether R7's segmentation reframe reports a *new* primary metric (coverage-MAE) or stays bridged to tile accuracy.
- R6 detector choice and prompt strategy (the "Does Your VFM Speak Plant?" finding: optimal prompts diverge from
  species names) — settle empirically on a few exemplars.
- Sequencing within the cheap tier: confirm R1→R2 land before R3–R5 are read, so the ruler is clean first.
