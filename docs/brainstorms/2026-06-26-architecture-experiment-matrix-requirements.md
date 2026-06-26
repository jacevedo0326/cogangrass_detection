---
date: 2026-06-26
type: feat
status: requirements
origin: docs/ideation/2026-06-26-best-vision-model-cogongrass-domain-shift.html
---

# Comprehensive architecture experiment matrix for cogongrass tile detection

## Summary

Design a **single, staged experiment matrix** that systematically tests every model and
preprocessing architecture worth trying for cogongrass tile classification — data inputs,
CLAHE on/off, frozen vs fine-tuned backbones, local VLM fine-tuning, external frontier-VLM
API benchmarks, domain-adaptation methods, and structural task reframes — all judged on the
**same honest cross-collection protocol** (train on 0606, test on the entirely held-out 0422
field). The goal is not to pick a model by intuition but to **measure** which configuration
best survives the cross-field domain shift that breaks the current ~0.80–0.82 baselines.

This is the measurement plan. It inherits the model-ranking from the ideation
(`docs/ideation/2026-06-26-best-vision-model-cogongrass-domain-shift.html`) and the negative
result from `vlm_zeroshot/` (zero-shot Qwen2-VL-7B ≈ chance, AUROC 0.535), and turns both into
a falsifiable sweep.

## Problem Frame

The maintained tile classifier generalizes poorly to a **new flight it has never seen**: it
scores ~0.80–0.82 balanced accuracy on the held-out 0422 field versus much higher in-collection.
Many individual ideas (DINOv3, continued self-supervised pretraining, robust TTA, exemplar
detectors, VLM fine-tuning) plausibly help, but they have **never been measured head-to-head on
one protocol**. Without that, model choice is guesswork. The deliverable closes that gap: an
apples-to-apples ablation across the full design space, with the 0606→0422 number as the single
arbiter.

## Goal & Success Criteria

- **Primary metric:** balanced accuracy on the held-out 0422 field, with **AUROC** and
  **cogongrass recall** as co-primary (false negatives — missed cogongrass — are the costly
  error). Report per-class recall and an F2 threshold sweep for every configuration.
- **Win bar:** a configuration "wins" if it **materially beats 0.817** (the current Stage-1 DA
  cross-collection balanced accuracy) on 0422, without collapsing cogongrass recall.
- **Protocol integrity:** every cell runs the identical frame-grouped 0606→0422 split with a
  leakage check; any cell that peeks at 0422 for tuning is invalid.
- **Decision output:** a ranked results table naming the best data × backbone × adaptation
  configuration and the external-VLM ceiling, sufficient to choose a production model in a
  follow-up.

## The Experiment Space (requirements)

Each axis group is a requirement to cover. Levels are the cells to sweep; staging (below) keeps
the matrix runnable rather than full-factorial.

- **R1 — Data input & preprocessing.** Tile size (224 / 512 / multi-scale); source resolution
  (`PREP_MAX` 1280 / 4096); **CLAHE on vs off**; tile-label rule (≥30% area threshold; ExG
  green-filter on/off). Reuses `prep_images.py`, `boxes_to_tiles.py`, `precompute_clahe.py`.
- **R2 — Backbone.** ResNet18 (baseline) · DINOv2 (existing) · **DINOv3** (S/B/L, + SAT variant)
  · **PlantCLEF-DINOv2** (plant-domain) · SigLIP2 · AIMv2 · C-RADIOv3. DINOv3 / PlantCLEF /
  SigLIP2 / AIMv2 / C-RADIOv3 are net-new to the repo.
- **R3 — Tuning mode (the "frozen vs not" axis).** Frozen + linear head · frozen + MLP/BatchNorm
  head (preserves the AdaBN inference trick) · **LoRA / parameter-efficient** · full fine-tune.
  Head regularization: dropout / label-smoothing / weight-decay. LoRA/PEFT is net-new.
- **R4 — Local VLM track ("hyper-tune a local LLM").** Qwen2-VL-7B (zero-shot done) · Qwen2.5-VL
  · InternVL2.5/3 · Llama-3.2-Vision · Molmo, each in three modes: **zero-shot · few-shot
  in-context (labeled example tiles in the prompt) · LoRA fine-tune**. Extends `vlm_zeroshot/`.
- **R5 — External frontier-VLM ceiling.** Gemini 2.x · GPT-4o/o-series · Claude, zero-shot and
  few-shot, as the "how good can any model get" ceiling. **External data egress approved** (cost
  no object, privacy not a blocker); net-new API-calling code.
- **R6 — Domain-adaptation layer (wraps any backbone).** Test-time adaptation: AdaBN · TENT
  (existing) · **EATA** · **RoTTA** (TENT is fragile on imbalanced single-frame streams).
  **Continued self-supervised pretraining on unlabeled drone frames** (ExPLoRA-style). Few-shot
  adapters: Tip-Adapter · prototype · Soup-Adapter. **Active-learning** tiny target-field label
  budget. Augmentation: domain-randomization (existing) · **Fourier amplitude-swap** · MixStyle.
- **R7 — Structural task reframe.** Tile classification (current) · **SAM 3 / T-Rex2 visual-
  exemplar → frozen-feature head** (factor "where is grass" from "which grass") · YOLO box
  detector (existing track). Mask-area as an honest coverage target.

## Staged Execution (scope structure)

Full factorial is infeasible (thousands of cells). The matrix runs as **staged elimination** —
vary one axis off a strong baseline, carry forward only winners:

1. **Stage 0 — harness & data.** Lock the leakage-checked 0606→0422 eval; generate the R1 data
   variants once (tile sizes × CLAHE on/off).
2. **Stage 1 — backbone tournament.** R2 at a fixed data config, frozen + linear head → keep top
   2–3 by 0422 balanced accuracy.
3. **Stage 2 — adaptation ablations on winners.** R3 freeze-mode × R1 CLAHE × head × R6
   augmentation/TTA, one axis at a time.
4. **Stage 3 — VLM track (parallel).** R4 local VLMs + R5 external-API ceiling.
5. **Stage 4 — structural reframes.** R7 (SAM3 / exemplar / YOLO).
6. **Stage 5 — best-of synthesis.** Combine the winning data × backbone × adaptation; report the
   final held-out number and the full ranked table.

## Evaluation Protocol (fixed across every cell)

- Train on the **2026-06-06** collection, test on the entirely held-out **2026-04-22** field,
  grouped by frame so no frame spans splits (per `train_tiles_collection.py` / `leakage_check.py`).
- Report **balanced accuracy + per-class recall + AUROC/AP + F2 sweep** — never raw accuracy
  (tiles are class-imbalanced).
- Apply AdaBN at inference for BatchNorm-bearing heads; record the operating threshold honestly
  (selected on 0606, applied to 0422).
- Each new experiment writes to its own weights/log; baseline scripts stay untouched.

## Scope Boundaries

**In scope:** measuring all R1–R7 axes on the cross-collection protocol; a ranked decision table;
net-new code for DINOv3/PlantCLEF backbones, LoRA tuning, external-API VLM calls, robust TTA,
continued-SSL, and exemplar/segmentation reframes.

**Out of scope (non-goals):**
- Choosing/shipping a single production model now — this plan *finds* it; deployment is a follow-up.
- Real-time / edge / on-drone compute constraints — Spark-class compute is assumed throughout.
- Re-collecting or re-labeling the core dataset beyond the small active-learning budget (R6).
- Multi-class / species-beyond-cogongrass work.

## Dependencies & Assumptions

- **Assumes** a small human-labeling budget on the target field is available (R6 active learning);
  "cost no object" is read to include modest labeling.
- **Assumes** 0422 and 0606 are genuinely different physical fields (the "new field" framing
  depends on it — carried from the origin plan; confirm before reading any number as new-field).
- **Depends on** the DGX Spark runtime already stood up (cu130 aarch64 torch, transformers 5.x)
  and the existing `tiles_dataset/` on the Spark.
- **External APIs:** egress approved; licensing/ToS of each provider is a note, not a blocker.
- **Backbone licensing:** DINOv3 ships under a bespoke (non-Apache) license — flag for legal
  before any production use, though it does not block benchmarking.

## Open Questions (for ce-plan / run time)

- Exact backbone checkpoints and sizes that fit alongside other Spark tenants (the GB10 is shared;
  memory is constrained).
- How many active-learning labels per field is the realistic budget (sets the R6 design).
- Whether the structural reframes (R7) are run in full or gated behind Stage 1–3 results.
- Per-axis "material improvement" delta that counts as a win (e.g. ≥ +2 balanced-accuracy points).

## Handoff

Ready for `/ce-plan` to turn the staged structure into an executable plan with concrete units,
scripts, and run order. The plan should preserve the cross-collection protocol and the
isolated-experiment convention (new `*.pt` / `*.log` per cell; baseline scripts untouched).
