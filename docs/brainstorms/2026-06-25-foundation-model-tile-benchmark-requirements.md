---
date: 2026-06-25
topic: foundation-model-tile-benchmark
---

# Foundation-Model Tile Benchmark (Zero-Shot + Light Fine-Tune)

## Summary

Benchmark strong off-the-shelf image models on the cogongrass tiles to see how
close out-of-the-box gets to the current trained classifier, then measure what a
light fine-tune adds. To measure a true ceiling, the models tested are the
strongest open-weight backbones the DGX Spark's memory can run — not 2060-era
sizes. Track 1 is zero-shot (local CLIP/SigLIP-class image-text models, plus a
local open VLM). Track 2 fine-tunes the best *trainable* local backbone, which is
not necessarily the zero-shot winner. Both share one field holdout — train on the
0606 flight, test on the held-out 0422 flight — so the floor, the lift, and the
existing 0.804 / 0.817 baselines are directly comparable. Everything lives in its
own isolated folder.

## Problem Frame

The current tile classifier is a trained CNN/ViT pipeline that needs labeled data
and per-experiment tuning to generalize to a new flight. Foundation image models
may already encode enough to classify these tiles with little or no training. The
open question is empirical: how far does zero-shot get, and how much does a cheap
fine-tune close the remaining gap — given that compute is now free on the DGX
Spark. The answer decides whether the trained pipeline is worth its maintenance
cost or whether an off-the-shelf model is good enough.

## Key Decisions

- **One shared field holdout, split by date.** Train on the 0606 collection, test
  on the entirely held-out 0422 collection. This reuses the repo's existing
  cross-collection protocol so results compare to the current 0.804 (ResNet18) and
  0.817 (Stage-1 DA) numbers, and so the zero-shot floor and the fine-tuned lift
  are measured on the same test set.

- **0422 is test-only.** No training and no validation on 0422; validation is
  carved from 0606. Tile-level splitting stays grouped by frame (the existing
  `frame_of` convention) so no frame's tiles span train and test.

- **Run the heavy work on the DGX Spark, and use its size budget.** The Spark's
  large unified memory removes the RTX 2060's 6 GB cap. That cap is the *only*
  reason the existing pipeline runs small models (the repo's DINOv2 scripts use the
  smallest `dinov2_vits14`), so the benchmark must actually exploit the headroom:
  test the strongest open-weight backbones that fit Spark memory, not a mid-size
  default. Otherwise a weak result reflects under-provisioning, not the off-the-
  shelf ceiling.

- **Two model-selection roles, not one.** The best *zero-shot* model (frozen
  text-image alignment) is often not the best *fine-tune* backbone — DINOv2-class
  self-supervised backbones have no text head and score poorly zero-shot yet are
  among the strongest trainable backbones. So the zero-shot track is restricted to
  image-text models (CLIP/SigLIP/EVA-CLIP class, which need a text encoder for
  prompts), while the fine-tune track picks the best trainable backbone for transfer
  strength — explicitly allowing a vision-only backbone like DINOv2.

- **The fine-tune stays light.** The goal is the free lift over zero-shot, not a
  hyperparameter-tuned SOTA. Two runs: fine-tune the zero-shot winner (so its lift
  is attributable to training, not a model swap), and fine-tune the best trainable
  backbone (so the achievable off-the-shelf number isn't capped by the zero-shot
  ranking).

- **Isolated folder.** All scripts, weights, logs, and outputs live in their own
  subdirectory and do not touch the baseline scripts or `tile_classifier.pt`.

## Requirements

**Track 1 — Zero-shot benchmark**

- R1. Run the strongest open-weight, image-text (CLIP/SigLIP/EVA-CLIP class) model
  that fits Spark memory, zero-shot over the full held-out 0422 tile set (~7,006
  tiles), producing a per-tile cogongrass score. "Largest open-weight model that
  fits the Spark" is an explicit selection criterion; the candidate shortlist (e.g.
  SigLIP2-so400m, EVA-CLIP-bigE, OpenCLIP ViT-bigG/H class) is fixed at planning,
  not left open. Run more than one when feasible and report the best.
- R2. Run a local open VLM on the Spark (e.g. Qwen2-VL, InternVL, Llama-3.2-Vision
  class) zero-shot over 0422, returning a per-tile prediction and a confidence. This
  keeps the VLM result on-goal (locally deployable) and comparable to the other
  local numbers; a remote API VLM, if run at all, is an out-of-band reference only
  (see Scope Boundaries).
- R3. Use a descriptive-prompt ensemble for the zero-shot text prompts (e.g.
  appearance of cogongrass: tall, feathery, pale seed plume), not the bare label
  "cogongrass" — prompt wording is the only knob in zero-shot.

**Track 2 — Light fine-tune**

- R4. Fine-tune the Track 1 zero-shot winner on the 0606 train field and evaluate on
  held-out 0422 (same split as Track 1); report the result as a lift over that
  model's own zero-shot score. This isolates what training adds to one model.
- R5. Also fine-tune the best *trainable* backbone — selected for transfer strength,
  not zero-shot rank, and allowed to be a vision-only backbone such as DINOv2 — and
  evaluate on 0422; report it as a lift over its own frozen / linear-probe baseline.
  This is the achievable off-the-shelf number, uncapped by the zero-shot ranking.

**Shared — evaluation and isolation**

- R6. Report balanced accuracy and per-class recall, never raw accuracy (the tile
  data is class-imbalanced). Include an F2-oriented threshold sweep so the decision
  threshold reflects that false negatives are costlier than false positives.
- R7. Present every model's number against the existing 0.804 / 0.817
  cross-collection baselines in one comparison.
- R8. Keep all code, weights, logs, and outputs for this experiment in a dedicated
  isolated folder, separate from the baseline pipeline.

## Success Criteria

- A single comparison table: each model (zero-shot image-text, zero-shot local VLM,
  fine-tuned zero-shot winner, fine-tuned best-trainable backbone) versus the 0.804 /
  0.817 baselines, on the same held-out 0422 set, by balanced accuracy + per-class
  recall + F2.
- A clear read on two questions: how close does zero-shot get to the trained
  baseline, and how much does the light fine-tune add.
- A pre-committed decision rule: off-the-shelf is declared "good enough to retire
  the trained pipeline" only when a tested model meets or exceeds the 0.804 / 0.817
  baseline by a stated F2 / balanced-accuracy margin. Matching-but-not-exceeding
  defaults to keeping the existing pipeline. A *light*-fine-tune result below
  baseline is inconclusive — it bounds free lift, not achievable performance — and
  does not on its own justify retiring the pipeline.
- No leakage: 0422 never seen in training or validation; tile split grouped by
  frame.

## Scope Boundaries

- Whole-image (non-tile) classification — out; tiles only.
- Hyperparameter search / tuned-SOTA fine-tuning — out; the fine-tune is
  deliberately light.
- Replacing or modifying the baseline tile classifier — out; this is a benchmark,
  not a deployment change.
- Site 15-vs-18 holdout — not used for this run; the date split is the chosen axis.
- Remote API VLM as a benchmarked entry — out of the headline comparison; the
  decision is about locally deployable models. An API VLM may be run on a small
  subsample as an optional out-of-band "cloud reference point," clearly labeled as
  non-comparable (different population, not locally deployable), never as part of the
  local ceiling.

## Dependencies / Assumptions

- **Assumption (load-bearing): 0422 and 0606 are physically different fields.** The
  date split is a true field holdout only under this assumption. If 0422 is the
  same parcel reflown on another date, field identity leaks across train/test
  despite the date split, and the "no contamination" goal is not fully met. Confirm
  before trusting the held-out number as a new-field result.
- DGX Spark access is set up with the data and a torch environment available on it.
  Because the Spark is ARM64 with a new CUDA arch and unified (shared) memory,
  before committing Track 2, run a smoke test confirming the selected backbone both
  loads and *trains* at the intended batch size, and record a fallback model/batch
  if it does not — so "eventually train on the Spark" is verified, not assumed.
- API access (key, budget, rate limits) exists for the chosen API VLM; the
  subsample-first scope keeps that cost small.
- Tiles are reused from the existing `tiles_dataset/` (folder name is the label);
  no re-tiling required for this experiment.

## Outstanding Questions

Deferred to planning:
- The exact shortlist of Spark-class models per track (the *criterion* — strongest
  open-weight that fits Spark memory — is fixed; the specific checkpoints and the
  tie-break among them are planning calls).
- Fine-tune depth (linear probe on frozen features vs light full fine-tune) — both
  are "light"; planning picks based on the zero-shot result.
- Whether to run an optional remote API VLM reference point at all, and if so the
  subsample size and stratification.
