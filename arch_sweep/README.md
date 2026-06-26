# arch_sweep — traditional-ML architecture sweep

Measures *which architecture best survives cross-field domain shift* for cogongrass tile
detection, on the honest **0606 → 0422 cross-collection protocol**, against the
**0.804** (ResNet18) and **0.817** (Stage-1 DA) balanced-accuracy baselines.

It is a set of **separate per-model training scripts** over a **shared leakage-safe
library**, plus one orchestrator that trains + tests every model and **saves each result
the instant it lands** (crash-safe, resumable). See
`docs/plans/2026-06-26-001-feat-arch-sweep-traditional-ml-plan.md` for the full design.

## The protocol (do not drift from it)

- **Train on 0606, test on the entirely held-out 0422 flight** — a proxy for "a new
  field it has never seen." This is *not* a random tile split.
- **Split by FRAME, never by tile.** `common.split_by_collection` groups tiles by their
  source frame so no frame spans train/val/test (frame leakage inflates the score).
- **The operating threshold is fit on 0606 and applied to 0422** — never selected on
  0422. Any cell that peeks at 0422 for tuning is invalid.
- **Report balanced accuracy + per-class recall**, never raw accuracy (the data is
  class-imbalanced). False negatives (missed cogongrass) are the costly error → F2 sweep.
- **Two eval settings, never blended (KTD4):** `cross_collection` (headline, no target
  labels) and `few_shot` (spends a small labeled 0422 budget) are reported in separate
  tables.

## Layout

```
common.py         # split + metrics + ResultRow + deterministic job_id + atomic writer + seed policy (U1)
data_variants.py  # materialize tile-size × CLAHE × PREP_MAX × ExG variants + manifest (U2)
backbones.py      # backbone registry: name -> (loader, feature_dim, processor) (U3)
features.py       # frozen-feature extraction + per-(backbone,variant) disk cache (U3)
heads.py          # head registry: linear, MLP+BatchNorm (U4)
trainer.py        # shared train+eval loop every model script calls (U4)
models/           # one thin entry-point script per model (U5)
results/          # per-job <job_id>.jsonl + cached features + checkpoints + logs (git-ignored)
tests/            # unit tests (pure where possible; ML-stack pieces gated behind smoke runs)
```

## Single source of truth (why separate scripts are safe)

Every model script imports the split, metrics, job identity, and result writer from
`common.py` and the train/eval loop from `trainer.py`. A script holds **only**
model-specific loading/config — it defines no split, metric, or result-writing logic of
its own (enforced by `tests/test_model_scripts.py`). So leakage prevention and
comparability are single-sourced and cannot drift between scripts.

## Crash-safety & resume (KTD8)

Each job writes its **own** `results/<job_id>.jsonl`, flushed + `fsync`'d and published
via an atomic temp-file → `os.replace`. `job_id` is a deterministic hash of the cell's
identity (model, variant, tuning mode, head, adaptation, eval setting, seed), so a
re-run **skips any cell whose result file already exists** and a kill loses at most the
single in-flight job. The report globs and merges per-job files — no shared-file write
race.

## Running on the Spark (GB10)

The Spark `.venv` (cu130 aarch64 torch 2.12.1, transformers 5.x) and `tiles_dataset/`
must be present, with `HF_HOME=/home/josh/hf_cache` for model downloads (the root-owned
default cache fails). Backbones load **explicitly onto `cuda`** — never
`device_map="auto"`, which CPU-offloads on the GB10 (~13× slowdown). **Stop other GPU
tenants (the root vLLM servers) before a full sweep** so the ~120 GB is actually free.

Backbone dependencies (beyond torch/torchvision/transformers), installed into the `.venv`
and confirmed by the smoke fit gate: **`timm`** (PlantCLEF + AIMv2), **`torchmetrics`**
(DINOv3 hub code), **`einops`** + **`open_clip_torch`** (C-RADIOv3). AIMv2 is loaded via
`timm` (`aimv2_large_patch14_224.apple_pt`) because the transformers remote-code path is
incompatible with transformers 5.x; C-RADIOv3 loads via the `NVlabs/RADIO` hub for the
same reason.

**DINOv3 is license-gated.** Accept the license on each `facebook/dinov3-*` repo, then
`hf auth login` once (token saved under `$HF_HOME`); the loader then downloads the gated
weights. Alternatively set `DINOV3_<SIZE>_WEIGHTS=/path/to/x.pth` to load local weights
offline. All 4 sizes (s/b/l/sat) are confirmed working with a token.

Run the whole sweep once dependencies + the DINOv3 token are in place:

```bash
bash arch_sweep/run_working.sh              # all 10 backbones, sequential (safest)
bash arch_sweep/run_working_concurrent.sh   # all 10 at once (faster start; VRAM-contended)
```

Fit-gate every backbone before the full sweep (loads each model + trains/tests on a tiny
subset, writes under `results/smoke/`):

```bash
export HF_HOME=/home/josh/hf_cache    # needs `hf auth login` first for DINOv3
for s in resnet18 dinov2 plantclef siglip2 aimv2 cradio; do
  .venv/bin/python arch_sweep/models/train_$s.py --smoke
done
for sz in s b l sat; do .venv/bin/python arch_sweep/models/train_dinov3.py --size $sz --smoke; done
```

All 10 backbones pass this gate (resnet18, dinov2, dinov3 s/b/l/sat, plantclef, siglip2,
aimv2, cradio).

Quick checks:

```bash
# U1: confirm the split matches the baselines (prints 0422 = 7006 tiles / 262 frames)
.venv/bin/python arch_sweep/common.py --self-check tiles_dataset

# run the unit tests
.venv/bin/python -m pytest arch_sweep/tests -q

# U3 fit gate: extract one backbone's features over a few real tiles (cheap, run FIRST)
.venv/bin/python arch_sweep/features.py --backbone resnet18 --limit 16

# U5: train + test a single model end-to-end (writes one result row)
.venv/bin/python arch_sweep/models/train_resnet18.py
```
