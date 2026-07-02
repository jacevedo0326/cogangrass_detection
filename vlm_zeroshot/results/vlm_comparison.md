# VLM zero-shot vs trained baselines

**Coverage:** full — scored 7006 of 7006 held-out 0422 tiles (100%)
**Abstains:** 0 (metrics computed over 7006 scored tiles)
**Model:** `Qwen/Qwen2-VL-7B-Instruct`  ·  **Prompt:** `v1`  ·  **Split:** 0422

| Model | Balanced Acc | Recall (cogongrass) | Recall (not) | AUROC | AP |
|-------|:------------:|:-------------------:|:------------:|:-----:|:--:|
| **VLM zero-shot** | 0.572 | 0.199 | 0.946 | 0.535 | 0.301 |
| ResNet18 cross-collection | 0.804 | — | — | — | — |
| Stage-1 DA cross-collection | 0.817 | — | — | — | — |

_No 0606 slice supplied — AUROC / average precision are the primary (threshold-free) metrics; no fixed operating point is claimed._

## F2 threshold sweep (recall-weighted)

| thr | recall | prec | F1 | F2 | missed cogongrass (FN) |
|:---:|:------:|:----:|:--:|:--:|:----------------------:|
| 0.50 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.40 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.30 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.25 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.20 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.15 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.10 | 0.228 | 0.384 | 0.286 | 0.248 | 1539/1993 |
| 0.05 | 1.000 | 0.284 | 0.443 | 0.665 | 0/1993 |
