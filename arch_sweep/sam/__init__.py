"""SAM label-repair + segmentation/coverage eval (U8).

Isolated subdir with its own external dep (ultralytics SAM2/SAM-3) — a load/OOM failure
records a row and never blocks the rest of the program (KTD6, mirrors the U3/U11 fit-gate
discipline). The mask geometry (box∩mask repair, collapse-to-tile, coverage-MAE, pixel IoU)
is pure and CPU-tested; only mask *generation* needs the model.
"""
