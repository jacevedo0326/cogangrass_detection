"""Visual-exemplar promptable detection (U9).

Isolated subdir with its own external dep (T-Rex2 / T-Rex-Omni or equivalent open-vocab,
visual-prompt detector) — the one mechanism that *re-anchors per field* from a few exemplars
rather than freezing a 0606 boundary. A load/OOM failure records a row and never blocks the
program (KTD6). The rasterize→tile geometry and the F2 sweep are pure and CPU-tested; only
the detector itself needs the model.

Because prompting uses a few **0422** exemplar boxes (labels), an exemplar-prompted cell is
tagged ``few_shot`` (KTD5) and kept in the report's separate table.
"""
