"""DINOv3 cell — newer SSL ViT; one script covers all sizes via --size {s,b,l,sat} (U5).

Thin entry point over common + trainer (KTD1); the size resolves to a registered backbone.
Run:
    python arch_sweep/models/train_dinov3.py --size l [--variant ...] [--head ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: F401 — single-source split/metrics/writer; this script defines none
import trainer as T  # noqa: E402


def main():
    T.run_cli(model="dinov3", add_size=True)


if __name__ == "__main__":
    main()
