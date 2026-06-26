"""PlantCLEF cell — DINOv2 ViT pretrained on plant imagery (in-domain prior) (U5).

Thin entry point over common + trainer (KTD1). Run:
    python arch_sweep/models/train_plantclef.py [--variant ...] [--head ...] [--seed ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: F401 — single-source split/metrics/writer; this script defines none
import trainer as T  # noqa: E402


def main():
    T.run_cli(model="plantclef")


if __name__ == "__main__":
    main()
