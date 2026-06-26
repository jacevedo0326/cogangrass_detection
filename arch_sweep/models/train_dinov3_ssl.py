"""Continued-SSL-adapted DINOv3 cell (U9) — scheduled like any other model.

Loads a DINOv3 backbone with continued-SSL weights (produced by continued_ssl on unlabeled
tiles) and runs the shared trainer, so the domain-adapted backbone appears as a normal job
``model=dinov3_ssl`` and its cross-collection number is comparable to its un-adapted base.

Run (after producing the SSL checkpoint):
    python arch_sweep/models/train_dinov3_ssl.py --size l --ckpt results/ssl/dinov3_l_ssl.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backbones as B  # noqa: E402, F401 — single-source registry; this script defines no split/metric
import common as C  # noqa: E402, F401
import continued_ssl as CSSL  # noqa: E402
import trainer as T  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "results" / "ssl" / "dinov3_ssl.pt"


def register_ssl_backbone(size: str, ckpt: str):
    """Register a 'dinov3_ssl' backbone whose loader applies the SSL checkpoint to the base."""
    base = B.dinov3_name(size)
    dim = B.get(base).feature_dim
    B.register(B.BackboneSpec(
        "dinov3_ssl", dim, build=lambda **_: CSSL.load_ssl_backbone(base, ckpt),
        source="torch.hub", ref=f"ssl:{base}"))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--size", default="l", choices=list(B.DINOV3_SIZES))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="continued-SSL checkpoint path")
    known, rest = ap.parse_known_args()
    if not Path(known.ckpt).exists():
        raise SystemExit(f"SSL checkpoint {known.ckpt} not found — run continued_ssl first "
                         f"(see arch_sweep/continued_ssl.py)")
    register_ssl_backbone(known.size, known.ckpt)
    T.run_cli(model="dinov3_ssl", argv=rest)


if __name__ == "__main__":
    main()
