"""Feasibility test: run SAM2 'segment everything' on a few drone frames.
Just to SEE what it produces and how slow it is on the RTX 2060 before
building the full propose->classify pipeline.
"""
import time
from pathlib import Path
from ultralytics import SAM


def main():
    imgs = sorted(Path("drone_dataset/images").glob("*.jpg"))[:3]
    model = SAM("sam2_t.pt")   # tiny SAM2 -> fits 6 GB; auto-downloads
    for img in imgs:
        t = time.time()
        # no prompts -> automatic full-image segmentation ("segment everything")
        r = model(str(img), save=True, project="runs/sam", name="explore",
                  exist_ok=True, verbose=False)
        n = len(r[0].masks) if r[0].masks is not None else 0
        print(f"{img.name}: {n} masks in {time.time()-t:.1f}s")
    print("overlays saved -> runs/sam/explore/")


if __name__ == "__main__":
    main()
