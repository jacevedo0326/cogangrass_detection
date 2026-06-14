"""Evaluate the trained detector on the HELD-OUT test split (scored once).
Run after train_yolo.py:  python test_yolo.py
"""
from pathlib import Path
from ultralytics import YOLO


def main():
    # locate the most recent trained weights
    weights = sorted(Path("runs/detect").glob("cogongrass_det*/weights/best.pt"),
                     key=lambda p: p.stat().st_mtime)
    assert weights, "no trained weights found - run train_yolo.py first"
    best = weights[-1]
    print("evaluating:", best)

    model = YOLO(best)

    # 1. metrics on the test split (never seen during training or early stopping)
    m = model.val(data="drone_dataset/data.yaml", split="test",
                  name="cogongrass_test", plots=True)
    print("\n===== TEST =====")
    print(f"mAP50     {m.box.map50:.3f}")
    print(f"mAP50-95  {m.box.map:.3f}")
    print(f"precision {m.box.mp:.3f}")
    print(f"recall    {m.box.mr:.3f}")

    # 2. save annotated predictions on the test images for eyeballing
    ds = Path("drone_dataset")
    test_imgs = [str(ds / line.strip().lstrip("./"))
                 for line in (ds / "test.txt").read_text().splitlines() if line.strip()]
    model.predict(source=test_imgs, save=True, conf=0.25,
                  name="cogongrass_test_preds", exist_ok=True)
    print("\nannotated test predictions -> runs/detect/cogongrass_test_preds/")
    print("metrics & plots           -> runs/detect/cogongrass_test/")


if __name__ == "__main__":   # required on Windows (dataloader uses multiprocessing)
    main()
