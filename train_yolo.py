"""Train a single-class cogongrass-patch detector on the drone imagery.
Transfer learning from COCO-pretrained YOLOv8s. Tuned for an RTX 2060 (6 GB).
Run:  python train_yolo.py
"""
from ultralytics import YOLO

if __name__ == "__main__":   # required on Windows (dataloader uses multiprocessing)
    model = YOLO("yolov8s.pt")          # COCO-pretrained -> transfer learning
    model.train(
        data="drone_dataset/data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,               # ~2.3 GB was used at batch 8, so 16 fits 6 GB; drop if OOM
        device=0,               # the RTX 2060
        patience=25,            # early stopping on val
        cache=True,             # 235 MB dataset fits in RAM -> faster epochs
        workers=4,
        name="cogongrass_det",  # -> runs/detect/cogongrass_det/
        plots=True,             # PR curve, confusion matrix, sample batches
    )
    # validation summary on the val split
    m = model.val()
    print(f"VAL  mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f} | P {m.box.mp:.3f} | R {m.box.mr:.3f}")
    print("best weights: runs/detect/cogongrass_det/weights/best.pt")
