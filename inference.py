"""Offline inference for the Code Alchemists YOLO26m submission."""

import os
from pathlib import Path


WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
CLASS_CODES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
IOU = 0.7
MAX_DETECTIONS = 100


def load_model(weights=WEIGHTS):
    """Load the trained checkpoint already stored in the repository."""
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint is missing: {weights}")

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    if model.names != CLASS_CODES:
        raise ValueError(f"Unexpected damage classes in {weights}: {model.names}")
    return model


def predict_paths(model, image_paths, device=0):
    """Return one list per image, each containing class, score and xyxy pixel boxes."""
    paths = [Path(image) for image in image_paths]
    if not paths:
        return []
    results = model.predict(
        source=[str(path) for path in paths],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU,
        max_det=MAX_DETECTIONS,
        device=device,
        verbose=False,
    )
    predictions = []
    for path, result in zip(paths, results, strict=True):
        if Path(result.path).name != path.name:
            raise ValueError(f"Prediction returned {result.path} for {path.name}")
        image_predictions = []
        if result.boxes is not None:
            for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist():
                if x2 > x1 and y2 > y1:
                    image_predictions.append((int(class_id), score, x1, y1, x2, y2))
        predictions.append(image_predictions)
    return predictions
