"""Regenerate the competition CSV or score the official validation split."""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from inference import BATCH_SIZE, WEIGHTS, load_model, predict_paths


BOX_FIELDS = ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_inputs(kit, split):
    image_dir = kit / f"{split}_images"
    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(f"Ambiguous image ID in {image_dir}: {path.stem}")
            images[path.stem] = path

    ids_path = kit / f"{split}_image_ids.csv"
    with ids_path.open(newline="", encoding="utf-8-sig") as handle:
        ids = [row["image_id"].strip() for row in csv.DictReader(handle)]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate IDs in {ids_path}")
    missing = [image_id for image_id in ids if Path(image_id).stem not in images]
    if missing:
        raise FileNotFoundError(f"{len(missing)} IDs have no image in {image_dir}, including {missing[0]}")
    return [(image_id, images[Path(image_id).stem]) for image_id in ids]


def export_boxes(model, inputs, destination, device):
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BOX_FIELDS)
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start:start + BATCH_SIZE]
            boxes = predict_paths(model, [path for _, path in batch], device=device)
            for (image_id, _), image_boxes in zip(batch, boxes, strict=True):
                writer.writerows([image_id, *box] for box in image_boxes)
            if (start // BATCH_SIZE + 1) % 50 == 0 or start + BATCH_SIZE >= len(inputs):
                print(f"Inferred {min(start + BATCH_SIZE, len(inputs))}/{len(inputs)} images", flush=True)


def run_kit_script(kit, work, name, *args):
    script = kit / name
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=work, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True, help="Extracted competition data and starter kit.")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--out", type=Path, help="Output CSV path.")
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu for a local check.")
    args = parser.parse_args()
    kit = args.kit.resolve()
    output = (args.out or Path("generated") / ("submission.csv" if args.split == "test" else "val_pred.csv")).resolve()
    if output in {WEIGHTS, Path(__file__).resolve().parent / "submission.csv"}:
        parser.error("Choose a new output path to preserve the packaged checkpoint and submitted CSV.")

    if args.split == "test":
        required = ["make_kaggle_submission.py", "check_submission.py", "test_image_ids.csv"]
    else:
        required = ["score.py", "val_labels.csv", "val_image_ids.csv"]
    for name in required:
        if not (kit / name).is_file():
            parser.error(f"Missing official starter-kit file: {kit / name}")

    start = perf_counter()
    inputs = image_inputs(kit, args.split)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="datasprint_", dir=output.parent) as scratch:
        work = Path(scratch)
        predictions = work / "predictions.csv"
        model = load_model()
        export_boxes(model, inputs, predictions, args.device)
        if args.split == "test":
            candidate = work / "submission.csv"
            ids = kit / "test_image_ids.csv"
            run_kit_script(
                kit, work, "make_kaggle_submission.py",
                "--pred", predictions, "--image-ids", ids, "--out", candidate,
            )
            run_kit_script(kit, work, "check_submission.py", "--pred", candidate, "--image-ids", ids)
        else:
            candidate = predictions
            run_kit_script(
                kit, work, "score.py",
                "--pred", predictions, "--gt", kit / "val_labels.csv",
                "--image-ids", kit / "val_image_ids.csv",
            )
        shutil.copyfile(candidate, output)
    print(f"Wrote {output} for {len(inputs):,} images in {perf_counter() - start:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
