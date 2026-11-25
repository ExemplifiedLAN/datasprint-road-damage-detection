# DataSprint Road Damage Detection

Code Alchemists' entry for Anveshan DataSprint 2.0. The YOLO26m detector locates cracks and potholes in road photos and returns their class, confidence and bounding box in original-image pixels. The included checkpoint and 4,200-row CSV come from the run that scored **44.22804 / 100** on the public Kaggle leaderboard. [Competition](https://www.kaggle.com/competitions/datasprint-2-road-damage-detection) · [Submitted notebook version](https://www.kaggle.com/code/bhuvanashokhottin/notebook2f8f471f5e?scriptVersionId=373243941)

| Class | Code | Damage |
| --- | --- | --- |
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

## Run inference

1. Download and extract the [competition kit](https://www.kaggle.com/competitions/datasprint-2-road-damage-detection/data). Keep its scripts, image folders, and CSV files together. The test command uses `test_images/`, `test_image_ids.csv`, `make_kaggle_submission.py`, and `check_submission.py`.
2. From the repository root, use Python 3.10 or newer with PyTorch and torchvision. Install CUDA-enabled builds for GPU inference. Install the kit dependencies first, then the repository's Ultralytics pin. [PyTorch installation](https://pytorch.org/get-started/locally/)

   ```bash
   python -m pip install -r "/path/to/competition-kit/requirements.txt"
   python -m pip install -r requirements.txt
   ```

3. Generate and check a new submission with the trained checkpoint.

   ```bash
   python reproduce.py --kit "/path/to/competition-kit" --device 0
   ```

The output is `generated/submission.csv`. Each image has one CSV row. The `PredictionString` contains repeating `class score x_min y_min x_max y_max` groups, or `none` when no boxes pass the threshold. Change `--device 0` to `--device cpu` for a CPU run. Once dependencies, competition files, and the checkpoint are in place, inference reads local files and runs offline.

## What the code does

| Path | Purpose |
| --- | --- |
| [inference.py](inference.py) | Loads the checkpoint from disk and returns damage boxes for a batch of images. |
| [reproduce.py](reproduce.py) | Reads the organizer's image ID list, runs inference in groups of eight, and writes detections. On test data it calls the organizer's formatter and submission checker. On validation data it calls the official scorer. |
| [weights/best.pt](weights/best.pt) | Trained YOLO26m checkpoint used for the final entry. |
| [submission.csv](submission.csv) | Original scored predictions for all 4,200 test images. |
| [requirements.txt](requirements.txt) | Ultralytics version used to save the model. |
| [training/](training/) | Original Kaggle training notebooks, epoch metrics, and validation scores. |

The final inference settings are 640 px input, a 0.001 confidence floor, an IoU argument of 0.7, and a maximum of 100 boxes per image. The application keeps the submitted CSV as a reference and generates a fresh file from the images and model.

## Training and data

The competition supplies 14,691 training images, 2,150 validation images, and 4,200 test images derived from RDD2020. Some test photos have blur, low light, noise, or JPEG compression. The notebooks convert pixel boxes to normalized YOLO labels, clip coordinates to image boundaries, and create empty label files for images without annotations. The final notebook adds degraded copies of a seeded 10% sample of the training split using brightness reduction, Gaussian blur, noise, or JPEG compression.

To retrain, open the [YOLO26m notebook](training/road_damage_yolo26m_second_entry.ipynb) in Kaggle, attach the competition data, enable a GPU accelerator, and run the cells in order. The run used 640 px inputs, a batch size of 16 across two T4 GPUs, seed 42, a 65-epoch cap, and patience of 15 epochs. It stopped at epoch 59 and selected the checkpoint from epoch 44. The final model started from Ultralytics `yolo26m.pt` pretrained on COCO. The [earlier YOLOv8s run](training/road_damage_yolov8s_50_epochs.ipynb) used COCO-pretrained `yolov8s.pt`. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

## Evaluate on validation images

```bash
python reproduce.py --kit "/path/to/competition-kit" --split val --device 0
```

| Evaluation | Score |
| --- | ---: |
| Official validation at 640 px | 45.27 |
| Official validation at 768 px | 44.64 |
| Public Kaggle leaderboard for the included CSV | 44.22804 |

The original run used 640 px for its final submission based on the [validation scores](training/validation_scores.json). A fresh timed run on one offline T4 and a hidden-test score comparison remain unverified.

## Data credit

The competition images are derived from RDD2020 by Arya et al. under CC BY-NC 3.0. Obtain the images and starter-kit scripts from the competition page. [Dataset citation](https://doi.org/10.17632/5ty2wb6gvg.1) · [Dataset license](https://creativecommons.org/licenses/by-nc/3.0/)
