# PCB Defect Detection Using YOLOv8

Automated detection system for identifying manufacturing defects in printed circuit boards using YOLOv8 object detection. This project implements incremental learning across balanced data subsets to handle memory constraints while maintaining model quality.

## Project Goal

Build a robust PCB defect detection model that identifies six common manufacturing defects: missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper deposits. The system is designed to catch defects before PCB assembly, preventing costly failures in production.

## Why This Project

Manual PCB inspection is slow and error-prone, and defects that slip through are far more expensive to fix once a board is assembled. This project explores whether an object detection model can catch these defects automatically and reliably enough to support a real inspection workflow.

It's also a hands-on study of how models are actually trained in production: you rarely have your entire dataset on day one. More data gets collected over time, and the model needs to keep learning from it without forgetting what it already knows or requiring a full retrain from scratch every time.

## Dataset

The project uses the [PCB Defect Dataset](https://universe.roboflow.com/test-jksby/pcb_defect-9cisw) hosted on Roboflow in COCO format.

### Dataset Statistics

Total Images: 10,628

| Split | Images | Annotations | Avg per Image | Resolution |
|-------|--------|-------------|---------------|------------|
| Train | 6,377 | 13,024 | 2.04 | 600x600 |
| Valid | 2,124 | 4,314 | 2.03 | 600x600 |
| Test | 2,127 | 4,278 | 2.01 | 600x600 |

### Defect Classes

All six classes are evenly represented (16-17% each):

| Class | Count | Percentage |
|-------|-------|-----------|
| Spurious Copper | 2,233 | 17.1% |
| Spur | 2,227 | 17.1% |
| Mouse Bite | 2,186 | 16.8% |
| Open Circuit | 2,141 | 16.4% |
| Short | 2,123 | 16.3% |
| Missing Hole | 2,114 | 16.2% |

### Defect Sizes

Defects range from 132 to 8,925 square pixels:

| Size | Count | Percentage |
|------|-------|-----------|
| Tiny (< 100 px) | 0 | 0% |
| Small (100-500 px) | 2,536 | 19% |
| Medium (500-2000 px) | 9,836 | 75% |
| Large (> 2000 px) | 652 | 5% |

The majority of defects are medium-sized and easily detectable.

## Approach: Incremental Training on Balanced Subsets

Rather than training on the full dataset at once, this project splits training data into 6 balanced subsets and uses incremental learning, fine-tuning the model on one new subset at a time. This mimics how models improve in the real world: instead of having all the data upfront, you get new batches over time as more is collected, and each time you retrain the existing model on the new data rather than starting over. The model should keep getting better as it sees more data, subset by subset.

The test set is kept completely separate from every subset and is only used once, at the very end, to evaluate the final model. This avoids data leakage and ensures the reported performance reflects how the model would do on truly unseen boards, not data it has already learned from.

```
Subset 1 (1,063 train + 354 valid) -> Train -> model_v1.pt
                    |
Subset 2 (1,063 train + 356 valid) -> Fine-tune model_v1 -> model_v2.pt
                    |
Subset 3 (1,063 train + 355 valid) -> Fine-tune model_v2 -> model_v3.pt
                    |
... continue for subsets 4, 5, 6
                    |
                 model_v6.pt
                    |
         Test Set (2,127 images - held out)
                    |
            Final Evaluation
```

### Why This Approach

1. Memory Efficiency: Reduces peak memory usage by 75% compared to full dataset training
2. Controlled Learning: Model progressively refines across subsets
3. Balanced Data: Each subset maintains equal class distribution
4. Test Integrity: Held-out test set never touches training or validation
5. Production Relevance: Mimics continuous learning in manufacturing environments

## Data Subset Structure

Training data split into 6 subsets with balanced class representation:

| Subset | Train | Valid | Total |
|--------|-------|-------|-------|
| 1 | 1,063 | 357 | 1,420 |
| 2 | 1,064 | 356 | 1,420 |
| 3 | 1,064 | 355 | 1,419 |
| 4 | 1,063 | 352 | 1,415 |
| 5 | 1,062 | 352 | 1,414 |
| 6 | 1,060 | 352 | 1,412 |
| Held-Out Test | - | - | 2,127 |

Each subset is stratified to maintain global class distribution.

## Model Specifications

- Architecture: YOLOv8n (nano)
- Transfer Learning: Pre-trained COCO weights
- Hardware: Mac M1 with CPU training
- Image Size: 416x416
- Batch Size: 8
- Epochs: 100
- Learning Rate: 0.001
- Optimizer: SGD
- Early Stopping Patience: 10 epochs

## Results

### Training Progression Across Subsets

| Subset | Best mAP50 | Best Precision | Best Recall | Epochs | Training Time |
|--------|-----------|----------------|------------|--------|---------------|
| Subset 1 | 0.8832 | 0.9372 | 0.8134 | 93 | 3.4 hours |
| Subset 2 | 0.8971 | 0.9266 | 0.8339 | 12 | 0.4 hours |
| Subset 3 | 0.9255 | 0.9224 | 0.8757 | 26 | 0.9 hours |
| Subset 4 | 0.9214 | 0.9460 | 0.8607 | 12 | 0.4 hours |
| Subset 5 | 0.9169 | 0.9189 | 0.8650 | 12 | 0.4 hours |
| Subset 6 | 0.9454 | 0.9287 | 0.9035 | 28 | 1.2 hours |

Incremental training shows consistent improvement in mAP50 across subsets. Model converges faster on subsequent subsets while retaining and building upon knowledge from previous training.

### Final Test Set Evaluation

Evaluation on completely held-out test set (2,127 unseen images):

| Metric | Value |
|--------|-------|
| mAP50 | 0.9338 |
| mAP50-95 | 0.4224 |
| Precision | 0.9395 |
| Recall | 0.8927 |

### Per-Class Performance

| Class | AP50 | Precision | Recall |
|-------|------|-----------|--------|
| missing_hole | 0.9800 | 0.98 | 0.98 |
| mouse_bite | 0.9310 | 0.91 | 0.91 |
| open_circuit | 0.9410 | 0.90 | 0.90 |
| short | 0.9360 | 0.93 | 0.93 |
| spur | 0.8910 | 0.83 | 0.83 |
| spurious_copper | 0.9210 | 0.90 | 0.90 |

Model performs exceptionally well across all defect classes. Missing hole detection is the strongest (0.98 AP50), while spur detection is most challenging (0.89 AP50) but still well above acceptable thresholds.

## Project Structure

```
PCB-Defect-Detection/
├── data/
│   ├── PCB_defect.v1i.coco/       (raw Roboflow download, COCO format)
│   │   ├── train/                 (6,377 images)
│   │   ├── valid/                 (2,124 images)
│   │   ├── test/                  (2,127 images, held out)
│   │   ├── README.dataset.txt
│   │   └── README.roboflow.txt
│   └── yolo_format/               (full dataset converted to YOLO format)
│       ├── images/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── dataset.yaml
│       └── dataset_test.yaml
│
├── data_subsets/
│   ├── subset_1/ ... subset_6/    (same structure in each)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── valid/
│   │   ├── labels/
│   │   │   ├── train/             (YOLO format .txt annotations)
│   │   │   └── valid/
│   │   └── dataset.yaml
│
├── code/
│   ├── split-dataset.py           (splits COCO data into subsets)
│   ├── train.py                   (trains on one subset)
│   ├── finalEval.py               (evaluates on test set)
│   ├── explore-data.py            (dataset exploration)
│   ├── yolov8n.pt                 (base pretrained YOLOv8n weights)
│   ├── results/
│   │   └── subset_1/ ... subset_6/
│   │       ├── weights/           (best.pt, last.pt)
│   │       ├── model_vN.pt        (final model checkpoint for the subset)
│   │       ├── validation_metrics_vN.txt
│   │       ├── args.yaml
│   │       ├── results.csv
│   │       ├── confusion_matrix.png, confusion_matrix_normalized.png
│   │       ├── BoxP_curve.png, BoxR_curve.png, BoxPR_curve.png, BoxF1_curve.png
│   │       └── labels.jpg, results.png, train_batch*.jpg, val_batch*.jpg
│   ├── runs/
│   │   └── detect/
│   │       ├── val/ , val-2/ ... val-9/   (per-subset validation runs; val-9 is the final test evaluation)
│   │       │   ├── confusion_matrix.png
│   │       │   ├── confusion_matrix_normalized.png
│   │       │   ├── BoxP_curve.png
│   │       │   ├── BoxR_curve.png
│   │       │   ├── BoxPR_curve.png
│   │       │   ├── BoxF1_curve.png
│   │       │   └── val_batch*_labels.jpg, val_batch*_pred.jpg
│   └── scraps/                    (earlier/experimental scripts, not part of the main pipeline)
│       ├── config.py, dataset.py, inference.py, main.py, pipeline.py, train1.py, utils.py
│       ├── yolov8n.pt, yolov8m.pt
│       └── results/
│
└── README.MD
```

## Usage

### 1. Download Dataset

Download the [PCB Defect Dataset](https://universe.roboflow.com/test-jksby/pcb_defect-9cisw) from Roboflow and extract to `data/PCB_defect.v1i.coco/`.

### 2. Split Dataset into Subsets

```bash
python code/split-dataset.py
```

This converts COCO format annotations to YOLO format and creates 6 balanced subsets.

### 3. Train on Each Subset

Train model incrementally on each subset:

```bash
python code/train.py --subset 1
python code/train.py --subset 2
python code/train.py --subset 3
python code/train.py --subset 4
python code/train.py --subset 5
python code/train.py --subset 6
```

Each subset trains the model for 100 epochs with early stopping. The model is loaded from the previous subset and fine-tuned on new data.

Estimated time per subset: 3-4 hours on Mac M1 CPU.

### 4. Evaluate on Test Set

After all subsets complete, evaluate final model on held-out test set:

```bash
python code/finalEval.py --model code/results/subset_6/model_v6.pt
```

## Technical Stack

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy

## Future Improvements

- Generate detailed per-class performance analysis on test set
- Create visualizations of model predictions on test images
- Analyze failure cases and common misclassifications
- Document inference time and throughput on different hardware
- Deploy model as REST API or edge service

## Author

Tamal Das

## Acknowledgments

Dataset provided by Roboflow from the PCB defect detection community.
