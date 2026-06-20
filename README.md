---
title: Teeth Segmentation (OdonoAI)
emoji: 🦷
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.29.0"
app_file: app/gradio_demo.py
pinned: true
---

# Dental Teeth Segmentation — Mask R-CNN

Instance segmentation of individual teeth in dental panoramic X-ray images using a fine-tuned **Mask R-CNN** (ResNet-50+FPN, torchvision, COCO pre-trained weights).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Configuration](#6-configuration)
7. [Setup & Installation](#7-setup--installation)
8. [Usage](#8-usage)
9. [API](#9-api)
10. [Docker](#10-docker)
11. [Results](#11-results)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [References](#14-references)

---

## 1. Project Overview

This project uses torchvision Mask R-CNN for **dental tooth instance segmentation**. Each tooth in a panoramic X-ray is detected and segmented independently, producing a per-tooth binary mask, bounding box, confidence score and FDI class label.

**What was done:**
- Implemented a custom `TeethDataset` loader that parses COCO JSON annotations
- Applied **CLAHE** contrast enhancement as a preprocessing step to improve tooth boundary visibility
- Fine-tuned the whole network (starting from COCO weights) on the OdontoAI dataset (1597 train / 400 val / 2000 test panoramic X-ray images)
- Supports **52 FDI classes** — adult permanent teeth (11–48) and deciduous/primary teeth (51–85)
- Evaluated with pycocotools — COCO-style mAP@50 and mAP@50-95
- Exposed predictions via a **FastAPI** REST endpoint
- Built a **Gradio** demo with sample images from previously unseen test images
- Containerised the full stack with Docker
- Weights hosted on Hugging Face Hub, demo deployed on Hugging Face Spaces

---

## 2. Project Structure

```
dental-segmentation/
│
├── app/
│   ├── gradio_demo.py           # Gradio application
│   ├── main.py                  # FastAPI 
│   └── *.jpg                    # Example images for Gradio gallery
│
├── configs/
│   └── model_config.py          # Model configurations
│
├── models/
│   └── teeth_segmentation.py    # TeethDataset, train(), build_model(),
│                                #   predict(), evaluate(), load_inference_model()
│
├── utils/
│   ├── preprocessing.py         # Image loading, CLAHE, COCO parsing
│   ├── visualization.py         # Mask overlays, EDA plots
│
├── notebooks/
│   ├── EDA.ipynb                # Exoloratory data analysis
│   └── Evaluation.ipynb         # Validation metrics and prediction visualisations
│
├── data/
│   ├── images/                  # Train + val images (gitignored)
│   ├── test/                    # Unseen test images (gitignored)
│   └── annotations/
│       ├── train.json           # 1597 images (gitignored)
│       └── val.json             # 400 images (gitignored)
│
├── outputs/
│   ├── results/
│   │   └── maskrcnn_torch/      # best.pth, last.pth, training_history.csv
│   └── visualizations/          # Results after predict()
apploed onto the test images
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Dataset

### Source
**OdontoAI** — a large-scale open dataset of dental panoramic radiographs, annotated with per-tooth polygon segmentation masks in COCO JSON format.

- **Train:** 1597 images · 45,426 annotations
- **Val:** 400 images · 11,453 annotations
- **Test:** 2000 images · unannotated
Dataset available at: https://arxiv.org/abs/2203.15856

### Annotation format (COCO JSON)

```json
{
  "images": [{"id": 1, "file_name": "pan-00577.jpg", "width": 2440, "height": 1292}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[1128, 498, 1180, 502, 1185, 620, 1130, 618]],
      "bbox": [1128, 498, 109, 246],
      "area": 18583
    }
  ],
  "categories": [{"id": 1, "name": "tooth-11", "supercategory": "tooth"}, ...]
}
```

### Categories — FDI numbering system

| Group | FDI range | Classes |
|---|---|---|
| Adult permanent — upper right | 11–18 | 8 |
| Adult permanent — upper left  | 21–28 | 8 |
| Adult permanent — lower left  | 31–38 | 8 |
| Adult permanent — lower right | 41–48 | 8 |
| Deciduous upper               | 51–55, 61–65 | 10 |
| Deciduous lower               | 71–75, 81–85 | 10 |
| **Total**                     |  | **52** |

### Split
- Pre-split into train/val/test — no manual splitting required

### Preprocessing
1. **Load** — grayscale X-rays are converted to 3-channel RGB (required by ResNet backbone)
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation on the L channel (LAB colour space) to improve local contrast at tooth boundaries. Applied during training and inference.
3. **Resize** — handled internally by torchvision (shorter side → 800px, longer side capped at 1333px)

---

## 4. Model Architecture

**Mask R-CNN** with:
- **Backbone:** ResNet-50 + Feature Pyramid Network (FPN)
- **Region Proposal Network (RPN):** generates tooth candidate regions
- **ROI Align:** extracts fixed-size features per proposal
- **Heads:** classification, bounding-box regression, and mask prediction heads

Pre-trained on **MS-COCO** (80 classes). The final heads are replaced and retrained for 52 FDI tooth classes + background (53 total).

### Key Configuration (`configs/model_config.py`)

| Parameter | Value | Reason |
|---|---|---|
| `NUM_CLASSES` | 53 | background + 52 FDI classes |
| `IMAGE_MIN_SIZE` | 800 | torchvision default |
| `IMAGE_MAX_SIZE` | 1333 | torchvision default |
| `ANCHOR_SIZES` | (16, 32,64,128,256) | standard FPN anchors |
| `CONF_THRESHOLD` | 0.25 | lower than default to retain partially occluded teeth |
| `NMS_THRESHOLD` | 0.6 | higher than default, because adjacent teeth naturally overlap |
| `MAX_DETECTIONS` | 60 | increased for mixed adult+deciduous dentition cases |
| `BATCH_SIZE` | 2 | NVIDIA RTX 4090 |

---

## 5. Training Strategy

Single-stage fine-tuning using SGD with early stopping.

- **Optimizer:** SGD — lr=0.001, momentum=0.9, weight_decay=0.0005
- **Scheduler:** StepLR — step_size=10, gamma=0.1 (LR drops 10× every 10 epochs)
- **Epochs:** up to 35 (early stopping with patience=10)
- **Batch size:** 2
- **Early stopping:** triggered when val loss does not improve for 10 consecutive epochs

All layers are trained from epoch 1.

Best checkpoint saved to `outputs/results/maskrcnn_torch/best.pth` based on lowest validation loss and saved on Huggingface Hub.

---

## 6. Configuration

All model parameters are in `configs/model_config.py`.

---

## 7. Setup & Installation

### Requirements
- Python 3.10
- CUDA 11.7+ / cuDNN 8 (for GPU training)
- 8 GB+ GPU VRAM recommended

### Steps

```bash
# 1. Clone this repo
git clone https://github.com/dorpetrovic/teeth-segmentation-odontoai

# 2. Create conda environment
conda create -n odonto-dental python=3.10
conda activate odonto-dental

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data
# Put train+val images in data/images/
# Put test images in data/test/
# Annotations already in data/annotations/train.json and val.json
```
---

## 8. Usage

### Training

```bash
python models/teeth_segmentation.py train
```

### Evaluate (calculates mAP on validation set)

```bash
python models/teeth_segmentation.py evaluate
```

### Inference on a single image

```bash
python models/teeth_segmentation.py predict --image data/test/pan-00001.jpg
```

Output saved to `outputs/visualizations/`.

---

## 9. API

Start the inference server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Upload image, receive segmentation results |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@xray.jpg" \
    | python -m json.tool
```

**Example response:**
```json
{
  "n_teeth": 28,
  "boxes": [[120, 80, 148, 110], ...],
  "labels": [1, 3, 5, ...],
  "class_names": ["tooth-11", "tooth-13", "tooth-15", ...],
  "scores": [0.97, 0.95, 0.92, ...],
  "masks_b64": ["iVBORw0KGgo...", ...],
  "overlay_b64": "/9j/4AAQSkZ..."
}
```

Interactive docs at: `http://localhost:8000/docs`

---

## 10. Docker

```bash
# Build
docker build -t dental-maskrcnn-torch:latest .

# Run API
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/data:/app/data \
    dental-maskrcnn-torch:latest

# Run Gradio demo
docker run --gpus all -p 7860:7860 \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/data:/app/data \
    dental-maskrcnn-torch:latest \
    python app/gradio_demo.py

# Train
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    --shm-size=4gb \
    --name maskrcnn_training \
    dental-maskrcnn-torch:latest \
    python models/teeth_segmentation.py train

# Evaluate
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    dental-maskrcnn-torch:latest \
    python models/teeth_segmentation.py evaluate
```

---

## 11. Results

Training and validation outputs are saved to `outputs/`:
- `outputs/results/maskrcnn_torch/best.pth` — best model weights
- `outputs/results/maskrcnn_torch/last.pth` — latest checkpoint
- `outputs/results/maskrcnn_torch/training_history.csv` — per-epoch loss breakdown
- `outputs/visualizations/` — predictions

---

## 12. Evaluation Metrics

Evaluated with **pycocotools** (COCO standard):

| Metric | Description |
|---|---|
| **mAP@50** | Mean AP at IoU threshold 0.50 |
| **mAP@50-95** | Mean AP averaged over IoU 0.50:0.05:0.95 |
| **MAE** | Mean absolute error on tooth count per image |

```bash
python models/teeth_segmentation.py evaluate
```
or run the evaluation.ipynb
---

## 13. References

- Silva B, Pinheiro L, Sobrinho B, Lima F, Sobrinho B, Abdalla K, Pithon M, Cury P, Oliveira L. Boosting research on dental panoramic radiographs: a challenging data set, baselines, and a task central online platform for benchmark. *Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization*. 2023;0(0):1-21. doi:10.1080/21681163.2022.2157747

- Lin T-Y, et al. Microsoft COCO: Common Objects in Context. https://arxiv.org/abs/1405.0312

- He K, et al. Mask R-CNN. https://arxiv.org/abs/1703.06870

### Citation


```bibtex
@article{doi:10.1080/21681163.2022.2157747,
  author    = {Bernardo Peters Menezes Silva and Laís Bastos Pinheiro and
               Brenda Pereira Pinheiro Sobrinho and Fernanda Pereira Lima and
               Bruna Pereira Pinheiro Sobrinho and Kalyf Abdalla Buzar Lima and
               Matheus Melo Pithon and Patricia Ramos Cury and
               Luciano Rebouças de Oliveira},
  title     = {Boosting research on dental panoramic radiographs: a challenging
               data set, baselines, and a task central online platform for benchmark},
  journal   = {Computer Methods in Biomechanics and Biomedical Engineering:
               Imaging \& Visualization},
  volume    = {0},
  number    = {0},
  pages     = {1-21},
  year      = {2023},
  publisher = {Taylor \& Francis},
  doi       = {10.1080/21681163.2022.2157747}
}
```