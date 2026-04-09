# MoE Edge Detector

A **Mixture-of-Experts** appliance detection pipeline using **MobileNetV3-Small** as a shared backbone with specialized expert models routed by a lightweight MLP.

## 🧠 Architecture

```
Image
 ↓
Preprocess (320×320)
 ↓
MobileNetV3-Small (backbone)
 ↓
GAP → (960-d)
 ↓
Router MLP
 ↓
Expert Selection
 ↓
Expert Model
 ↓
NMS → Output
```

### Expert Groups

| Expert | ID | Classes |
|--------|----|---------|
| Display | 0 | TV |
| Kitchen | 1 | Refrigerator, Microwave |
| Climate | 2 | Air Conditioner |

## 📁 Project Structure

```
moe-edge-detector/
├── data/
│   ├── train/
│   │   ├── tv/
│   │   ├── refrigerator/
│   │   ├── microwave/
│   │   └── air_conditioner/
│   ├── val/
│   ├── gap_vectors_train.npy
│   └── gap_labels_train.npy
├── models/
│   ├── backbone.pt
│   ├── router.pt
│   ├── expert_0_display.pt
│   ├── expert_1_kitchen.pt
│   └── expert_2_climate.pt
├── src/
│   ├── preprocess.py
│   ├── backbone.py
│   ├── extract_gap.py
│   ├── router.py
│   ├── nms_utils.py
│   └── inference_demo.py
├── train/
│   ├── train_backbone.py
│   ├── generate_gap.py
│   └── train_router.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your training images in `data/train/` using an **ImageFolder** layout:

```
data/train/
├── tv/
│   ├── img001.jpg
│   └── ...
├── refrigerator/
│   ├── img001.jpg
│   └── ...
├── microwave/
│   ├── img001.jpg
│   └── ...
└── air_conditioner/
    ├── img001.jpg
    └── ...
```

### 3. Train backbone

```bash
python train/train_backbone.py
```

Finetunes MobileNetV3-Small on your appliance classes. Saves to `models/backbone.pt`.

### 4. Generate GAP dataset

```bash
python train/generate_gap.py
```

Extracts 960-d GAP vectors from every training image and maps classes to expert groups. Saves to `data/gap_vectors_train.npy` and `data/gap_labels_train.npy`.

### 5. Train router

```bash
python train/train_router.py
```

Trains the lightweight MLP router on GAP vectors. Saves best checkpoint to `models/router.pt`.

### 6. Run inference

```bash
python src/inference_demo.py <image_path>
```

Runs the full pipeline: preprocess → backbone → GAP → router → expert selection.

## ✅ What You Get

- **MobileNet backbone** — Lightweight, pretrained feature extractor
- **Router MLP** — Fast expert gating (960 → 256 → 3)
- **Expert specialization** — Domain-specific detection heads
- **NMS utilities** — Clean post-processing with per-class suppression
- **Clean PyTorch pipeline** — No unnecessary complexity
