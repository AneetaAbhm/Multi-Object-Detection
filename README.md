# Multi Object Detection — YOLOv8

Multi-object detection system trained on 5 urban classes using **YOLOv8m**.

---

## Project Overview

This project builds a real-world urban object detection system using the **26-Class Urban Object Detection Dataset** from Kaggle. A balanced subset of 500 images across 5 classes was used for training. The trained model supports real-time detection with custom features and has been deployed both as local scripts and as a modern **web application** using FastAPI.

**Dataset:** [26-Class Object Detection Dataset](https://www.kaggle.com/datasets/mohamedgobara/26-class-object-detection-dataset)  
**Model:** YOLOv8m (medium) — Pretrained on COCO, fine-tuned on custom subset  
**Training Platform:** Google Colab (Tesla T4 GPU)

---

## Selected Classes

| Index | Class   | Train | Val | Test |
|-------|---------|-------|-----|------|
| 0     | Person  | 70    | 20  | 10   |
| 1     | bicycle | 70    | 20  | 10   |
| 2     | car     | 70    | 20  | 10   |
| 3     | tree    | 70    | 20  | 10   |
| 4     | door    | 70    | 20  | 10   |
| **Total** |     | **350** | **100** | **50** |

---

## Project Structure

```
urban-object-detection/
│
├── data_preparation/
│   ├── coco_yolo.py
│   ├── count_class.py
│   └── yolo_subset.py
│
├── training/
│   └── yolo_object_detection.ipynb
│

├── count_webcam.py        # Local real-time object counter
├── alert_detct.py         # Local alert system
└── app.py                 # FastAPI Web Interface
│
├── normal_mode.py             # Supporting file for FastAPI
├── alert_mode.py              # Supporting file for FastAPI
│
├── yolo_trained_model/
│   └── best.pt                # Trained model weights
│
└── README.md
```

---

## Results

| Metric       | Value |
|--------------|-------|
| mAP@0.5      | 0.518 |
| mAP@0.5:0.95 | 0.338 |
| Precision    | 0.526 |
| Recall       | 0.537 |

### Per-Class Performance (mAP@0.5)

| Class   | mAP@0.5 | Precision | Recall |
|---------|---------|-----------|--------|
| Person  | 0.089   | 0.185     | 0.132  |
| bicycle | 0.358   | 0.491     | 0.345  |
| car     | 0.410   | 0.695     | 0.397  |
| tree    | 0.918   | 0.598     | 0.913  |
| door    | 0.815   | 0.663     | 0.900  |

---

## Setup & Installation

```bash
pip install ultralytics opencv-python fastapi uvicorn pyyaml torch
```

Place your trained `best.pt` file inside the `yolo_trained_model/` folder.

---

## Running the Application

### Option 1: Local Webcam Modes (Original)

**1. Normal Mode (Object Counter)**
```bash
python count_webcam.py --source usb0
```

**2. Alert Mode**
```bash
python alert_detct.py --source usb0 --class_name Person
```

### Option 2: FastAPI Web Version *(Recommended for Demo & Presentation)*

This version provides a clean browser-based interface combining both Normal and Alert modes.

```bash
uvicorn app:app --reload
```

Open your browser and go to:
```
http://127.0.0.1:8000
```

**Features of FastAPI Version:**
- Clean, modern web interface accessible from any browser
- **Normal Mode:** Live stream with left-side class count panel
- **Alert Mode:** Dropdown to select class + red/green alert banner
- Runs on laptop, phone, or any device on the same network
- Professional and easy to demonstrate during presentations
- Single command to run both features

---

## Data Preparation *(Run Locally Before Training)*

**Step 1 — Convert COCO to YOLO format**
```bash
python data_preparation/coco_yolo.py --dataset_path "C:/path/to/dataset"
```

**Step 2 — Check class image counts**
```bash
python data_preparation/count_class.py
```

**Step 3 — Create 500-image balanced subset**
```bash
python data_preparation/yolo_subset.py --dataset_path "C:/path/to/dataset"
```

---

## Training *(Google Colab)*

1. Upload `yolo_subset.zip` to Google Drive
2. Open `training/yolo_object_detection.ipynb` in Colab
3. Set runtime to **T4 GPU**
4. Run all cells

**Training Configuration:**

| Parameter  | Value      |
|------------|------------|
| Model      | yolov8m.pt |
| Epochs     | 50         |
| Batch size | 16         |
| Image size | 640×640    |
| Patience   | 20         |

---

## Custom Features

### 1. Real-Time Object Counter (`count_webcam.py`)
Displays live webcam feed  with a detailed side panel showing the count of each detected object class in real time.

### 2. Alert System (`alert_detct.py`)
Monitors a specific class and displays a **red alert banner** if the class is not detected, and a **green banner** when it is present. Works on webcam.

### 3. FastAPI Web Interface (`app.py`)
Modern web deployment combining both features:
- Easy mode selection from the browser
- Same visual style as local versions
- Suitable for presentations and remote sharing

> **Tip:** For best presentation, demonstrate the local scripts first (to show speed), then switch to the FastAPI version to showcase modern deployment.

---

## Key Learnings

- YOLO bounding box format uses normalized `[cx, cy, w, h]` values in the 0–1 range
- Transfer learning from COCO pretrained weights greatly reduces training time and improves accuracy
- Data quality matters more than quantity on small custom datasets
- YOLOv8m performed significantly better than smaller variants on this dataset
- FastAPI + MJPEG streaming enables clean and easy web deployment of computer vision models
- Confidence threshold tuning is critical — lower values (0.25) work better for less common classes like Person

---

## Requirements

```
ultralytics
opencv-python
fastapi
uvicorn
pyyaml
torch
```
