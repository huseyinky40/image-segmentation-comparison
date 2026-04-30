# Product Background Removal via Image Segmentation

A comparative study of **6 classical image segmentation algorithms** implemented in Python/OpenCV for automatic background removal on product photographs. Each algorithm is evaluated against manually annotated ground-truth masks using pixel-wise metrics.

> **Course:** Image Processing  
> **Authors:** Hüseyin Kaya · Ada Şevval Sari

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

This project implements and benchmarks six image segmentation approaches on a dataset of **6 Jordan toothpaste product photographs**. The goal is to isolate the product from its background — a fundamental task in e-commerce image processing.

Each algorithm generates a **binary mask** (foreground = white, background = black). Masks are then compared to manually created ground-truth masks to compute quantitative performance scores. The best-performing mask is then applied to the original color image to produce the final background-removed result.

---

## Algorithms

| # | Algorithm | Type | Key Parameters |
|---|-----------|------|----------------|
| 1 | **Otsu's Thresholding** | Global threshold | Auto-computed optimal threshold |
| 2 | **Adaptive Thresholding** | Local threshold | Block size: 11, C: 2 |
| 3 | **Canny Edge Detection** | Edge-based | Low: 100, High: 200 |
| 4 | **GrabCut** | Graph-cut | 5 iterations, manual bounding box |
| 5 | **K-Means Clustering** | Unsupervised clustering | K=3, luminance-based cluster selection |
| 6 | **Watershed** | Marker-based | Distance transform + connected components |

All algorithms include morphological post-processing (opening/closing) to refine the binary masks.

---

## Dataset

- **6 product images** (Jordan toothpaste) in JPEG format
- **6 ground-truth binary masks** manually annotated per image
- Images preprocessed with **Bilateral Filtering** (d=15, σColor=75, σSpace=75) and converted to grayscale

---

## Project Structure

This is the exact folder structure you will find in this repository:

```
├── Demo/
│   │
│   ├── original_images/                  # Raw input images
│   │   ├── 1.jpeg
│   │   ├── 2.jpeg
│   │   ├── 3.jpeg
│   │   ├── 4.jpeg
│   │   ├── 5.jpeg
│   │   └── 6.jpeg
│   │
│   ├── masked_images/                    # Ground truth binary masks (manually annotated)
│   │   ├── 1_masked.png
│   │   ├── 2_masked.png
│   │   ├── 3_masked.png
│   │   ├── 4_masked.png
│   │   ├── 5_masked.png
│   │   └── 6_masked.png
│   │
│   ├── denoised_images/                  # Preprocessed grayscale images (output of step 1)
│   │   ├── 1_denoised.png
│   │   └── ...
│   │
│   ├── {1-6}_algo_result_{method}.png    # Binary mask outputs for each algorithm (36 files)
│   │
│   ├── final_bg_removed_results/         # Final background-removed color images (36 files)
│   │   ├── 1_final_otsu.png
│   │   ├── 1_final_adaptive.png
│   │   ├── 1_final_canny.png
│   │   ├── 1_final_grabcut.png
│   │   ├── 1_final_kmeans.png
│   │   ├── 1_final_watershed.png
│   │   └── ...
│   │
│   ├── 01_preprocessing.py               # Step 1 — Bilateral filter + grayscale
│   ├── 02_mask_alignment_check.py        # Step 2 — Ground truth alignment check
│   ├── 03_segmentation_otsu_thrs.py      # Step 3a — Otsu's thresholding
│   ├── 03_segmentation_adaptive_thrs.py  # Step 3b — Adaptive thresholding
│   ├── 03_segmentation_canny_edge.py     # Step 3c — Canny edge detection
│   ├── 03_segmentation_grabcut.py        # Step 3d — GrabCut
│   ├── 03_segmentation_kmeans.py         # Step 3e — K-Means clustering
│   ├── 03_segmentation_watershed.py      # Step 3f — Watershed
│   ├── 04_final_performance_report.py    # Step 4 — Metrics + comparison chart
│   └── 05_bg_removal.py                  # Step 5 — Background removal pipeline
│
├── presentation.pdf   # Project presentation
└── report.pdf   # Project report
```

---

## Pipeline

```
original_images/
        │
        ▼
01_preprocessing.py
  → Bilateral Filter + Grayscale conversion
        │
        ▼
denoised_images/
        │
        ├──▶ 02_mask_alignment_check.py  (optional — verifies GT mask alignment)
        │
        ▼
03_segmentation_*.py  (run all 6 scripts)
  → Produces: {1-6}_algo_result_{method}.png  (binary masks)
        │
        ▼
04_final_performance_report.py
  → Computes: Accuracy, Precision, Recall, F1-Score, IoU, MSE
  → Outputs: printed summary table + bar chart
        │
        ▼
05_bg_removal.py
  → Applies masks to original color images
  → Outputs: final_bg_removed_results/{num}_final_{method}.png
```

---

## Evaluation Metrics

All algorithms are evaluated using **pixel-wise comparison** against ground-truth masks:

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / Total | Overall correct pixel classification |
| **Precision** | TP / (TP + FP) | Avoids labeling background as foreground |
| **Recall** | TP / (TP + FN) | Captures the entire foreground object |
| **F1-Score** | 2 · (P · R) / (P + R) | Harmonic mean of Precision and Recall |
| **IoU** | TP / (TP + FP + FN) | Overlap ratio — primary ranking metric |
| **MSE** | mean((GT − Pred)²) | Pixel-level squared error |

> TP = True Positive · FP = False Positive · FN = False Negative · TN = True Negative

---

## Results

Mean performance scores across all 6 images, sorted by IoU (best → worst):

| Model | Accuracy (%) | Precision | Recall | F1-Score | IoU | MSE |
|-------|:------------:|:---------:|:------:|:--------:|:---:|:---:|
| **GrabCut** | 90.11 | 0.38 | 0.46 | 0.38 | **0.27** | 6428.48 |
| **Adaptive** | 92.87 | 0.42 | 0.15 | 0.22 | 0.12 | 4638.95 |
| **K-Means** | 59.59 | 0.10 | 0.63 | 0.16 | 0.09 | 26276.74 |
| **Otsu** | 46.30 | 0.09 | 0.81 | 0.16 | 0.09 | 34920.57 |
| **Canny** | 93.87 | 0.83 | 0.04 | 0.08 | 0.04 | 3990.10 |
| **Watershed** | 69.73 | 0.00 | 0.01 | 0.00 | 0.00 | 19684.45 |

### Key Observations

- **GrabCut** achieves the highest IoU (0.27) and the best overall balance between Precision and Recall, making it the top performer for this dataset.
- **Canny** has the highest Accuracy (93.87%) and Precision (0.83) but near-zero Recall (0.04) — it detects almost none of the actual foreground, making it unsuitable as a standalone method.
- **Adaptive Thresholding** ranks second in IoU (0.12) with high Accuracy but low Recall, missing large portions of the foreground.
- **Otsu** and **K-Means** achieve high Recall (0.81 and 0.63) but very low Precision — they over-segment and include excessive background noise.
- **Watershed** performs poorly on this dataset (IoU ≈ 0.00), likely because the product's uniform white color makes marker-based region separation unreliable.

---

## Requirements

Python 3.8+

```bash
pip install opencv-python numpy matplotlib pandas
```

| Library | Purpose |
|---------|---------|
| `opencv-python` | Image processing & all segmentation algorithms |
| `numpy` | Array operations |
| `matplotlib` | Visualization & comparison plots |
| `pandas` | Metrics table & summary report |

---

## Usage

All scripts must be run from inside the **`Demo/`** directory:

```bash
cd Demo/
```

**Step 1 — Preprocess images**
```bash
python 01_preprocessing.py
```
Applies bilateral filtering and grayscale conversion. Outputs saved to `denoised_images/`.

**Step 2 — (Optional) Verify mask alignment**
```bash
python 02_mask_alignment_check.py
```
Visually confirms ground-truth mask alignment on the denoised images.

**Step 3 — Run segmentation algorithms**
```bash
python 03_segmentation_otsu_thrs.py
python 03_segmentation_adaptive_thrs.py
python 03_segmentation_canny_edge.py
python 03_segmentation_grabcut.py
python 03_segmentation_kmeans.py
python 03_segmentation_watershed.py
```
Each script generates 6 binary mask PNGs (one per image) directly in `Demo/`.

**Step 4 — Generate performance report**
```bash
python 04_final_performance_report.py
```
Prints per-photo and overall mean metrics to the terminal and displays a bar chart comparing all algorithms.

**Step 5 — Background removal**
```bash
python 05_bg_removal.py
```
Applies every binary mask to the original color images. Outputs 36 PNG files to `final_bg_removed_results/`.
