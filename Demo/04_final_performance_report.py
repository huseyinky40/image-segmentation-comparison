import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

def calculate_pixelwise_metrics(gt, algo):
    """
    Calculates standard segmentation metrics by comparing the algorithm output 
    with the Ground Truth (Manual Mask).
    """
    # 1. PREPARATION: Resize to match dimensions and convert to binary (0 or 255)
    algo = cv2.resize(algo, (gt.shape[1], gt.shape[0]))
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    _, algo = cv2.threshold(algo, 127, 255, cv2.THRESH_BINARY)

    # 2. PIXEL-WISE COMPARISON: Calculate Confusion Matrix components
    # Positive = White pixels (Object), Negative = Black pixels (Background)
    tp = np.sum((algo == 255) & (gt == 255)) # True Positive
    fp = np.sum((algo == 255) & (gt == 0))   # False Positive
    fn = np.sum((algo == 0) & (gt == 255))   # False Negative
    tn = np.sum((algo == 0) & (gt == 0))     # True Negative

    # 3. CORE METRICS CALCULATION
    # Accuracy: Percentage of correctly classified pixels (Object + BG)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision: Ability to avoid labeling background as foreground
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: Ability to capture the entire foreground object
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # IoU (Intersection over Union): Overlap ratio between Predicted and Ground Truth
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0

    # F1-Score (Dice Coefficient): Harmonic mean of Precision and Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # MSE (Mean Squared Error): Cumulative squared error at pixel level
    mse = np.mean((gt.astype("float") - algo.astype("float")) ** 2)

    return {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1_score, 2),
        "IoU": round(iou, 2),
        "MSE": round(mse, 2)
    }

# --- DATASET CONFIGURATION ---
photo_numbers = [1, 2, 3, 4, 5, 6]
methods = ["otsu", "adaptive", "canny", "grabcut", "kmeans", "watershed"]
all_results = []

# --- BATCH PROCESSING ---
for num in photo_numbers:
    gt_path = f"masked_images/{num}_masked.png"
    if not os.path.exists(gt_path): 
        print(f"Warning: Ground Truth not found -> {gt_path}")
        continue
    
    gt_img = cv2.imread(gt_path, 0)
    current_photo_data = []
    
    for method in methods:
        algo_path = f"{num}_algo_result_{method}.png"
        if os.path.exists(algo_path):
            algo_img = cv2.imread(algo_path, 0)
            metrics = calculate_pixelwise_metrics(gt_img, algo_img)
            metrics["Model"] = method.capitalize()
            metrics["Photo"] = num
            current_photo_data.append(metrics)
            all_results.append(metrics)
    
    # Display individual photo results in the terminal
    if current_photo_data:
        df_photo = pd.DataFrame(current_photo_data)
        print(f"\n--- PHOTO {num} PIXEL-WISE ANALYSIS ---")
        print(df_photo[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "IoU", "MSE"]].to_string(index=False))

# --- SUMMARY TABLE AND DATA VISUALIZATION ---
if all_results:
    df_all = pd.DataFrame(all_results)
    
    # Calculate Average Performance for each model across all photos
    summary = df_all.groupby("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "IoU", "MSE"]].mean().reset_index()
    
    # Sort models by IoU performance (Descending)
    summary = summary.sort_values(by="IoU", ascending=False)
    
    print("\n" + "="*80)
    print("--- OVERALL MEAN PERFORMANCE SUMMARY (Sorted by IoU) ---")
    print("="*80)
    print(summary.round(2).to_string(index=False))

    # --- PERFORMANCE CHART GENERATION ---
    plot_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "IoU"]
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bar_width = 0.15
    index = np.arange(len(summary["Model"]))
    
    for i, metric in enumerate(plot_metrics):
        if metric == "Accuracy":
            # Normalize Accuracy to 0-1 scale for chart consistency
            values = summary[metric] / 100
            label = "Accuracy (Normalized)"
        else:
            values = summary[metric]
            label = metric
            
        ax.bar(index + (i * bar_width), values, bar_width, label=label)

    # Chart Styling and Formatting
    ax.set_xlabel('Segmentation Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (0.0 - 1.0)', fontweight='bold', fontsize=12)
    ax.set_title('Overall Performance Metrics Comparison', fontweight='bold', fontsize=16)
    
    ax.set_xticks(index + bar_width * 2)
    ax.set_xticklabels(summary["Model"], rotation=0)
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim(0, 1.1)
    
    # Professional legend placement
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=len(plot_metrics), fancybox=True, shadow=True, fontsize=10)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) 
    
    plt.show()