import cv2
import numpy as np
import pandas as pd
import os

# --- 1. BACKGROUND REMOVAL FUNCTION ---
def remove_bg_with_mask(original_img, mask):
    """
    Applies a binary mask to the original image to extract the foreground object.
    """
    # Resize mask to match the original image dimensions exactly
    mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
    
    # Normalize mask to range [0.0, 1.0] for pixel-wise multiplication
    mask_normalized = mask_resized.astype(float) / 255.0
    
    # Stack the single-channel mask into 3 channels (BGR) to match the color image
    mask_3d = cv2.merge([mask_normalized, mask_normalized, mask_normalized])
    
    # Multiply original image by the mask (Blackens the background pixels)
    result = (original_img * mask_3d).astype(np.uint8)
    
    return result

# --- 2. OUTPUT DIRECTORY SETUP ---
output_dir = "final_bg_removed_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 3. QUANTITATIVE METRICS FUNCTION ---
def calculate_pixelwise_metrics(gt, algo):
    """
    Compares the predicted mask with Ground Truth to generate performance scores.
    """
    algo = cv2.resize(algo, (gt.shape[1], gt.shape[0]))
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    _, algo = cv2.threshold(algo, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate Pixel counts: TP, FP, FN, TN
    tp = np.sum((algo == 255) & (gt == 255))
    fp = np.sum((algo == 255) & (gt == 0))
    fn = np.sum((algo == 0) & (gt == 255))
    tn = np.sum((algo == 0) & (gt == 0))
    
    # Calculate Score Ratios
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mse = np.mean((gt.astype("float") - algo.astype("float")) ** 2)
    
    return {
        "Accuracy": round(accuracy * 100, 2), 
        "Precision": round(precision, 2), 
        "Recall": round(recall, 2), 
        "F1-Score": round(f1_score, 2), 
        "IoU": round(iou, 2), 
        "MSE": round(mse, 2)
    }

# --- 4. BATCH PROCESSING PIPELINE ---
photo_numbers = [1, 2, 3, 4, 5, 6]
methods = ["otsu", "adaptive", "canny", "grabcut", "kmeans", "watershed"]
all_results = []

print(f"Starting Process... Creating 36 final results in '{output_dir}'.")

for num in photo_numbers:
    gt_path = f"masked_images/{num}_masked.png"
    orig_path = f"original_images/{num}.jpeg"
    
    if not os.path.exists(gt_path) or not os.path.exists(orig_path):
        print(f"Warning: Photo {num} or its Ground Truth mask is missing!")
        continue
    
    # Read core files
    gt_img = cv2.imread(gt_path, 0)
    original_img = cv2.imread(orig_path)
    
    for method in methods:
        algo_path = f"{num}_algo_result_{method}.png"
        
        if os.path.exists(algo_path):
            algo_img = cv2.imread(algo_path, 0)
            
            # 1. Calculate Metrics for current model
            metrics = calculate_pixelwise_metrics(gt_img, algo_img)
            metrics["Model"] = method.capitalize()
            metrics["Photo"] = num
            all_results.append(metrics)
            
            # 2. GENERATE FINAL BACKGROUND-REMOVED IMAGE
            final_product = remove_bg_with_mask(original_img, algo_img)
            
            # Physical Saving of the final e-commerce ready image
            save_name = f"{output_dir}/{num}_final_{method}.png"
            cv2.imwrite(save_name, final_product)
            
    print(f"Processed all methods for Photo {num}.")