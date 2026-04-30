import cv2
import numpy as np
import os

# 1. Load Preprocessed Data (Using the .png outputs from the pipeline)
# We read the denoised image and the manually created binary mask 
img_path = 'denoised_images/5_denoised.png' 
mask_path = 'masked_images/5_masked.png'

# Safety check for file existence
if not os.path.exists(img_path) or not os.path.exists(mask_path):
    print("Error: Files not found! Please check the directory paths.")
else:
    # Read the denoised image (Load as BGR to allow colored drawings)
    img = cv2.imread(img_path)
    # Read the manual ground truth mask in grayscale mode
    mask = cv2.imread(mask_path, 0)

    # 2. Dimension Consistency Check [cite: 20]
    # Ensure the image and mask have the same resolution for 100% pixel-wise registration
    if img.shape[:2] != mask.shape:
        print(f"WARNING: Dimensions mismatch! Image: {img.shape[:2]}, Mask: {mask.shape}")
        # Resize mask to match the image dimensions if necessary
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 3. Boundary Verification (Red Contours) [cite: 88]
    # Identify the external boundary of the manual mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    # Draw the red contours onto the denoised image to verify edge alignment [cite: 80]
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # 4. Transparent Overlay Creation (Alpha Blending) 
    # Convert grayscale mask to BGR to apply color
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Change white pixels to green for visualization purposes
    mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
    # Blend the original image with the green mask (70% original, 30% mask)
    blended = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

    # Visualizing the verification results
    cv2.imshow('Boundary Verification (Denoised Image + Red Mask Outlines)', overlay)
    cv2.imshow('Alpha Blending (Denoised Image + Green Mask Overlay)', blended)

    print("Alignment verification complete. Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()