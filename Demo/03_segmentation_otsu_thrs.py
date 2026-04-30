import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define photo numbers for batch processing
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # INPUT: Read denoised images from the preprocessing stage
    input_path = f"denoised_images/{num}_denoised.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Please run preprocessing first.")
        continue
        
    # Read the image in grayscale mode
    denoised = cv2.imread(input_path, 0)

    # 1. IMAGE SEGMENTATION (Otsu's Thresholding)
    # Automatically calculates the optimal threshold value by analyzing the image histogram
    # We use cv2.THRESH_BINARY + cv2.THRESH_OTSU to perform global thresholding
    ret, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. MORPHOLOGICAL OPERATIONS (Post-processing)
    # Define a 5x5 kernel for refining the binary mask
    kernel = np.ones((5,5), np.uint8)
    
    # Closing: Fills gaps and holes inside the product caused by text or reflections
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Opening: Removes small background noise and light artifacts
    final_mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # 3. PHYSICAL SAVING
    # Save the resulting Otsu mask for the quantitative performance comparison
    save_name = f"{num}_algo_result_otsu.png"
    cv2.imwrite(save_name, final_mask)
    print(f"Success: {save_name} saved. Optimal Threshold Found: {ret}")

    # 4. VISUALIZATION (Comparison Plots)
    plt.figure(figsize=(15, 5))
    
    # Display the Denoised Input
    plt.subplot(1, 3, 1)
    plt.title(f'Input {num} (Denoised)')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    # Display Raw Otsu Thresholding Result (with the calculated threshold value)
    plt.subplot(1, 3, 2)
    plt.title(f'Otsu (Threshold: {int(ret)})')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    # Display Final Refined Mask after Morphological Cleaning
    plt.subplot(1, 3, 3)
    plt.title('Morphological Cleaning')
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')
    
    plt.show()