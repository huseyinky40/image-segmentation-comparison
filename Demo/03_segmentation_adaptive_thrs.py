import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the dataset range
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # INPUT: Read the denoised images generated from the preprocessing stage
    input_path = f"denoised_images/{num}_denoised.png"
    
    # Check if the file exists to prevent runtime errors
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Run preprocessing first.")
        continue
        
    # Read the image in grayscale mode for thresholding operations
    denoised = cv2.imread(input_path, 0)

    # 1. IMAGE SEGMENTATION (Adaptive Thresholding)
    # We use ADAPTIVE_THRESH_GAUSSIAN_C to overcome non-uniform lighting and shadows
    # Block Size = 11: Size of the local neighborhood
    # C = 2: Constant subtracted from the mean
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. MORPHOLOGICAL OPERATIONS (Post-processing)
    # Define a 3x3 kernel for morphological refinement
    kernel = np.ones((3,3), np.uint8)
    
    # Closing: Used to fill small gaps and holes inside the object body
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) 
    
    # Opening: Used to remove background 'salt-and-pepper' noise pixels
    final_mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. PHYSICAL SAVING
    # Save the processed binary mask to the main directory for evaluation
    save_name = f"{num}_algo_result_adaptive.png"
    cv2.imwrite(save_name, final_mask)
    print(f"Success: {save_name} saved.")

    # 4. VISUALIZATION
    # Display the transition from raw input to the final cleaned mask
    plt.figure(figsize=(15, 5))
    
    # Display Denoised Input
    plt.subplot(1, 3, 1)
    plt.title(f'Input {num} (Denoised)')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    # Display Raw Adaptive Result
    plt.subplot(1, 3, 2)
    plt.title('Adaptive Thresholding')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    # Display Final Refined Result
    plt.subplot(1, 3, 3)
    plt.title('Morphological Cleaning')
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')
    
    # Block execution until the plot window is closed
    plt.show()