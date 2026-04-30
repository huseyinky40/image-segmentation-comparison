import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the dataset range
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # INPUT: Read the denoised images from the preprocessing stage
    input_path = f"denoised_images/{num}_denoised.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Run preprocessing first.")
        continue
        
    # Read the image in grayscale
    denoised = cv2.imread(input_path, 0)

    # 1. CANNY EDGE DETECTION
    # Low Threshold = 100, High Threshold = 200
    # Used to identify structural boundaries of the object
    edges = cv2.Canny(denoised, 100, 200)

    # 2. MORPHOLOGICAL DILATION
    # Bridges small gaps between fragmented edge pixels to create a closed loop
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 3. CONTOUR ANALYSIS & FILLING
    # Create an empty black mask to draw our final segmentation
    mask = np.zeros(dilated_edges.shape, dtype=np.uint8)
    
    # Identify external contours from the dilated edge map
    cnts, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts:
        # Select the largest contour to filter out background noise
        c = max(cnts, key=cv2.contourArea)
        # Verify the area size to ensure we are capturing a significant object
        if cv2.contourArea(c) > 500: 
            # Fill the identified contour to transform edges into a solid mask
            cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

    # 4. PHYSICAL SAVING
    # Save the final binary mask for quantitative evaluation
    save_name = f"{num}_algo_result_canny.png"
    cv2.imwrite(save_name, mask)
    print(f"Success: {save_name} (Final Mask) saved.")

    # 5. VISUALIZATION
    # Display the three-step pipeline flow for the demonstration
    plt.figure(figsize=(15, 5))
    
    # Show Initial Canny Edges
    plt.subplot(1, 3, 1)
    plt.title(f'Step 1: Canny Edges ({num})')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    # Show Dilated Boundaries
    plt.subplot(1, 3, 2)
    plt.title('Step 2: Dilated Edges')
    plt.imshow(dilated_edges, cmap='gray')
    plt.axis('off')
    
    # Show the Resulting Solid Mask
    plt.subplot(1, 3, 3)
    plt.title('Step 3: Final Filled Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()