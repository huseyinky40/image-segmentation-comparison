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
        print(f"Error: {input_path} not found!")
        continue
        
    # Read grayscale image and convert to BGR for Watershed compatibility
    denoised_gray = cv2.imread(input_path, 0)
    img_color = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)

    # 1. INITIAL THRESHOLDING & NOISE REMOVAL
    # Use Otsu's to get a rough binary image
    ret, thresh = cv2.threshold(denoised_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. DEFINING BACKGROUND AND FOREGROUND AREAS
    # Sure background area (Dilation expands the black regions)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance Transform: Finds the center of the objects (Foremost points)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Sure foreground area (Thresholding the distance map)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region: The transition area between sure background and foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 3. MARKER LABELING
    # Label sure foreground regions with positive integers
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # 4. WATERSHED ALGORITHM
    # The algorithm 'floods' the image starting from the markers
    markers = cv2.watershed(img_color, markers)
    
    # Generate the Final Binary Mask (Regions with label > 1 are the product)
    final_mask = np.zeros(denoised_gray.shape, dtype=np.uint8)
    final_mask[markers > 1] = 255

    # 5. VISUAL EXTRACTION (For display purposes)
    # Apply bitwise-AND to isolate the product using the watershed mask
    result_extraction = cv2.bitwise_and(img_color, img_color, mask=final_mask)

    # PHYSICAL SAVING: Store the mask for quantitative evaluation
    save_name = f"{num}_algo_result_watershed.png"
    cv2.imwrite(save_name, final_mask)
    print(f"Success: {save_name} saved.")

    # 6. VISUALIZATION (4-Stage Pipeline Display)
    plt.figure(figsize=(20, 5))
    
    # Plot 1: Distance Transform (Topographical peaks)
    plt.subplot(1, 4, 1)
    plt.title(f'1. Distance Trans. ({num})')
    plt.imshow(dist_transform, cmap='jet')
    plt.axis('off')
    
    # Plot 2: Watershed Markers (Catchment basins)
    plt.subplot(1, 4, 2)
    plt.title('2. Watershed Markers')
    plt.imshow(markers, cmap='tab20b')
    plt.axis('off')
    
    # Plot 3: Resulting Binary Mask
    plt.subplot(1, 4, 3)
    plt.title('3. Final Watershed Mask')
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')

    # Plot 4: Final Extraction (BGR to RGB for Matplotlib)
    plt.subplot(1, 4, 4)
    plt.title('4. Final Product Extraction')
    plt.imshow(cv2.cvtColor(result_extraction, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()