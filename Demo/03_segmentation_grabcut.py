import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the dataset range
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # INPUT: Loading the preprocessed images
    input_path = f"denoised_images/{num}_denoised.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        continue
        
    # Read the grayscale image and convert to BGR for GrabCut compatibility
    img_gray = cv2.imread(input_path, 0)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. GRABCUT INITIALIZATION
    # Creating an empty mask and internal models for foreground/background distribution
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64) # Background model
    fgdModel = np.zeros((1, 65), np.float64) # Foreground model

    # 2. DEFINE REGION OF INTEREST (ROI)
    # Manual Bounding Box: (x, y, width, height)
    # This rectangle helps the algorithm identify the initial foreground pixels
    rect = (550, 300, 450, 1600)

    # 3. RUN ALGORITHM (Iterative Refinement)
    # 5 iterations of the GrabCut algorithm to refine the segmentation
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 4. MASK POST-PROCESSING
    # GrabCut labels pixels as: 0 (BG), 1 (FG), 2 (Probable BG), 3 (Probable FG)
    # We consolidate labels 0 and 2 into 'Background' (0) and 1 and 3 into 'Foreground' (1)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    final_mask = mask2 * 255

    # 5. VISUAL EXTRACTION (Display Only)
    # Using bitwise_and to isolate the product from the original background
    result_extraction = cv2.bitwise_and(img_rgb, img_rgb, mask=mask2)

    # 6. PHYSICAL SAVING
    # Saving the binary mask for the quantitative performance table
    save_name = f"{num}_algo_result_grabcut.png"
    cv2.imwrite(save_name, final_mask)
    print(f"Success: {save_name} saved.")

    # 7. VISUALIZATION (3-Step Display)
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original image with the Red ROI box
    plt.subplot(1, 3, 1)
    plt.title(f'1. Input & ROI ({num})')
    plt.imshow(img_rgb)
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], 
                                     edgecolor='red', facecolor='none', lw=2))
    plt.axis('off')

    # Plot 2: Resulting Binary Mask
    plt.subplot(1, 3, 2)
    plt.title('2. GrabCut Binary Mask')
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')

    # Plot 3: Final Product Extraction (Background removed)
    plt.subplot(1, 3, 3)
    plt.title('3. Final Product Extraction')
    plt.imshow(result_extraction)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()