import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define photo numbers for batch processing (Jordan Toothpaste Dataset)
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # INPUT: Read denoised images from the previous preprocessing stage
    input_path = f"denoised_images/{num}_denoised.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        continue
        
    # Read grayscale image and convert to BGR for color-based clustering analysis
    img_gray = cv2.imread(input_path, 0)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # DATA PREPARATION: Reshape the 2D image into a 1D array of RGB pixels for K-Means
    data = img_color.reshape((-1, 3)).astype(np.float32)

    # STEP 1: K-MEANS CONFIGURATION
    # Define termination criteria: 10 iterations or 1.0 epsilon change
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set K=3 to identify three main regions: Product (White), Background (Gray), Shadows (Dark)
    K = 3
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # STEP 2: AUTOMATIC CLUSTER SELECTION (Engineering Logic)
    # K-Means gives random labels (0, 1, 2). We must identify which label is the product.
    # We calculate the Luminance for each cluster center; the brightest center is the white toothpaste.
    brightness = [0.299*c[2] + 0.587*c[1] + 0.114*c[0] for c in center]
    target_label = np.argmax(brightness)

    # STEP 3: MASK GENERATION
    # Reshape the flat labels array back into the original 2D image dimensions
    label_image = label.reshape(img_color.shape[:2])
    # Create a binary mask: Target cluster becomes white (255), everything else becomes black (0)
    mask = (label_image == target_label).astype(np.uint8) * 255
    
    # STEP 4: POST-PROCESSING (Morphological Refinement)
    # Use a 5x5 kernel and Opening operation to remove small background outliers (artifacts)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # STEP 5: FINAL PRODUCT EXTRACTION
    # Apply bitwise-AND to isolate the color product using our generated K-Means mask
    result_extraction = cv2.bitwise_and(img_color, img_color, mask=mask)

    # PHYSICAL SAVING: Save the mask for quantitative evaluation metrics
    save_name = f"{num}_algo_result_kmeans.png"
    cv2.imwrite(save_name, mask)
    print(f"Success: {save_name} saved.")

    # STEP 6: VISUALIZATION (Pipeline Display)
    # Reconstruct the image using only the cluster centroids (Posterization effect)
    center_uint8 = np.uint8(center)
    res = center_uint8[label.flatten()]
    clustered_img = res.reshape((img_color.shape))

    plt.figure(figsize=(15, 5))
    
    # Plot 1: Show the K clusters (See how the image is partitioned into 3 colors)
    plt.subplot(1, 3, 1)
    plt.title(f'1. K-Means Clusters ({num})')
    plt.imshow(cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Plot 2: Show the binary mask selected by the Luminance logic
    plt.subplot(1, 3, 2)
    plt.title('2. Automated K-Means Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # Plot 3: Show the final background-removed product
    plt.subplot(1, 3, 3)
    plt.title('3. Final Extraction Result')
    plt.imshow(cv2.cvtColor(result_extraction, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()