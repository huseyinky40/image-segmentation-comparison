import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create the output directory for processed images if it doesn't exist
output_folder = "denoised_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the dataset range (6 images of Jordan Toothpaste)
photo_numbers = [1, 2, 3, 4, 5, 6]

for num in photo_numbers:
    # Specify the relative path for input images
    input_path = f"original_images/{num}.jpeg"
    
    # Load the original BGR image using OpenCV
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: {input_path} not found!")
        continue
        
    # Convert BGR to RGB for correct Matplotlib visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the color image to Grayscale to simplify the data for filtering
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter to reduce Gaussian noise
    # d=15: Diameter of each pixel neighborhood
    # sigmaColor=75: Filter sigma in the color space (preserves edges)
    # sigmaSpace=75: Filter sigma in the coordinate space (cleans spatial noise)
    denoised = cv2.bilateralFilter(gray, d=15, sigmaColor=75, sigmaSpace=75)

    # Physically save the denoised result as a PNG to maintain quality
    save_path = f"{output_folder}/{num}_denoised.png"
    cv2.imwrite(save_path, denoised)
    print(f"Success: {save_path} saved.")

    # Generate a side-by-side comparison for the demonstration
    plt.figure(figsize=(15, 5))
    
    # Show Original Color Image
    plt.subplot(1, 3, 1)
    plt.title(f'Original {num}')
    plt.imshow(img_rgb)
    plt.axis('off')

    # Show Grayscale Version
    plt.subplot(1, 3, 2)
    plt.title('Grayscale Conversion')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    # Show Final Denoised Result (Bilateral Output)
    plt.subplot(1, 3, 3)
    plt.title('Bilateral Filtering (Denoised)')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')
    
    # Display the plots and wait for the user to close the window
    plt.show()