import os
import cv2
import numpy as np
import argparse

# This code generates binary masks for all low density area.
# Like a semantic segmentation of the superbubble regions

# Usage: python gen_whole_mask.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS

parser = argparse.ArgumentParser(description="Generating whole cube binary masks for all the input data as input to Swin-UNETR")
parser.add_argument("--data_dir", default="~/Desktop/Dataset/MHD-3DIS", type=str, help="input data directory")

def create_masks(data_dir):
    # Define paths for input images and output masks
    imgs_dir = os.path.join(data_dir, "imgs")
    masks_dir = os.path.join(data_dir, "masks")

    # Ensure the output masks directory exists
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    # Iterate over each subfolder in the imgs/ directory
    for subfolder in os.listdir(imgs_dir):
        subfolder_path = os.path.join(imgs_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Create corresponding subfolder in masks/
            masks_subfolder_path = os.path.join(masks_dir, subfolder)
            os.makedirs(masks_subfolder_path, exist_ok=True)

            # Process each image slice in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg"):
                    # Read the image in grayscale
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Apply Otsu's thresholding
                    _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Save the binary mask as a .png file in the corresponding masks/ folder
                    mask_filename = os.path.splitext(filename)[0] + ".png"
                    mask_path = os.path.join(masks_subfolder_path, mask_filename)
                    cv2.imwrite(mask_path, binary_mask)

                    print(f"Processed and saved mask: {mask_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    
    create_masks(args.data_dir)
    print("All masks have been successfully created and saved.")
