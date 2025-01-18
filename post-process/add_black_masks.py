import os
from PIL import Image
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Add black masks for those missing z coordinates.")
parser.add_argument("--mask_root", default="./Dataset", type=str, help="input directory")


def add_missing_masks(mask_root):
    """
    Add black masks for missing z slices in each timestamp directory.

    Args:
        mask_root (str): Path to the root directory containing timestamp subdirectories.
    """
    # Define the size of the black mask image
    mask_size = (256, 256, 3)  # 256x256 with 3 color channels (RGB)

    # Loop through all timestamp folders
    for timestamp_dir in os.listdir(mask_root):
        timestamp_path = os.path.join(mask_root, timestamp_dir)

        if not os.path.isdir(timestamp_path):
            continue  # Skip files, process directories only

        # Collect existing slice filenames
        existing_files = set()
        for filename in os.listdir(timestamp_path):
            if filename.endswith(".png"):
                try:
                    z_index = int(filename.split(".png")[0])
                    existing_files.add(z_index)
                except ValueError:
                    print(f"Invalid file name: {filename}")

        # Check for missing slices and add black masks
        for z_index in range(256):
            if z_index not in existing_files:
                black_image = Image.new("RGB", (256, 256), (0, 0, 0))  # Create a black image
                black_image_path = os.path.join(timestamp_path, f"{z_index}.png")
                black_image.save(black_image_path)
                print(f"Added missing black mask: {black_image_path}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Call the function to add missing black masks
    add_missing_masks(args.mask_root)

# python add_black_masks.py --mask_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20617