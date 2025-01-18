import os
import shutil
from pathlib import Path
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description="Rescale mask images from 1000 pixel resolution to 256")
parser.add_argument("--input_root", default="./Dataset", type=str, help="input directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")



def rescale_and_reorganize_masks(root_dir, output_dir):
    """
    Rescale and reorganize mask slices from the original directory structure to the new one.

    Args:
        root_dir (str): Path to the root directory containing the original timestamp subdirectories.
        output_dir (str): Path to the new directory where reorganized slices will be stored.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Loop through all subdirectories under the root directory
    for timestamp_dir in os.listdir(root_dir):
        timestamp_path = os.path.join(root_dir, timestamp_dir)

        if not os.path.isdir(timestamp_path):
            continue  # Skip files, process directories only

        # Create corresponding directory in the new location
        new_timestamp_path = os.path.join(output_dir, timestamp_dir)
        os.makedirs(new_timestamp_path, exist_ok=True)

        # Process each image slice in the timestamp directory
        for filename in os.listdir(timestamp_path):
            if not filename.endswith(".png"):
                continue  # Skip non-PNG files

            # Parse the z-coordinate from the filename
            try:
                z_coordinate = int(filename.split("z")[-1].split(".png")[0])
            except ValueError:
                print(f"Skipping invalid file: {filename}")
                continue

            # Rescale the z-coordinate to 256 scale
            rescaled_z = z_coordinate * 256 // 1000

            # Rename the image file with the new z-coordinate
            new_filename = f"{rescaled_z}.png"
            src_file = os.path.join(timestamp_path, filename)
            dest_file = os.path.join(new_timestamp_path, new_filename)

            # Copy the file to the new location
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

import os
from PIL import Image
from pathlib import Path

def resize_images(root_dir):
    """
    Loop through the folder tree structure and resize images to (256, 256, 3) if needed.

    Args:
        root_dir (str): Path to the root directory containing image files.
    """
    target_size = (256, 256)  # Target size for resizing

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                try:
                    # Open the image
                    with Image.open(file_path) as img:
                        # Check the size
                        if img.size != target_size:
                            # Resize the image proportionally
                            resized_img = img.resize(target_size, Image.LANCZOS)
                            resized_img.save(file_path)
                            print(f"Resized and saved: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Define the root and output directories
    args = parser.parse_args()

    # Call the function to reorganize the mask slices
    rescale_and_reorganize_masks(args.input_root, args.output_root)
    resize_images(args.output_root)

# python rescale_mask_10002256.py --input_root /UBC-O/joy0921/Desktop/Dataset/Isolated_case/SN_20617 --output_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20617