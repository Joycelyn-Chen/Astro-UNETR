import os
import sys
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Removing incorrect instance mask generation from sam2 output.")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--start_z", type=int, required=True, help="lower limit for z-coordinate range")
parser.add_argument("--end_z", type=int, required=True, help="upper limit for z-coordinate range")

# python remove-unwanted-masks.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/SB_tracks/230/410 --start_z 143 --end_z 253

def process_images(data_dir, start_z, end_z):
    """
    Process images in the given directory.

    Parameters:
    data_dir (str): The root directory containing the images.
    start_z (int): The starting z-coordinate.
    end_z (int): The ending z-coordinate.

    """
    # Convert start and end z-coordinates to integers
    start_z = int(start_z)
    end_z = int(end_z)

    # Ensure the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        sys.exit(1)

    # Get a sorted list of all image files
    image_files = sorted(os.listdir(data_dir), key=lambda x: int(os.path.splitext(x)[0]))

    for image_file in image_files:
        # Get the z-coordinate from the filename
        try:
            z_coord = int(os.path.splitext(image_file)[0])
        except ValueError:
            print(f"Skipping invalid file: {image_file}")
            continue

        # Check if the z-coordinate is outside the range
        if z_coord < start_z or z_coord > end_z:
            # Replace with a black image
            black_image = np.zeros((256, 256), dtype=np.uint8)
            black_image = Image.fromarray(black_image)

            # Save the black image with the same name
            black_image.save(os.path.join(data_dir, image_file))

            print(f"Replaced {image_file} with a black image.")
        else:
            print(f"Kept {image_file}.")

if __name__ == "__main__":
    args = parser.parse_args()

    process_images(args.data_dir, args.start_z, args.end_z)
