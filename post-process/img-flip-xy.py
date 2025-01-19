import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Flip the images horizontally.")
parser.add_argument("--input_root", default="./Dataset", type=str, help="input directory")

# python img-flip-xy.py --input_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20617

def flip_images_horizontally(root_dir):
    """
    Loop through the folder tree and flip all image slices horizontally, saving them back.

    Args:
        root_dir (str): Path to the root directory containing image files.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                try:
                    # Open the image
                    with Image.open(file_path) as img:
                        # Flip the image horizontally
                        # flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        flipped_img = img.transpose(Image.TRANSPOSE)
                        
                        # Save the flipped image back to the same path
                        flipped_img.save(file_path)
                        print(f"Flipped and saved: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Call the function to flip images horizontally
    flip_images_horizontally(args.input_root)
