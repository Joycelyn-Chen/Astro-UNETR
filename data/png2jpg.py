import os
import shutil
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Converting png masks into jpg masks for SAM2 input.")
parser.add_argument("--png_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--jpg_dir", default="./Dataset", type=str, help="outputput data directory")

def rename_png_to_jpg(source_root, destination_root):
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)

    for root, dirs, files in os.walk(source_root):
        # Calculate the destination directory path
        rel_path = os.path.relpath(root, source_root)
        dest_dir = os.path.join(destination_root, rel_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for file in files:
            if file.endswith('.png'):
                # Source file path
                source_file = os.path.join(root, file)
                # Destination file path with changed extension
                destination_file = os.path.join(dest_dir, file[:-4] + '.jpg')

                # Open the image and convert to an array
                with Image.open(source_file) as img:
                    # Multiply image data by 255
                    img_array = np.array(img) * 255
                    # Ensure no overflow by capping values
                    img_array = np.clip(img_array, 0, 255).astype('uint8')
                    # Convert array back to an image
                    enhanced_img = Image.fromarray(img_array)
                    # Save as JPEG
                    enhanced_img.save(destination_file, "JPEG", quality=95)

if __name__ == "__main__":
    args = parser.parse_args()

    rename_png_to_jpg(args.png_dir, args.jpg_dir)
