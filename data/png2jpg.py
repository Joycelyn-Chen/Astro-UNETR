import os
import argparse
from PIL import Image

# python png2jpg.py --png_dir /home/joy0921/Desktop/Dataset/MHD-3DIS/masks/ --jpg_dir /home/joy0921/Desktop/Dataset/MHD-3DIS/masks-jpg/

parser = argparse.ArgumentParser(description="Converting PNG masks into JPG masks for SAM2 input.")
parser.add_argument("--png_dir", type=str, required=True, help="Input directory containing PNG images")
parser.add_argument("--jpg_dir", type=str, required=True, help="Output directory for JPG images")

def convert_png_to_jpg(source_root, destination_root):
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)

    for root, _, files in os.walk(source_root):
        rel_path = os.path.relpath(root, source_root)
        dest_dir = os.path.join(destination_root, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.png'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(dest_dir, file[:-4] + '.jpg')

                with Image.open(source_file) as img:
                    # Convert image to RGB mode if it has transparency
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    img.save(destination_file, "JPEG", quality=95)

        print(f"Conversion completed. JPG files saved in: {dest_dir}")

if __name__ == "__main__":
    args = parser.parse_args()
    convert_png_to_jpg(args.png_dir, args.jpg_dir)

