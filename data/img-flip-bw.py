import os
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description="Flipping grayscale images to fit Swin-UNETR input usage")
parser.add_argument("--input_root", default="./Dataset", type=str, help="input directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")



def process_images(input_root, output_root):
    # Ensure the output root directory exists
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Traverse through all subdirectories and images
    for subdir, _, files in os.walk(input_root):
        # Determine the relative path of the current subdirectory
        relative_path = os.path.relpath(subdir, input_root)

        print(f"Processing timestamp: {subdir}")
        # Create the corresponding subdirectory in the output root
        output_subdir = os.path.join(output_root, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Process each file in the current directory
        for file in files:
            file_path = os.path.join(subdir, file)

            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Ensure it's a grayscale image
                    if img.mode != 'L':
                        img = img.convert('L')

                    # Flip the black and white scale
                    flipped_img = Image.eval(img, lambda pixel: (255 - pixel) % 255)

                    # Save the flipped image in the corresponding output subdirectory
                    output_path = os.path.join(output_subdir, file)
                    flipped_img.save(output_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Process the images
    process_images(args.input_root, args.output_root)

    print(f"Done. Flipped images stored at {args.output_root}")

# python img-flip-bw.py --input_root /home/joycelyn/Desktop/Dataset/MHD-3DIS/imgs --output_root /home/joycelyn/Desktop/Dataset/MHD-3DIS/imgs-bw-flipped