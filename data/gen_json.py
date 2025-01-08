import os
import json
import argparse


parser = argparse.ArgumentParser(description="Generating json file linking all the input data as input to Swin-UNETR")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--output_file", default="test.json", type=str, help="output filename")

def generate_mhd_json(data_dir, output_file):
    data_structure = {"training": []}

    # Define paths for imgs and masks directories
    imgs_dir = os.path.join(data_dir, "imgs")
    masks_dir = os.path.join(data_dir, "masks")

    # Traverse through the imgs directory to process files
    for img_file in sorted(os.listdir(imgs_dir)):
        if img_file.endswith(".nii.gz"):
            timestamp = os.path.splitext(os.path.splitext(img_file)[0])[0]  # Extract timestamp (e.g., '380' from '380.nii.gz')

            # Construct relative paths for image and corresponding label
            image_path = os.path.join("imgs", img_file)
            label_path = os.path.join("masks", f"{timestamp}.seg.nii.gz")

            # Ensure the label file exists in the masks directory
            if os.path.exists(os.path.join(masks_dir, f"{timestamp}.seg.nii.gz")):
                data_structure["training"].append({
                    "fold": 1,
                    "image": [image_path] * 4,  # Repeat the image path 4 times
                    "label": label_path
                })
            else:
                print(f"Warning: Corresponding mask for {img_file} not found in masks directory.")

    # Write the data structure to the output JSON file
    with open(output_file, "w") as json_file:
        json.dump(data_structure, json_file, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()

    generate_mhd_json(args.data_dir, os.path.join(args.data_dir, args.output_file))
    print(f"Generated {args.output_file} successfully.")

# python gen_json.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_file "MHD-NII.json"