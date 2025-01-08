import os
import numpy as np
import nibabel as nib
from PIL import Image

import argparse


parser = argparse.ArgumentParser(description="Converting grayscale images to .nii.gz format to fit Swin-UNETR input usage")
parser.add_argument("--input_root", default="./Dataset", type=str, help="input directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")


def load_slices_as_cube(folder_path):
    """
    Load slices from a folder and stack them into a 3D cube.
    """
    slices = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0])):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            slices.append(np.array(img))
    return np.stack(slices, axis=-1)  # Stack along z-axis

def save_as_nii(cube, output_path, affine):
    """
    Save a 3D cube as a .nii.gz file with the specified affine transformation.
    """
    nii_image = nib.Nifti1Image(cube, affine)
    # Add metadata (you can customize further if needed)
    nii_image.header['xyzt_units'] = 10  # Set units to mm (millimeters)
    nii_image.header['cal_max'] = np.max(cube)
    nii_image.header['cal_min'] = np.min(cube)
    nib.save(nii_image, output_path)

def convert_dataset_to_nii(root_dir, output_dir):
    """
    Convert folders of image/mask slices to .nii.gz files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the affine matrix
    affine = np.array([
        [-1.0, -0.0, -0.0, 0.0],
        [-0.0, -1.0, -0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    for data_type in ['imgs', 'masks']:
        input_base = os.path.join(root_dir, data_type)
        output_base = os.path.join(output_dir, data_type)
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        for timestamp in os.listdir(input_base):
            print(f"Processing timestamp {timestamp}")
            timestamp_folder = os.path.join(input_base, timestamp)
            if not os.path.isdir(timestamp_folder):
                continue

            print(f"Processing {data_type}/{timestamp}...")
            cube = load_slices_as_cube(timestamp_folder)

            # Define output file extension based on data type
            file_extension = ".nii.gz" if data_type == "imgs" else ".seg.nii.gz"
            output_path = os.path.join(output_base, f"{timestamp}{file_extension}")

            save_as_nii(cube, output_path, affine)

if __name__ == "__main__":
    args = parser.parse_args()

    convert_dataset_to_nii(args.input_root, args.output_root)
    print("Conversion complete!")

# python img2nii.py --input_root /home/joycelyn/Desktop/Dataset/MHD-3DIS/ --output_root /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII