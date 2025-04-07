#!/usr/bin/env python

import glob
import logging
import os
import sys
from pathlib import Path
import argparse

import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Resize, Activations, AsDiscrete
from monai.networks.nets import UNet

# python test.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/unet-epoch300/masks-output --model_path /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/unet-test/logs/net_checkpoint_1200.pt

def main():
    parser = argparse.ArgumentParser(description="UNet 3D Segmentation Inference")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing input images (im*.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save segmentation outputs")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Starting inference...")

    # Create the UNet model (make sure its configuration matches the training setup).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Load the checkpoint.
    checkpoint = torch.load(args.model_path, map_location=device)
    if "net" in checkpoint:
        # If the checkpoint contains a dict with key "net"
        model.load_state_dict(checkpoint["net"].state_dict() if hasattr(checkpoint["net"], "state_dict") else checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Define the inference transform.
    infer_trans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        # Resize((96, 96, 96)),
    ])
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Get list of input images.
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test', 'imgs', "*.nii.gz")))
    if not image_paths:
        logging.error("No input images found in the specified directory.")
        sys.exit(1)

    for image_path in image_paths:
        logging.info(f"Processing {image_path}...")
        # Load image (and get the affine from the original file)
        img_obj = nib.load(image_path)
        affine = img_obj.affine
        # Apply inference transform.
        image_array = infer_trans(image_path)
        # Convert to tensor, add batch dimension.
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            output = post_pred(output)
        # Remove batch and channel dimensions.
        seg_array = output.squeeze().cpu().numpy().astype(np.uint8)
        # Save the output segmentation using the original affine.
        output_filename = os.path.basename(image_path).split('.')[0] + '.seg.nii.gz'
        output_path = os.path.join(args.output_dir, output_filename)
        seg_nifti = nib.Nifti1Image(seg_array, affine)
        nib.save(seg_nifti, output_path)
        logging.info(f"Saved segmentation to {output_path}")

    logging.info("Inference completed.")

if __name__ == "__main__":
    main()
