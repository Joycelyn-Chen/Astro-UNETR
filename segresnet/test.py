#!/usr/bin/env python

import glob
import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Resize, Activations, AsDiscrete
from monai.data import ArrayDataset, DataLoader
from monai.networks.nets import SegResNet

def main():
    parser = argparse.ArgumentParser(description="SegResNet 3D Segmentation Inference")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing input images (under test/imgs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save segmentation outputs")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Starting inference...")

    # Create the SegResNet model (must match training configuration).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,   # 4-channel input
        out_channels=3,  # 3 segmentation channels
        dropout_prob=0.2,
    ).to(device)

    # Load the checkpoint.
    checkpoint = torch.load(args.model_path, map_location=device)
    # If the checkpoint dictionary contains a "model" key, use it; otherwise assume it is the state dict.
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Define the inference transform.
    infer_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
    ])
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Get list of test images (under test/imgs).
    test_img_paths = sorted(glob.glob(os.path.join(args.data_dir, "test", "imgs", "*.nii.gz")))
    if not test_img_paths:
        logging.error("No input test images found in the specified directory.")
        sys.exit(1)

    # Create a dataset and data loader.
    test_ds = ArrayDataset(test_img_paths, infer_imtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    for idx, image in enumerate(test_loader):
        # Retrieve the original file’s affine.
        image_path = test_img_paths[idx]
        img_obj = nib.load(image_path)
        affine = img_obj.affine

        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            output = post_pred(output)
        seg_array = output.squeeze().cpu().numpy().astype(np.uint8)
        output_filename = os.path.basename(image_path).split('.')[0] + '.seg.nii.gz'
        output_path = os.path.join(args.output_dir, output_filename)
        seg_nifti = nib.Nifti1Image(seg_array, affine)
        nib.save(seg_nifti, output_path)
        logging.info(f"Saved segmentation to {output_path}")

    logging.info("Inference completed.")

if __name__ == "__main__":
    main()
