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
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Resize
from monai.data import ArrayDataset, DataLoader
from monai.networks.nets import SegResNet

# python test.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/segresnet-test/masks-output --model_path /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/segresnet-test/logs/segresnet_checkpoint_10600.pt
# ------------------------------------------------------------------------------
# Duplicate channels helper function (same as used during training).
def duplicate_channels(x):
    """
    If x has one channel, duplicate it along the channel axis to create a 4-channel input.
    Supports both numpy arrays and torch tensors.
    """
    if x.shape[0] == 1:
        if isinstance(x, np.ndarray):
            return np.repeat(x, 4, axis=0)
        elif isinstance(x, torch.Tensor):
            return x.repeat(4, 1, 1, 1)
    return x

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
        in_channels=4,   # 4-channel input after duplicating channels
        out_channels=3,  # 3 segmentation channels
        dropout_prob=0.2,
    ).to(device)

    # Load the checkpoint.
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Define the inference transform (matching training/validation transforms).
    infer_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        # Resize((96, 96, 96)),
        duplicate_channels,
    ])

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
            # Obtain model prediction (3-channel output).
            output = model(image)
            # Apply sigmoid activation to get probabilities.
            activated_output = torch.sigmoid(output)
            # Aggregate channels by taking the maximum probability across them.
            aggregated_output = activated_output.max(dim=1, keepdim=True)[0]
            # Threshold the aggregated output to get a binary mask.
            binary_mask = (aggregated_output > 0.5).float()

        # Squeeze extra dimensions, move to CPU, convert to numpy, and cast to uint8.
        seg_array = binary_mask.squeeze().cpu().numpy().astype(np.uint8)
        output_filename = os.path.basename(image_path).split('.')[0] + '.seg.nii.gz'
        output_path = os.path.join(args.output_dir, output_filename)
        seg_nifti = nib.Nifti1Image(seg_array, affine)
        nib.save(seg_nifti, output_path)
        logging.info(f"Saved segmentation to {output_path}")

    logging.info("Inference completed.")
    print(f"Done. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()