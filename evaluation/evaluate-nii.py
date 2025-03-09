import os
import glob
import torch
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import argparse

# python evaluate-nii.py --pred_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/unet-test/masks-output --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks

def load_nifti(filepath):
    """Load a nii.gz file and return a torch tensor."""
    data = nib.load(filepath).get_fdata()
    # Convert to tensor and ensure type is float32 (or long for integer labels)
    return torch.tensor(data).float()

def add_batch_channel(tensor):
    """
    Ensures the tensor has a batch and channel dimension.
    Expected shape for MONAI: (B, C, H, W, D)
    If tensor is 3D, add channel and batch dimensions.
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    elif tensor.ndim == 4:
        # If already has a channel (C, H, W, D) then add batch dimension.
        tensor = tensor.unsqueeze(0)  # (1, C, H, W, D)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference Evaluation")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing result images (.nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing gt images (im*.nii.gz)")
    args = parser.parse_args()
    

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))

    if len(pred_files) != len(gt_files):
        raise ValueError("Mismatch between number of prediction and ground truth files.")

    # Create metric objects.
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")

    dice_scores = []
    hausdorff_scores = []

    for pred_path, gt_path in zip(pred_files, gt_files):
        # Load the prediction and ground truth volumes.
        pred = load_nifti(pred_path)
        gt = load_nifti(gt_path)
        
        # Add batch and channel dimensions as required by MONAI.
        pred = add_batch_channel(pred)
        gt = add_batch_channel(gt)
        
        # Reset the metrics (since they are stateful)
        dice_metric.reset()
        hausdorff_metric.reset()
        
        # Update the metrics with the current pair.
        # Note: These metrics expect the segmentation to be in a binary or multi-class format.
        dice_metric(y_pred=pred, y=gt)
        hausdorff_metric(y_pred=pred, y=gt)
        
        # Compute aggregated metrics.
        dice_value = dice_metric.aggregate().item()
        hausdorff_value = hausdorff_metric.aggregate().item()
        
        dice_scores.append(dice_value)
        hausdorff_scores.append(hausdorff_value)
        
        print(f"{os.path.basename(pred_path)}:")
        print(f"  Dice Score: {dice_value:.4f}")
        print(f"  Hausdorff Distance: {hausdorff_value:.4f}")

    # Compute average scores over all volumes.
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_hausdorff = sum(hausdorff_scores) / len(hausdorff_scores)

    print("\nOverall Evaluation:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")

if __name__ == "__main__":
    main()