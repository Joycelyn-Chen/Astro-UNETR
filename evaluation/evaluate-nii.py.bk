import os
import glob
import torch
import torch.nn.functional as F  # Import functional module for interpolation
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import argparse
import numpy as np

# python evaluate-nii.py --pred_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/unet-test-epoch100/masks-output --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks
# python evaluate-nii.py --pred_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/segresnet-test/masks-output --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks

def load_nifti(filepath):
    """Load a nii.gz file and return a torch tensor."""
    data = nib.load(filepath).get_fdata()
    # Convert to tensor and ensure type is float32.
    return torch.tensor(data).float()

def add_batch_channel(tensor):
    """
    Ensure the tensor has batch and channel dimensions.
    Expected shape for MONAI: (B, C, H, W, D)
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)  # (1, C, H, W, D)
    return tensor

def compute_bounding_box_diagonal(segmentation):
    """
    Compute the diagonal length of the bounding box of a segmentation.
    Assumes segmentation is a NumPy array of shape (H, W, D) where foreground is > 0.5.
    """
    indices = np.argwhere(segmentation > 0.5)
    if indices.size == 0:
        return 0.0
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    diag_length = np.linalg.norm(max_idx - min_idx)
    return diag_length

def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference Evaluation")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing result images (.nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images (.nii.gz)")
    parser.add_argument("--resolution", type=int, default=256, help="Desired resolution (e.g., 256 or 128) to adjust GT data to match prediction resolution.")
    args = parser.parse_args()
    
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))

    print(len(pred_files))
    print(len(gt_files))

    if len(pred_files) != len(gt_files):
        raise ValueError("Mismatch between number of prediction and ground truth files.")

    # Create metric objects.
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")

    dice_scores = []
    hausdorff_scores = []
    hd_percentages = []

    for pred_path, gt_path in zip(pred_files, gt_files):
        # Load the prediction and ground truth volumes.
        pred = load_nifti(pred_path)
        gt = load_nifti(gt_path)
        
        # Add batch and channel dimensions as required by MONAI.
        pred = add_batch_channel(pred)
        gt = add_batch_channel(gt)
        
        # Adjust ground truth resolution if it differs from the desired resolution.
        # Assumes original ground truth has spatial dimensions (H, W, D) equal to 256.
        if gt.shape[2] != args.resolution:
            gt = F.interpolate(gt, size=(args.resolution, args.resolution, args.resolution), mode='nearest')
        
        # Reset the metrics (they are stateful)
        dice_metric.reset()
        hausdorff_metric.reset()
        
        # Update the metrics with the current pair.
        dice_metric(y_pred=pred, y=gt)
        hausdorff_metric(y_pred=pred, y=gt)
        
        # Compute aggregated metrics.
        dice_value = dice_metric.aggregate().item()
        hausdorff_value = hausdorff_metric.aggregate().item()
        
        # Compute the bounding box diagonal from the ground truth.
        # We assume gt is single-channel; extract first channel of first batch.
        gt_np = gt.cpu().numpy()[0, 0]
        diag_length = compute_bounding_box_diagonal(gt_np)
        if diag_length == 0:
            hd_percentage = 0.0
        else:
            hd_percentage = (hausdorff_value / diag_length) * 100.0

        dice_scores.append(dice_value)
        hausdorff_scores.append(hausdorff_value)
        hd_percentages.append(hd_percentage)
        
        print(f"{os.path.basename(pred_path)}:")
        print(f"  Dice Score: {dice_value:.4f}")
        print(f"  Hausdorff Distance: {hausdorff_value:.4f}")
        print(f"  HD Percentage: {100 - hd_percentage:.2f}%\n")

    # Compute average scores over all volumes.
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_hausdorff = sum(hausdorff_scores) / len(hausdorff_scores)
    avg_hd_percentage = sum(hd_percentages) / len(hd_percentages)

    print("\nOverall Evaluation:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")
    print(f"Average HD Percentage: {100 - avg_hd_percentage:.2f}%")

if __name__ == "__main__":
    main()
