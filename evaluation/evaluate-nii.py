import os
import glob
import torch
import torch.nn.functional as F  # For interpolation
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import argparse
import numpy as np

# python evaluate-nii.py --pred_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/unet-test-epoch100/masks-output --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks
# python evaluate-nii.py --pred_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/segresnet-test/masks-output --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks

def load_nifti(filepath):
    """Load a nii.gz file and return a torch tensor."""
    data = nib.load(filepath).get_fdata()
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
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted images (.nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images (.nii.gz)")
    parser.add_argument("--resolution", type=int, default=256, help="Desired resolution to adjust GT data to match prediction resolution.")
    args = parser.parse_args()
    
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))
    
    print("Number of prediction files:", len(pred_files))
    print("Number of ground truth files:", len(gt_files))
    
    if len(pred_files) != len(gt_files):
        raise ValueError("Mismatch between number of prediction and ground truth files.")
    
    # Define fold groups.
    # Fold 1: first half of lifespan (years 380 to 590)
    fold1_years = set(range(380, 600, 10))
    # Fold 2: second half of lifespan (years 600 to 790)
    fold2_years = set(range(600, 800, 10))
    # Fold 3: discrete bubble years.
    fold3_years = {380, 390, 400, 410, 430, 440, 450, 460, 570, 580, 590, 600, 610, 620}
    # Fold 4: interconnected bubble years.
    fold4_years = {420, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 630, 640, 650,
                     660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790}
    
    # Prepare a list to store computed metrics for each file.
    results = []

    # Create metric objects.
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
    
    for pred_path, gt_path in zip(pred_files, gt_files):
        # Load prediction and ground truth.
        pred = load_nifti(pred_path)
        gt = load_nifti(gt_path)
        
        # Add batch and channel dimensions.
        pred = add_batch_channel(pred)
        gt = add_batch_channel(gt)
        
        # Adjust GT resolution if needed.
        if gt.shape[2] != args.resolution:
            gt = F.interpolate(gt, size=(args.resolution, args.resolution, args.resolution), mode='nearest')
        
        # Reset and update metrics.
        dice_metric.reset()
        hausdorff_metric.reset()
        dice_metric(y_pred=pred, y=gt)
        hausdorff_metric(y_pred=pred, y=gt)
        
        dice_value = dice_metric.aggregate().item()
        hausdorff_value = hausdorff_metric.aggregate().item()
        
        # Compute the bounding box diagonal from ground truth.
        gt_np = gt.cpu().numpy()[0, 0]
        diag_length = compute_bounding_box_diagonal(gt_np)
        hd_percentage = (hausdorff_value / diag_length * 100.0) if diag_length > 0 else 0.0
        
        # Extract the year from the ground truth filename.
        filename = os.path.basename(gt_path)
        if filename.endswith('.nii.gz'):
            year_str = filename[:-11]  # remove ".seg.nii.gz"
        else:
            year_str = os.path.splitext(filename)[0]
        try:
            year = int(year_str)
        except ValueError:
            raise ValueError(f"Filename {filename} does not start with a valid year.")
        
        # Save the results for this file.
        results.append({
            "filename": os.path.basename(pred_path),
            "year": year,
            "dice": dice_value,
            "hd_percentage": hd_percentage
        })
        
        print(f"{os.path.basename(pred_path)}:")
        # print(f"  Dice Score: {dice_value:.4f}")
        # print(f"  Hausdorff Distance: {hausdorff_value:.4f}")
        # print(f"  HD Percentage: {100 - hd_percentage:.2f}%\n")
    
    # Function to calculate averages for a given fold.
    def compute_fold_metrics(years_set, fold_name):
        fold_items = [item for item in results if item["year"] in years_set]
        if fold_items:
            avg_dice = sum(item["dice"] for item in fold_items) / len(fold_items)
            avg_hd_pct = sum(item["hd_percentage"] for item in fold_items) / len(fold_items)
            print(f"{fold_name}:")
            print(f"  Average Dice Score: {avg_dice:.4f}")
            print(f"  Average HD Percentage: {100 - avg_hd_pct:.2f}%\n")
        else:
            print(f"{fold_name}: No data available.\n")
            
    print("\nFold-wise Evaluation:")
    compute_fold_metrics(fold1_years, "Fold 1 (Years 380-590)")
    compute_fold_metrics(fold2_years, "Fold 2 (Years 600-790)")
    compute_fold_metrics(fold3_years, "Fold 3 (Discrete Bubble Years)")
    compute_fold_metrics(fold4_years, "Fold 4 (Interconnected Bubble Years)")

    # Overall metrics across all files.
    overall_dice = sum(item["dice"] for item in results) / len(results)
    overall_hd_pct = sum(item["hd_percentage"] for item in results) / len(results)
    print("\nOverall Evaluation:")
    print(f"Average Dice Score: {overall_dice:.4f}")
    print(f"Average HD Percentage: {100 - overall_hd_pct:.2f}%")

if __name__ == "__main__":
    main()
