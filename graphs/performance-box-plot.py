import os
import glob
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric

# python performance-box-plot.py --root_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/results-output/ --gt_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test/masks


exp_root = {
        "single-modal": {
            "epoch 0": "swin-unetr-pretrained",
            "epoch 100": "swin-unetr-epoch100",
            "epoch 200": "swin-unetr-epoch200",
            "epoch 300": "swin-unetr-epoch300"
        },
        "multi-modal": {
            "epoch 0": "astro-unetr-multimodal",
            "epoch 100": "astro-unetr-multimodal-epoch100",
            "epoch 200": "astro-unetr-multimodal-epoch200",
            "epoch 300": "astro-unetr-multimodal-epoch300"
        },
        "r-loss": {
            "epoch 0": "",
            "epoch 100": "astro-unetr-r-loss-epoch100",
            "epoch 200": "astro-unetr-r-loss-epoch200",
            "epoch 300": ""
        }
    }

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

def compute_dice(pred_path, gt_path, resolution=256):
    """
    Compute the dice score between a prediction and ground truth .nii.gz file.
    """
    pred = load_nifti(pred_path)
    gt = load_nifti(gt_path)
    pred = add_batch_channel(pred)
    gt = add_batch_channel(gt)
    
    # Adjust ground truth resolution if needed.
    if gt.shape[2] != resolution:
        gt = F.interpolate(gt, size=(resolution, resolution, resolution), mode='nearest')
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric.reset()
    dice_metric(y_pred=pred, y=gt)
    dice_value = dice_metric.aggregate().item()
    return dice_value



def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference Evaluation")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory leading to all the result folders.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images (.nii.gz)")
    args = parser.parse_args()
    
    
    # Dictionary to store dice scores for each experiment.
    # Structure: results[category][epoch] = list of dice scores
    results = {}
    
    # Loop over categories and experiment cases.
    for category, experiments in exp_root.items():
        results[category] = {}
        for epoch_label, folder in experiments.items():
            if not folder:  # Skip experiments with empty folder names.
                continue
            exp_dir = os.path.join(args.root_dir, folder, "masks-output")
            pred_files = sorted(glob.glob(os.path.join(exp_dir, "*.nii.gz")))
            gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))
            
            if len(pred_files) != len(gt_files):
                print(f"Warning: For {category} {epoch_label} the number of prediction files does not match ground truth files.")
            
            dice_scores = []
            # Assuming that the sorted file lists correspond to the same test cases.
            for pred_file, gt_file in zip(pred_files, gt_files):
                try:
                    dice_val = compute_dice(pred_file, gt_file)
                    dice_scores.append(dice_val)
                except Exception as e:
                    print(f"Error processing files:\n  {pred_file}\n  {gt_file}\nError: {e}")
            results[category][epoch_label] = dice_scores
            print(f"Category '{category}', {epoch_label}: computed {len(dice_scores)} dice scores")
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Offsets for the x positions (to avoid overlapping boxes at the same epoch)
    # Adjust the offset (in the same units as epoch values) as needed.
    offsets = {
        "single-modal": -5,
        "multi-modal": 0,
        "r-loss": 5
    }
    
    # To collect line-plot data (x positions and median values) for each category.
    line_data = {}
    
    # Iterate over each category and its experiments.
    for category, experiments in results.items():
        line_data[category] = ([], [])
        # Sort experiment cases by epoch number (e.g., "epoch 0", "epoch 100", ...)
        sorted_exps = sorted(experiments.items(), key=lambda x: int(x[0].split()[1]))
        for epoch_label, dice_scores in sorted_exps:
            epoch_num = int(epoch_label.split()[1])
            x_pos = epoch_num + offsets[category]
            # Create a box plot for the dice score distribution at this experiment.
            bp = ax.boxplot(dice_scores, positions=[x_pos], widths=8, patch_artist=True, showfliers=False)
            
            # Color the boxes differently for each category.
            if category == "single-modal":
                box_color = 'skyblue'
            elif category == "multi-modal":
                box_color = 'lightgreen'
            elif category == "r-loss":
                box_color = 'salmon'
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
            
            # Record the median value for the line plot.
            if dice_scores:
                median_val = np.median(dice_scores)
                line_data[category][0].append(x_pos)
                line_data[category][1].append(median_val)
    
    # Plot a line (with markers) connecting the medians for each category.
    for category, (x_vals, medians) in line_data.items():
        if category == "single-modal":
            line_color = 'blue'
        elif category == "multi-modal":
            line_color = 'green'
        elif category == "r-loss":
            line_color = 'red'
        ax.plot(x_vals, medians, marker='o', linestyle='-', color=line_color, label=category)
    
    # Set x-axis ticks at the standard epoch values.
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Box Plot of Dice Score Performance Across Experiment Cases")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/peformace-box-plot.png")

if __name__ == "__main__":
    main()
