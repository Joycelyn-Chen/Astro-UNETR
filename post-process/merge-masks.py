import os
import cv2 as cv
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Flipping grayscale images to fit Swin-UNETR input usage")
parser.add_argument("--mask1_dir", default="./Dataset", type=str, help="1st input mask directory")
parser.add_argument("--mask2_dir", default="./Dataset", type=str, help="2nd input mask directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")
parser.add_argument("--timestamp", default=380, type=int, help="timestamp of interest")


def merge_masks(timestamp, root_dir1, root_dir2, output_root):
    """
    Merge two masks for the same object using a logical OR operation, ensuring sorted input files.

    Args:
        timestamp (str): Timestamp of the masks.
        root_dir1 (str): Root directory for the first set of masks.
        root_dir2 (str): Root directory for the second set of masks.
        output_root (str): Root directory to save merged masks.

    Returns:
        None
    """
    # Define paths for both mask directories and the output directory
    mask_dir1 = os.path.join(root_dir1, str(timestamp))
    mask_dir2 = os.path.join(root_dir2, str(timestamp))
    output_dir = os.path.join(output_root, str(timestamp))
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List and sort mask files in both directories
    files1 = sorted([f for f in os.listdir(mask_dir1) if f.endswith(".png")])
    files2 = sorted([f for f in os.listdir(mask_dir2) if f.endswith(".png")])

    # Create dictionaries with z-coordinates as keys
    files1_dict = {os.path.splitext(f)[0]: f for f in files1}
    files2_dict = {os.path.splitext(f)[0]: f for f in files2}

    # Merge masks by iterating over the sorted union of filenames
    for z_coord in sorted(files1_dict.keys() | files2_dict.keys(), key=lambda x: int(x)):
        # Read mask slices, default to zeros if one mask is missing
        mask1_path = os.path.join(mask_dir1, files1_dict.get(z_coord, ""))
        mask2_path = os.path.join(mask_dir2, files2_dict.get(z_coord, ""))
        
        mask1 = cv.imread(mask1_path, cv.IMREAD_GRAYSCALE) if z_coord in files1_dict else None
        mask2 = cv.imread(mask2_path, cv.IMREAD_GRAYSCALE) if z_coord in files2_dict else None

        # Convert masks to binary (values > 0 -> 1, else 0)
        binary1 = (mask1 > 0).astype(np.uint8) if mask1 is not None else np.zeros_like(mask2, dtype=np.uint8)
        binary2 = (mask2 > 0).astype(np.uint8) if mask2 is not None else np.zeros_like(mask1, dtype=np.uint8)

        # Perform logical OR operation
        merged_mask = np.logical_or(binary1, binary2).astype(np.uint8) * 255  # Convert to uint8 for saving

        # Save the merged mask
        output_path = os.path.join(output_dir, f"{z_coord}.png")
        cv.imwrite(output_path, merged_mask)

    print(f"Merged masks saved under {output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    
    merge_masks(args.timestamp, args.mask1_dir, args.mask2_dir, args.output_root)

# python merge-masks.py --mask1_dir /home/joy0921/Desktop/Dataset/img_pix256/masks/SB-large/SB230 --mask2_dir /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230 --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks-merged --timestamp 420