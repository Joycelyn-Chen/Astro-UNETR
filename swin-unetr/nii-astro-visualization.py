# This code is for visualizing the .nii.gz results as image
# Input: 
#   - data_dir: where .nii.gz and .seg.nii.gz is stored
#   - mask_dir: where segmentation result .nii.gz is stored
#   - slice_num: the slice number to be plotted out
#   - timestamp: the timestamp of interest

# Usage: python nii-astrovisualization.py --mask_dir ./outputs/official_pretrained --slice_num 67 --timestamp 380
# Will store the result under mask_dir, with case number and slice number as .png image


import os
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="/home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/", type=str, help="dataset directory")
parser.add_argument("--mask_dir", default="./outputs/test1", type=str, help="mask directory")
parser.add_argument("--timestamp", default="380", type=str, help="target timestamp")
parser.add_argument("--slice_num", default=67, type=int, help="slice number within the cube input")


def main():
    args = parser.parse_args()

    img_add = os.path.join(args.data_dir, "imgs", f"{args.timestamp}.nii.gz")
    label_add = os.path.join(args.data_dir,"masks", f"{args.timestamp}.seg.nii.gz")
    seg_add = os.path.join(args.mask_dir, args.timestamp + ".nii.gz")

    img = nib.load(img_add).get_fdata()
    label = nib.load(label_add).get_fdata()
    seg = nib.load(seg_add).get_fdata()
    
    plt.figure(f"image - slice {args.slice_num}", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"image - slice {args.slice_num}")
    plt.imshow(img[:, :, args.slice_num], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(label[:, :, args.slice_num])
    plt.subplot(1, 3, 3)
    plt.title("segmentation")
    plt.imshow(seg[:, :, args.slice_num])
    # plt.show()
    plt.savefig(os.path.join(args.mask_dir, f"{args.timestamp}-{args.slice_num}.png"))

    print(f"Visualization saved at: {os.path.join(args.mask_dir, f'{args.timestamp}-{args.slice_num}.png')}")

if __name__ == "__main__":
    main()
