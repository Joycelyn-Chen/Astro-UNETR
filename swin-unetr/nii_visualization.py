# This code is for visualizing the .nii.gz results as image
# Input: 
#   - data_dir: where -t1c.nii.gz and -seg.nii.gz is stored
#   - mask_dir: where segmentation result .nii.gz is stored
#   - slice_num: the slice number to be plotted out
#   - case_num: the case number of interest

# Usage: python nii_visualization.py --mask_dir /media/joycelyn/HDD1/home/joycelyn/Medical-3DIS/research-contributions/SwinUNETR/BRATS21/outputs/official_pretrained --slice_num 67 --case_num 01033
# Will store the result under mask_dir, with case number and slice number as .png image


import os
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

import numpy as np

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="/media/joycelyn/HDD1/home/joycelyn/Medical-3DIS/Dataset/brats_2023/", type=str, help="dataset directory")
parser.add_argument("--mask_dir", default="./outputs/test1", type=str, help="mask directory")
parser.add_argument("--slice_num", default=67, type=int, help="slice number within the cube input")
parser.add_argument("--case_num", type=str, help="case number for the cube")


def main():
    args = parser.parse_args()

    img_add = os.path.join(
        args.data_dir,
        "TrainingData/BraTS-GLI-" + args.case_num + "-000/BraTS-GLI-" + args.case_num + "-000-t1c.nii.gz",
    )
    label_add = os.path.join(
        args.data_dir,
        "TrainingData/BraTS-GLI-" + args.case_num + "-000/BraTS-GLI-" + args.case_num + "-000-seg.nii.gz",
    )

    seg_add = os.path.join(
        args.mask_dir,
        "BraTS-GLI-" + args.case_num + ".nii.gz",
    )
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
    # plt.savefig(os.path.join(args.mask_dir, f"BraTS-GLI-{args.case_num}-{args.slice_num}.png"))
    print(label[:, :, args.slice_num])

    np.savetxt("tmp.txt", label[:, :, args.slice_num])
    # with open("tmp.txt", "w") as f:
    #     f.write(str(label[:, :, args.slice_num]))

    print(f"Visualization saved at: {os.path.join(args.mask_dir, f'BraTS-GLI-{args.case_num}-{args.slice_num}.png')}")

if __name__ == "__main__":
    main()
