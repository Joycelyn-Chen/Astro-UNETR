from PIL import Image
import numpy as np
import cv2 as cv
import os
import argparse
from matplotlib import pyplot as plt
import yt

hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'

# python hist-of-temp.py --hdf5_root --mask_root --timestamp 209

parser = argparse.ArgumentParser(description="Plotting the histogram of temperature values around the bubble")
parser.add_argument("--hdf5_root", default="./Dataset", type=str, help="input image directory")
parser.add_argument("--mask_root", default="./Dataset", type=str, help="input mask directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")
parser.add_argument("--timestamp", default='209', type=str, help='timestamp of the HDF5 file')
parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default = 0, type = int)
parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default = 256, type = int)
parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)

def get_temp(obj, x_range, y_range, z_range):    
    temp = obj["flash", "temp"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('K').value      

    return temp

if __name__ == "__main__":
    args = parser.parse_args()
    # 1. read the input 3D cube of binary masks, store in a 3D array
    mask = np.zeros((256, 256, 256), dtype=np.int8)
    for mask_file in sorted(os.listdir(args.mask_root, args.timestamp)):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(args.mask_root, args.timestamp, mask_file)
            mask_slice = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            mask[:, :, int(mask_file.split('.')[0])] = mask_slice
    

    # 2. read HDF5 raw data, read temperature value for center 256 cube
    ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(hdf5_prefix, args.timestamp)))

    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))

    temp_cube = get_temp(obj, (args.lower_bound, args.upper_bound), (args.lower_bound, args.upper_bound), (args.lower_bound, args.upper_bound))

    # 3. both dilate and erode the 3D mask by 10 pixels
    kernel = np.ones((10, 10, 10), np.uint8)
    dilated = cv.dilate(mask, kernel, iterations=1)
    eroded = cv.erode(mask, kernel, iterations=1)


    # 4. then minus the dilated 3D masks by the eroded 3D mask.
    mask = dilated - eroded 

    # 5. accumulate the remaining data points into an 1D array
    temp_1D = temp_cube[mask==1]

    # 6. graph the 1D array as a histogram of temperature.
    plt.hist(temp_1D, bins=100, alpha=0.7, color='b')
    plt.savefig(os.path.join(args.output_root, 'hist-of-temp.png'))

    # 7. look for a sharp cut off
    plt.axvline(x=1e4, color='r', linestyle='--')