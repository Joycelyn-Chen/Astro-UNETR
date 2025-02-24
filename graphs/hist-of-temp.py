import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yt
import cv2 as cv  
import scipy.ndimage as ndimage  # for 3D morphological operations

def load_mask_cube(mask_root, timestamp, cube_dim=256):
    """
    Step 1: Read the input 3D cube of binary masks.
    Assumes files are stored in: mask_root/timestamp and named as <slice_index>.png.
    """
    mask_dir = os.path.join(mask_root, timestamp)
    mask_cube = np.zeros((cube_dim, cube_dim, cube_dim), dtype=np.uint8)
    
    for file_name in sorted(os.listdir(mask_dir)):
        if file_name.endswith('.png'):
            slice_index = int(os.path.splitext(file_name)[0])
            if slice_index < cube_dim:
                file_path = os.path.join(mask_dir, file_name)
                mask_slice = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if mask_slice is None:
                    raise ValueError(f"Failed to read {file_path}")
                mask_cube[:, :, slice_index] = mask_slice
    return mask_cube

def load_temperature_cube(hdf5_root, timestamp, hdf5_prefix, pixel_boundary, lower_bound, upper_bound):
    """
    Step 2: Read HDF5 raw data and extract temperature values for the center cube.
    """
    hdf5_file = os.path.join(hdf5_root, f'{hdf5_prefix}{timestamp}')
    ds = yt.load(hdf5_file)
    
    # Define center (here simply the origin in code units)
    arb_center = ds.arr([0, 0, 0], 'code_length')
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    
    # Create an arbitrary grid of desired resolution
    grid = ds.arbitrary_grid(left_edge, right_edge, dims=(pixel_boundary, pixel_boundary, pixel_boundary))
    
    # Extract the temperature cube from the arbitrary grid and convert to Kelvin
    temp_cube = grid[("flash", "temp")][
        lower_bound:upper_bound,
        lower_bound:upper_bound,
        lower_bound:upper_bound
    ].to('K').value
    return temp_cube

def spherical_kernel(size):
    """
    Creates a 3D spherical structuring element with the given size.
    The sphere's radius is (size - 1)/2.
    """
    center = (np.array([size, size, size]) - 1) / 2.0
    x, y, z = np.indices((size, size, size))
    kernel = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= (center[0]**2)
    return kernel.astype(np.uint8)

def morphological_difference(mask, kernel_size=5):
    """
    Steps 3 & 4: Dilate and erode the 3D mask and subtract the eroded mask from the dilated mask.
    """
    # Create a cubic structuring element
    # structure = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
    structure = spherical_kernel(kernel_size)
    
    # Perform 3D dilation and erosion
    eroded = ndimage.binary_erosion(mask, structure=structure)
    mask_original = ndimage.binary_dilation(eroded, structure=structure)

    dilated = ndimage.binary_dilation(mask_original, structure=structure)
    # eroded = ndimage.binary_erosion(mask, structure=structure)
    
    # Compute the difference (convert boolean arrays to uint8 for subtraction)
    mask_diff = dilated.astype(np.uint8) - eroded.astype(np.uint8)

    slice_index = mask_diff.shape[2] // 2
    # Multiply by 255 to scale the binary mask to full grayscale range
    mask_slice = (eroded[:, :, slice_index] * 255).astype(np.uint8)
    # mask_slice = (mask[:, :, slice_index]).astype(np.uint8)
    save_path = os.path.join('/home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230', 'mask.png')
    cv.imwrite(save_path, mask_slice)

    return mask_diff

def plot_temperature_histogram(temp_cube, mask_diff, output_path):
    """
    Steps 5, 6 & 7: Extract temperature values where the mask difference equals 1,
    then plot the histogram with a vertical line indicating a sharp cutoff.
    """
    # Accumulate the temperature values into a 1D array
    temp_values = temp_cube[mask_diff == 1]
    
    plt.figure()
    plt.hist(temp_values, bins=50, color='b') #facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=2.5, alpha=1)

    # n, bins, patches = plt.hist(temp_values, bins=500, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

    # n = n.astype('int') # it MUST be integer
    # # Good old loop. Choose colormap of your taste
    # for i in range(len(patches)):
    #     patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
    
    # Make one bin stand out   
    # patches[47].set_fc('red') # Set color
    # patches[47].set_alpha(1) # Set opacity
    

    # Mark a sharp cutoff at 1e4 K
    plt.axvline(x=10e2, color='r', linestyle='--', label='Cut off at 1e2 K')

    plt.xlabel("Temperature (K)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title("Histogram of Temperature Values", fontsize=12)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot the histogram of temperature values around the bubble"
    )
    parser.add_argument("--hdf5_root", default="./Dataset", type=str, help="Input HDF5 directory")
    parser.add_argument("--mask_root", default="./Dataset", type=str, help="Input mask directory")
    parser.add_argument("--output_root", default="./Dataset", type=str, help="Output directory")
    parser.add_argument("--timestamp", default='209', type=str, help="Timestamp of the HDF5 file")
    parser.add_argument('-lb', '--lower_bound', default=0, type=int, help="Lower bound for the cube")
    parser.add_argument('-up', '--upper_bound', default=256, type=int, help="Upper bound for the cube")
    parser.add_argument('-pixb', '--pixel_boundary', default=256, type=int, help="Pixel resolution for the grid")
    args = parser.parse_args()

    # Step 1: Load the 3D cube of binary masks
    mask_cube = load_mask_cube(args.mask_root, args.timestamp, cube_dim=256)
    
    # Step 2: Read HDF5 raw data and extract the temperature cube
    hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'
    temp_cube = load_temperature_cube(
        args.hdf5_root, args.timestamp, hdf5_prefix,
        args.pixel_boundary, args.lower_bound, args.upper_bound
    )
    
    # Steps 3 & 4: Apply morphological operations and compute the difference mask
    mask_diff = morphological_difference(mask_cube, kernel_size=10)
    
    # Steps 5, 6 & 7: Accumulate temperature data, plot the histogram, and mark the cutoff
    output_file = os.path.join(args.output_root, 'hist-of-temp.png')
    plot_temperature_histogram(temp_cube, mask_diff, output_file)

    print(f"Done. Plot saved at: {output_file}")

if __name__ == "__main__":
    main()
