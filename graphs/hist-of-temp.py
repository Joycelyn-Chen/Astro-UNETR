import os
import numpy as np
import matplotlib.pyplot as plt
import yt
import cv2 as cv  
import scipy.ndimage as ndimage  # for 3D morphological operations
from matplotlib import colors

DEBUG = True
hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'

# python hist-of-temp.py --hdf5_root /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc --mask_root /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230 --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/hist-of-temp/ratio --start_timestamp 380 --end_timestamp 480 -i 10

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

def spherical_kernel(kernel_size):
    """
    Creates a 3D spherical structuring element with the given size.
    The sphere's radius is (size - 1)/2.
    """
    center = (np.array([kernel_size, kernel_size, kernel_size]) - 1) / 2.0
    x, y, z = np.indices((kernel_size, kernel_size, kernel_size))
    kernel = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= (center[0]**2)
    return kernel.astype(np.uint8)

def timestamp2Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191


def morphological_difference(mask, kernel_size=5):
    """
    Steps 3 & 4: Dilate and erode the 3D mask and subtract the eroded mask from the dilated mask.
    """
    # Create a cubic structuring element
    # structure = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
    structure = spherical_kernel(kernel_size=3)
    # Perform 3D dilation and erosion
    eroded = ndimage.binary_erosion(mask_cube, structure=structure)

    structure = spherical_kernel(kernel_size=5)
    mask_original = ndimage.binary_dilation(eroded, structure=structure)

    structure = spherical_kernel(kernel_size=10)
    dilated = ndimage.binary_dilation(mask_original, structure=structure)

    # Compute the difference (convert boolean arrays to uint8 for subtraction)
    mask_diff = np.logical_xor(dilated, mask_original)

    return mask_diff

def calc_ratio(temp_values):
    # Compute normalized weights so that the total volume sums to 1.
    # Each data point is assigned an equal fraction.
    weights = np.ones_like(temp_values) / len(temp_values)

    # Calculate the volume (fraction) of points below 1e3 K and above 1e6 K.
    vol_below = weights[temp_values < 1e3].sum()
    vol_above = weights[temp_values > 1e6].sum()

    # Avoid division by zero in case no point is above 1e6 K.
    # fraction = vol_above / vol_below if vol_above > 0 else np.nan
    fraction = vol_above / 1 if vol_above > 0 else np.nan # len(temp_values)

    return fraction

    
def plot_temperature_histogram(temp_cube, mask_diff, output_path):
    """
    Steps 5, 6 & 7: Extract temperature values where the mask difference equals 1,
    then plot the histogram with a vertical line indicating a sharp cutoff.
    """
    # Accumulate the temperature values into a 1D array
    temp_values = temp_cube[mask_diff == 1]
    temp_values_original = temp_cube[eroded == 1]

    ratio = calc_ratio(temp_values)

    # Convert temperatures to log10 scale for plotting.
    log_temp = np.log10(temp_values)
    log_temp_interior = np.log10(temp_values_original)

    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = 80
    x_min, x_max = 0, 10
    bins = np.linspace(x_min, x_max, n_bins + 1)

    # Create normalized weights for histogram plotting.
    dx = (x_max - x_min) / n_bins
    bins_per_order = 1.0 / dx  # typically 50 for [0,10] with dx=0.2
    weights_hist = np.ones_like(log_temp) / len(log_temp) * bins_per_order
    weights_hist_interior = np.ones_like(log_temp_interior) / len(log_temp_interior) * bins_per_order

    # Plot the step histogram on top.
    ax.hist(log_temp, bins=bins, histtype='step', linestyle='solid',
            weights=weights_hist, label='Bubble Exterior', color='black')

    ax.hist(log_temp_interior, bins=n_bins, histtype='step', linestyle='solid',          #dashed
            weights=weights_hist_interior, label='Bubble Interior', color='gray')

    # Mark a vertical line at the mean log-temperature.
    ax.axvline(x=log_temp.mean(), color='r', linestyle='--', label='Exterior Mean')

    # Set axis labels and title (including the fraction result).
    ax.set_xlabel("Temperature ($\\log_{10}(K)$)", fontsize=12)
    ax.set_ylabel("fraction / dex", fontsize=12)
    ax.set_title(f"t = {timestamp2Myr(int(timestamp))} Myr, Ratio = {fraction:.3f}", fontsize=14)

    # Set axis limits and scaling.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1e-5, 1e2)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return ratio, temp_values

def plot_ratio(temp_dict, ratio_dict, output_file):
    """
    Creates a combined plot for each timestamp.
    
    For each timestamp:
      - Categorize the 1D temperature array into:
            cold: T < 39.8,
            cool: 39.8 < T < 1e4,
            warm: 1e4 < T < 2e4,
            transition: 2e4 < T < 1e5.5, and
            hot: T > 1e5.5.
      - Compute the fraction of the total (normalized so that the 5 fractions sum to 1).
      - Plot a stacked bar (one bin per timestamp) where each colored section corresponds
        to a temperature category.
      - Overlay a line plot showing the ratio value for that timestamp.
    
    Inputs:
        temp_dict: dictionary with keys as timestamps and values as 1D arrays of temperatures.
        ratio_dict: dictionary with keys as timestamps and values as the ratio (a scalar).
        output_file: path to save the resulting plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sort timestamps (assume keys are convertible to int)
    timestamps = sorted(temp_dict.keys(), key=lambda x: int(x))
    # Convert timestamps to Myr for x-axis (using the provided conversion function)
    timeMyrs = [timestamp2Myr(int(ts)) for ts in timestamps]
    
    # Initialize lists for the normalized fractions for each temperature category.
    cold_fracs = []
    cool_fracs = []
    warm_fracs = []
    trans_fracs = []
    hot_fracs = []
    
    # Loop over each timestamp to calculate fractions
    for ts in timestamps:
        # Convert temperature data to a NumPy array
        temps = np.array(temp_dict[ts])
        total = len(temps)
        if total == 0:
            # If no data, use zeros (avoid division by zero)
            cold_fracs.append(0)
            cool_fracs.append(0)
            warm_fracs.append(0)
            trans_fracs.append(0)
            hot_fracs.append(0)
            continue
        
        # Count data points in each category (note the use of strict inequalities as described)
        cold = np.sum(temps < 39.8)
        cool = np.sum((temps > 39.8) & (temps < 1e4))
        warm = np.sum((temps > 1e4) & (temps < 2e4))
        trans = np.sum((temps > 2e4) & (temps < 10**5.5))
        hot = np.sum(temps > 10**5.5)
        
        # Normalize so that the fractions sum to 1
        cold_fracs.append(cold / total)
        cool_fracs.append(cool / total)
        warm_fracs.append(warm / total)
        trans_fracs.append(trans / total)
        hot_fracs.append(hot / total)
    
    # Create the figure and axis
    plt.figure(figsize=(12, 8))
    bar_width = 0.8  # width of each bar
    
    # Plot the stacked bars.
    # The first layer (cold) starts at 0.
    plt.bar(timeMyrs, cold_fracs, width=bar_width, color='blue', label='Cold (T < 39.8)')
    # Next layer (cool) starts at the top of cold layer.
    plt.bar(timeMyrs, cool_fracs, width=bar_width, bottom=cold_fracs, color='cyan', label='Cool (39.8 < T < 1e4)')
    
    # Compute cumulative bottoms for stacking subsequent layers
    cum_bottom = np.array(cold_fracs) + np.array(cool_fracs)
    plt.bar(timeMyrs, warm_fracs, width=bar_width, bottom=cum_bottom, color='green', label='Warm (1e4 < T < 2e4)')
    
    cum_bottom += np.array(warm_fracs)
    plt.bar(timeMyrs, trans_fracs, width=bar_width, bottom=cum_bottom, color='orange', label='Transition (2e4 < T < 1e5.5)')
    
    cum_bottom += np.array(trans_fracs)
    plt.bar(timeMyrs, hot_fracs, width=bar_width, bottom=cum_bottom, color='red', label='Hot (T > 1e5.5)')
    
    # Overlay the ratio line plot.
    # Get ratio values in the same sorted order of timestamps.
    ratio_list = [ratio_dict[ts] for ts in timestamps]
    plt.plot(timeMyrs, ratio_list, label='Ratio', color='black', linestyle='-', marker='o', linewidth=2)
    
    # Add labels, title, and grid
    plt.xlabel('Time (Myr)', fontsize=18)
    plt.ylabel('Ratio', fontsize=18)
    plt.title('Temperature Composition and Ratio over Time', fontsize=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description="Plot the histogram of temperature values around the bubble"
    )
    parser.add_argument("--hdf5_root", default="./Dataset", type=str, help="Input HDF5 directory")
    parser.add_argument("--mask_root", default="./Dataset", type=str, help="Input mask directory")
    parser.add_argument("--output_root", default="./Dataset", type=str, help="Output directory")
    parser.add_argument("--start_timestamp", default=380, type=int, help="Timestamp of the HDF5 file")
    parser.add_argument("--end_timestamp", default=480, type=int, help="Timestamp of the HDF5 file")
    parser.add_argument('-lb', '--lower_bound', default=0, type=int, help="Lower bound for the cube")
    parser.add_argument('-up', '--upper_bound', default=256, type=int, help="Upper bound for the cube")
    parser.add_argument('-pixb', '--pixel_boundary', default=256, type=int, help="Pixel resolution for the grid")
    parser.add_argument('i', 'interval', default=10, type = int, help='Timestamp increase interval')
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    ratio_dict = {}
    temp_dict = {}

    for current_timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.interval):
        # Step 1: Load the 3D cube of binary masks
        mask_cube = load_mask_cube(args.mask_root, str(current_timestamp), cube_dim=256)
        
        # Step 2: Read HDF5 raw data and extract the temperature cube
        hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'
        temp_cube = load_temperature_cube(
            args.hdf5_root, current_timestamp, hdf5_prefix,
            args.pixel_boundary, args.lower_bound, args.upper_bound
        )
        
        # Steps 3 & 4: Apply morphological operations and compute the difference mask
        mask_diff = morphological_difference(mask_cube, kernel_size=10)
        
        # Steps 5, 6 & 7: Accumulate temperature data, plot the histogram, and mark the cutoff
        
        output_file = os.path.join(args.output_root, f'{current_timestamp}.png')

        ratio, temp_1d = plot_temperature_histogram(temp_cube, mask_diff, output_file)
        ratio_dict[current_timestamp] = ratio
        temp_dict[current_timestamp] = temp_1d


    output_file = os.path.join(args.output_root, 'ratio-evolution.png')
    plot_ratio(temp_dict, ratio_dict, output_file)
    print(f"Done. Plot saved at: {args.output_file}")

if __name__ == "__main__":
    main()
