import os
import numpy as np
import yt
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import argparse

k = yt.physical_constants.kb
mu = 1.4
m_H = yt.physical_constants.mass_hydrogen

# python analysis/plot_vol_energy.py --mask_root /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230 --hdf5_root /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230

def count_white_pixels(image_path):
    """Count the number of white pixels in an image."""
    with Image.open(image_path) as img:
        # Convert image to grayscale and count pixels with a value of 255
        return sum(pixel == 255 for pixel in img.convert('L').getdata())

def erg_to_joule(energy_erg):
    """
    Convert energy from erg to Joules.
    
    Parameters:
    energy_erg (array-like): Energy values in ergs.
    
    Returns:
    array-like: Energy values in Joules.
    """

    return np.array(energy_erg) * 1e-7

def timestamp2time_Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191


def plot_energy_volume(timeMyrs, kinetic_energies, thermal_energies, total_energies, total_volumes, output_root):
    # Convert energies to Joules
    kinetic_energies_joule = erg_to_joule(kinetic_energies)
    thermal_energies_joule = erg_to_joule(thermal_energies)
    total_energies_joule = erg_to_joule(total_energies)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plotting energy data in Joules    
    ax1.plot(timeMyrs, kinetic_energies_joule, label='Kinetic Energy (J)', color='#7b7d7b', linestyle='dotted', linewidth=2, marker='o')    # #6A9C89
    ax1.plot(timeMyrs, thermal_energies_joule, label='Thermal Energy (J)', color='#969696', linestyle='dashed', linewidth=2, marker='o')    # #E1D7B7
    ax1.plot(timeMyrs, total_energies_joule, label='Total Energy (J)', color='#000000', linestyle='solid', linewidth=2.5, marker='o')  # #CD5C08
    
    # supernovae energy injection
    one_hot_explosion = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    E_sn = [num * 1e44 for num in one_hot_explosion]
    ax1.plot(timeMyrs, E_sn, 'r+') 

    # Set logarithmic scale for the y-axis
    ax1.set_yscale('log')
    
    # Adding labels and title
    ax1.set_xlabel('Time (Myr)', fontsize=22)
    ax1.set_ylabel('Energy (J)', fontsize=22, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.plot(timeMyrs, total_volumes, label='Total Volume ($pc^3$)', color='#CD5C08', linestyle='solid', linewidth=2.5, marker='o') # #DDB665
    
    # Set logarithmic scale and color for the volume axis
    ax2.set_yscale('log')
    ax2.set_ylabel('Volume ($pc^3$)', fontsize=22, color='#CD5C08')
    ax2.tick_params(axis='y', labelcolor='#CD5C08', labelsize=20)
    
    # Adding grid for better readability
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray')
    
    # Customizing the legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=20, loc='best', frameon=True, fancybox=True) #, shadow=True)

    
    # Adjusting layout for better spacing
    plt.tight_layout()
    
    # Save the figure to the specified output root
    plt.savefig(os.path.join(output_root, 'energy_volume.png'), dpi=300)

    print(f"Done. Plot saved at: {os.path.join(output_root, 'energy_volume.png')}")



def calc_energy_vol(args, hdf5_filename, root_dir, timestamp):
    ds = yt.load(hdf5_filename)

    center = [0, 0, 0] * yt.units.pc    
    arb_center = ds.arr(center, 'code_length')
    xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))

    timestamp_energy = {'kinetic_energy': 0, 'thermal_energy': 0, 'total_energy': 0, 'volume' : 0}

    mask_names = sorted(os.listdir(os.path.join(root_dir, str(timestamp))), key=lambda mask_name: int(mask_name.split('.')[0])) 

    for mask_name in mask_names:
        mask_img = cv.imread(os.path.join(root_dir, str(timestamp), mask_name), cv.IMREAD_GRAYSCALE)
        # coordinates = np.argwhere(mask_img == 255)
        mask_boolean = mask_img == 255

        z = int(mask_name.split('.')[0])
        # z = pixel2pc(int(mask_path.split(".")[-2].split("z")[-1]), x_y_z="z")

        temp = obj["flash", "temp"][:, :, z]
        n = obj["flash", "dens"][:, :, z] / (mu * m_H)

        rho = obj["flash", "dens"][:, :, z]
        v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

        cell_volume = obj["flash", "cell_volume"][:, :, z]

        kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
        thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')
        # total_energy = (kinetic_energy + thermal_energy).to('erg/cm**3')

        timestamp_energy['kinetic_energy'] += np.sum(kinetic_energy[mask_boolean])
        timestamp_energy['thermal_energy'] += np.sum(thermal_energy[mask_boolean])
        timestamp_energy['total_energy'] += np.sum(kinetic_energy[mask_boolean] + thermal_energy[mask_boolean])
        timestamp_energy['volume'] += count_white_pixels(os.path.join(root_dir, str(timestamp), mask_name))


    return timestamp_energy




def main(args):
    timestamps = os.listdir(args.mask_root)
    timestamps = [int(timestamp) for timestamp in sorted(timestamps) if os.path.isdir(os.path.join(args.mask_root, timestamp))] 
    energy_data = {}

    for timestamp in timestamps:
        if(timestamp < 300):
            continue
        #DEBUG
        print(f"Processing {timestamp}")
        hdf5_filename = os.path.join(args.hdf5_root, f"{args.file_prefix}{timestamp}")
        timestamp_energy = calc_energy_vol(args, hdf5_filename, args.mask_root, timestamp)
        energy_data[timestamp] = timestamp_energy


    # Plotting
    timestamps = list(energy_data.keys())
    kinetic_energies = [energy_data[timestamp]['kinetic_energy'] for timestamp in timestamps]
    thermal_energies = [energy_data[timestamp]['thermal_energy'] for timestamp in timestamps]
    total_energies = [energy_data[timestamp]['total_energy'] for timestamp in timestamps]
    total_volumes = [energy_data[timestamp]['volume'] * ((1000/256) ** 3) for timestamp in timestamps]          # converting volume from pixels to pc^3

    # filter out only the valid reading and convert timestamps to Myr
    filtered_keys = [key for key in energy_data.keys() if energy_data[key]['volume'] != 0]
    timeMyrs = [timestamp2time_Myr(x) for x in filtered_keys]
    # timeMyrs = [timestamp2time_Myr(x) for x in list(energy_data.keys())] 

    plot_energy_volume(timeMyrs, kinetic_energies, thermal_energies, total_energies, total_volumes, args.output_root)

    # Accumulated total energy
    print(f"Accumulated Kinetic Energy: {sum(kinetic_energies)} erg")
    print(f"Accumulated Thermal Energy: {sum(thermal_energies)} erg")
    print(f"Accumulated Total Energy: {sum(total_energies)} erg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--hdf5_root", help="The root directory to the dataset")
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)
    
  
    args = parser.parse_args()
    main(args)









