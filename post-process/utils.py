import pandas as pd
import os
import matplotlib.pyplot as plt
import yt
import numpy as np
import os 
import matplotlib.colors as colors
from glob import glob
import cv2

# low_x0, low_y0, low_w, low_h, bottom_z, top_z = 0, 0, 256, 256, 0, 256 # -500, -500, 1000, 1000, -500, 500

top_z = 500

def get_velz_dens(obj, x_range, y_range, z_range):
    # read a 3D grid of velz and density array
    velz = obj["flash", "velz"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value        
    dens = obj["flash", "dens"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('g/cm**3').value        
    temp = obj["flash", "temp"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('K').value 

    print(f"obj.shape: {obj['flash', 'velz'].shape}")
    print("x, y, z ranges: ", x_range, y_range, z_range)
    print(f"velz.shape: {velz.shape}\tdens.shape: {dens.shape}\n\n")      
     

    dz = obj['flash', 'dz'][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value # proton mass

    # calculate the density as column density
    coldens = dens * dz / (1.4 * mp)

    return velz, coldens, temp

def get_velx_vely(obj, x_range, y_range, z_range):
    # read a 3D grid of vely and density array
    vely = obj["flash", "vely"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value
    velx = obj["flash", "velx"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value
    
    print(f"velx.shape: {velx.shape}\tvely.shape: {vely.shape}\n\n")      
    
    return velx, vely

# convert seconds to Megayears
def seconds_to_megayears(seconds):
    return seconds / (1e6 * 365 * 24 * 3600)

def pc2pixel(coord, x_y_z):
    if x_y_z == "x":
        return coord + top_z
    elif x_y_z == "y":
        return top_z - coord
    elif x_y_z == "z":
        return coord + top_z
    return coord

def pixel2pc(coord, x_y_z):
    if x_y_z == "x":
        return coord - top_z
    elif x_y_z == "y":
        return top_z - coord
    elif x_y_z == "z":
        return coord - top_z
    return coord

def cm2pc(cm):
    return cm * 3.24077929e-19

# filter the DataFrame
# def filter_data(df, range_coord):
#     return df[(df['posx_pc'] > range_coord[0]) & (df['posx_pc'] < range_coord[0] + range_coord[2]) & (df['posy_pc'] > range_coord[1]) & (df['posy_pc'] < range_coord[1] + range_coord[3]) & (df['posz_pc'] > range_coord[4] & (df['posz_pc'] < range_coord[5]))]

def timestamp2Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

def time_Myr2timestamp(time_Myr):
    return round(10 * (time_Myr - 191) + 200)

def pix_256_2pc(pix_256):
    return pix_256 * (1000 / 256)

def pc2pix_256(pc):
    return pc * (256 / 1000)


def read_SNfeedback(hdf5_root, filename):
    dat_files = glob(os.path.join(hdf5_root, filename))

    all_data = pd.DataFrame()

    # Read and concatenate data from all .dat files
    for dat_file in dat_files:
        # Assuming space-separated values in the .dat files
        df = pd.read_csv(dat_file, delim_whitespace=True, header=None,
                        names=['n_SN', 'type', 'n_timestep', 'n_tracer', 'time',
                                'posx', 'posy', 'posz', 'radius', 'mass'])
        
        # Convert the columns to numerical
        df = df.iloc[1:]
        df['n_SN'] = df['n_SN'].map(int)
        df['type'] = df['type'].map(int)
        df['n_timestep'] = df['n_timestep'].map(int)
        df['n_tracer'] = df['n_tracer'].map(int)
        df['time'] = pd.to_numeric(df['time'],errors='coerce')
        df['posx'] = pd.to_numeric(df['posx'],errors='coerce')
        df['posy'] = pd.to_numeric(df['posy'],errors='coerce')
        df['posz'] = pd.to_numeric(df['posz'],errors='coerce')
        df['radius'] = pd.to_numeric(df['radius'],errors='coerce')
        df['mass'] = pd.to_numeric(df['mass'],errors='coerce')
        all_data = pd.concat([all_data, df], ignore_index=True)
        all_data = all_data.drop(df[df['n_tracer'] != 0].index)


    # Convert time to Megayears
    all_data['time_Myr'] = seconds_to_megayears(all_data['time'])

    # Convert 'pos' from centimeters to parsecs
    all_data['posx_pc'] = cm2pc(all_data['posx'])
    all_data['posy_pc'] = cm2pc(all_data['posy'])
    all_data['posz_pc'] = cm2pc(all_data['posz'])

    # Sort the DataFrame by time in ascending order
    all_data.sort_values(by='time_Myr', inplace=True)

    return all_data

def filter_data(df, range_coord):
    return df[(df['posx_pc'] > range_coord[0]) & (df['posx_pc'] < range_coord[0] + range_coord[2]) & 
              (df['posy_pc'] > range_coord[1]) & (df['posy_pc'] < range_coord[1] + range_coord[3]) & 
              (df['posz_pc'] > range_coord[4]) & (df['posz_pc'] < range_coord[5])]


def apply_otsus_thresholding(image):
    # _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # return threshold
    _, binary_image = cv2.threshold(image.astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary_image)

def within_range(min, max, target):
    if min < target and max > target:
        return True
    return False

def SN_center_in_bubble(posx_px, posy_px, x1, y1, w, h):
    if within_range(x1, x1 + w, posx_px) and within_range(y1, y1 + h, posy_px):
        return True
    return False

def normalize4thresholding(arr):
    slice = np.log10(arr)
    return ((slice - np.min(slice)) / (np.max(slice) - np.min(slice))) * 255 

def plot_dens_z(dens, center_z):
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(dens[:, :, center_z].T[::]), cmap='viridis', aspect='auto')
    fig.colorbar(im, label='density ($g*cm^{-2}$)')
    plt.title('Density ($g*cm^{-2}$)')
    plt.xlabel('X')
    plt.ylabel('Y')
    # fig.savefig(f'../expanding_velocity/{time_Myr}/dzoom_{center_z}.png')
    plt.show()

def update_pos_pix256(filtered_data):
    converted_points = list(zip(
        pc2pix_256(filtered_data['posx_pc']) + 128,
        pc2pix_256(filtered_data['posy_pc']) + 128,
        pc2pix_256(filtered_data['posz_pc']) + 128
    ))
    # Converting the list of tuples into separate lists
    posx_pix256, posy_pix256, posz_pix256 = zip(*converted_points)

    # Adding the new columns to the DataFrame
    filtered_data['posx_pix256'] = posx_pix256
    filtered_data['posy_pix256'] = posy_pix256
    filtered_data['posz_pix256'] = posz_pix256

    return converted_points, filtered_data

def segment_cube_roi(args, dens_cube, mask_cube):
    for current_z in range(args.upper_bound - args.lower_bound):
        dens_slice = normalize4thresholding(dens_cube[:, :, current_z + args.lower_bound]) 
        
        # threshold + connected component
        binary_mask = apply_otsus_thresholding(dens_slice)
        _, labels, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # retrieve all mask only
        i = 0
        binary_mask = labels == i
        binary_mask = ~binary_mask
        mask_cube[:, :, current_z] = binary_mask

    return mask_cube