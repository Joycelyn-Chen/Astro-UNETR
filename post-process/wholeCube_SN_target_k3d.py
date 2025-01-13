import os
import yt
import numpy as np
import k3d
from k3d import matplotlib_color_maps
import cv2 as cv

from utils import *
import argparse


DEBUG = True
hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'
low_x0, low_y0, low_w, low_h, bottom_z, top_z = 0, 0, 1000, 1000, 0, 1000
range_coord = [low_x0, low_y0, low_w, low_h, bottom_z, top_z]

# python post-process/wholeCube_SN_target_k3d.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -m ~/Desktop/Dataset/MHD-3DIS/SB_tracks/230 -k /home/joy0921/Desktop/Dataset/MHD-3DIS/htmls -st 380 -et 420

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

def segment_target_roi(args, mask_target, timestamp):
    # Directory containing masks for the current timestamp
    mask_dir = os.path.join(args.mask_root, str(timestamp))
    
    # Iterate through all .png mask files
    for root, _, files in os.walk(mask_dir):
        for mask_file in files:
            if mask_file.endswith(".png"):
                # Read the mask file in grayscale mode
                mask_slice = cv.imread(os.path.join(root, mask_file), cv.IMREAD_GRAYSCALE)
                
                # Resize mask_slice to the target size if necessary
                if mask_slice.shape != (args.pixel_boundary, args.pixel_boundary):
                    mask_slice = cv.resize(mask_slice, (args.pixel_boundary, args.pixel_boundary), interpolation=cv.INTER_NEAREST)
                
                # Convert grayscale mask to binary mask in-place for memory efficiency
                np.clip(mask_slice, 0, 255, out=mask_slice)  # Ensure valid pixel range
                mask_slice = (mask_slice > 0).astype(np.uint8)  # Convert >0 to 1, retain 0 as 0
                
                # Extract the z-coordinate from the mask filename
                z_coord = int(os.path.splitext(mask_file)[0])
                
                # Assign the binary mask slice to the appropriate position in the target 3D array
                mask_target[:, :, z_coord] = mask_slice
    
    return mask_target



def saving_k3d_plots(args, time_Myr, dens_cube_roi, dens_target_roi, converted_points):
    coords_whole_cube = np.argwhere(~np.isnan(dens_cube_roi))
    coords_target = np.argwhere(~np.isnan(dens_target_roi))

    values_whole_cube = np.log10(dens_cube_roi[coords_whole_cube[:, 0], coords_whole_cube[:, 1], coords_whole_cube[:, 2]])
    values_target = np.log10(dens_target_roi[coords_target[:, 0], coords_target[:, 1], coords_target[:, 2]])

    cube_points = k3d.points(positions=coords_whole_cube,
                            point_size=1,
                            shader='3d',
                            opacity=0.2,
                            color_map=matplotlib_color_maps.Viridis,
                            attribute=values_whole_cube,
                            ) # color=0x3f6bc5

    target_points = k3d.points(positions=coords_target,
                            point_size=1,
                            shader='3d',
                            opacity=1.0,
                            color_map=matplotlib_color_maps.Viridis,
                            attribute=values_target,
                            ) # color=0x3f6bc5

    SB_center = k3d.points(positions = converted_points, 
                            point_size=3.0,
                            shader='3d',
                            opacity=1.0,
                            color=0xc30010)     # original point: [149, 178, 141]

    plot = k3d.plot(grid=(0, 0, 0, 10, 10, 10),
                    axes=['X', 'Y', 'Z'])


    plot += cube_points
    plot += target_points
    plot += SB_center

    # plot.display()

    
    with open(os.path.join(args.k3d_root, f'{time_Myr}.html'),'w') as fp:
        fp.write(plot.get_snapshot())

    if(DEBUG):
        print("Done. Plot file stored at {}".format(f'{args.k3d_root}/{time_Myr}.html'))


def saving_SN_in_bound(args, time_Myr, filtered_data, mask_target):
    # record the SN in bounds
    # Open a text file to write the results
    temp_df = pd.DataFrame(columns=filtered_data.columns)

    # Loop through each row in filtered_data
    for _, row_data in filtered_data.iterrows():
        # Step 1: Read the 'posz_pix256' field
        posz_pix256 = int(row_data['posz_pix256'])
        
        # Step 2: Access the z slice in 3D array dens_target_roi
        if(args.lower_bound <= posz_pix256 < args.upper_bound):
            mask = mask_target[:, :, posz_pix256]
        else:
            continue
        
        # Step 3: Read 'posx_pix256' and 'posy_pix256' values as (x, y) coordinates
        posx_pix256 = int(row_data['posx_pix256'])
        posy_pix256 = int(row_data['posy_pix256'])

        
        # Step 4: Verify if the (x, y) value in the binary mask is white (value == 255)
        if mask[posy_pix256, posx_pix256] != 0:
            # Step 5: Output this row data into a .txt file
            temp_df = pd.concat([temp_df, pd.DataFrame([row_data])], ignore_index=True)

    if(not temp_df.empty):
        output_file = os.path.join(args.k3d_root, f"SN_{time_Myr}_info.txt")
        temp_df.to_csv(output_file, index=False, sep='\t')  
        # with open(os.path.join(args.k3d_root, f"SN_{time_Myr}_info.txt"), 'w') as f:
        #     f.write(f'{row_data.to_dict()}\n')



def main(args):
    for timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.incr):
        # Caculate the current time in Myr
        time_Myr = timestamp2Myr(timestamp) 

        # Input HDF5 raw data
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(hdf5_prefix, timestamp)))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))

        # retrieve the center (256, 256, 256) grid
        x_range_scaled = (args.lower_bound, args.upper_bound) 
        y_range_scaled = (args.lower_bound, args.upper_bound)  
        z_range_scaled = (args.lower_bound, args.upper_bound)

        # center_x, center_y, center_z = args.pixel_boundary // 2, args.pixel_boundary // 2, args.pixel_boundary // 2
        velz_cube, dens_cube, temp_cube = get_velz_dens(obj, x_range_scaled, y_range_scaled, z_range_scaled)
        # new_velx, new_vely = get_velx_vely(obj, x_range_scaled, y_range_scaled, z_range_scaled)

        # Reading all the SN within the last Myr
        all_data = read_SNfeedback(hdf5_root=args.hdf5_root, filename=args.dat_filename)

        start_Myr, end_Myr = time_Myr - (0.1 * args.incr), time_Myr
        # Filter data based on specified conditions
        filtered_data = filter_data(all_data[(all_data['time_Myr'] >= start_Myr) & (all_data['time_Myr'] <= end_Myr)], range_coord=(-500, -500, 1000, 1000, -500, 500)) 

        converted_points, filtered_data = update_pos_pix256(filtered_data)

        # process all bubbles in the entire cube
        
        mask_cube = np.zeros((args.pixel_boundary, args.pixel_boundary, args.upper_bound - args.lower_bound))
        mask_cube = segment_cube_roi(args, dens_cube, mask_cube)

        # velz_cube_roi = np.where(mask_cube, velz_cube[:, :, args.lower_bound:args.upper_bound], np.nan)
        dens_cube_roi = np.where(mask_cube, dens_cube[:, :, args.lower_bound:args.upper_bound], np.nan)

        # Read all the masks slices into a 3D mask array
        mask_target = np.zeros((args.pixel_boundary, args.pixel_boundary, args.upper_bound - args.lower_bound))
        mask_target = segment_target_roi(args, mask_target, timestamp)

        # velz_roi = np.where(mask, velz_cube[:, :, lower_b:upper_b], np.nan)
        dens_target_roi = np.where(mask_target, dens_cube[:, :, args.lower_bound:args.upper_bound], np.nan)
        # velz_target_roi = np.where(mask_target, velz_cube[:, :, args.lower_bound:args.upper_bound], np.nan)

        # Visualize in 3D using k3d
        saving_k3d_plots(args, time_Myr, dens_cube_roi, dens_target_roi, converted_points)

        # Record SN in bound
        saving_SN_in_bound(args, time_Myr, filtered_data, mask_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='wholeCube_SN_target_k3d.py',
                    description='Model the low density area for the entire cube, label the SN, and point out the target bubble',
                    epilog='Contact Joycelyn if you dont understand')
    
    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.')       #  "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
    parser.add_argument('-m', '--mask_root', help='Input the root path to where mask files are stored.')        # '/home/joy0921/Desktop/Dataset/img_pix256/masks'
    parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type = int)                        # 206
    parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type = int)                            # 235
    parser.add_argument('-i', '--incr', help='The timestamp increment unit', default = 1, type = int)
    parser.add_argument('-df', '--dat_filename', help='Input the .dat filename', default="SNfeedback.dat")                              
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)
    parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default = 0, type = int)
    parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default = 256, type = int)
    parser.add_argument('-k', '--k3d_root', help='Input the root path to where the k3d plots should be stored')                # '/home/joy0921/Desktop/Dataset/img_pix256/k3d_html'
    # parser.add_argument('-', '--', help='')
 
    args = parser.parse_args()

    main(args)

