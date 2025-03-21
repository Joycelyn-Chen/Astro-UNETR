import yt
import os
import numpy as np
import argparse
import cv2 as cv

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_cube(obj, x_range, y_range, z_range, modality='dens'):
    # read a 3D grid of velz and density array
    if modality == 'dens':
        dens = obj["flash", "dens"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('g/cm**3').value
        dz = obj['flash', 'dz'][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('cm').value
        mp = yt.physical_constants.mp.value # proton mass

        # calculate the density as column density
        coldens = dens * dz / (1.4 * mp)
        return coldens
    
    elif modality == 'temp':
        temp = obj["flash", "temp"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('K').value 
        return temp
    
    elif modality == 'velz':
        velz = obj["flash", "velz"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value
        return velz
        

    # print(f"obj.shape: {obj['flash', 'velz'].shape}")
    # print("x, y, z ranges: ", x_range, y_range, z_range)
    # print(f"velz.shape: {velz.shape}\tdens.shape: {dens.shape}\n\n")      
    
    # return temp # velz, coldens, temp

def pix_256_2pc(pix_256):
    return pix_256 * (1000 / 256)

def main(args):
    for timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.offset):
        if (timestamp < 1000):
            filename = f"{args.file_prefix}0{timestamp}"
        else:
            filename = f"{args.file_prefix}{timestamp}"

        print(f"Processing timestamp: {timestamp}")

        # loading img data
        ds = yt.load(os.path.join(args.hdf5_root, filename))
        #ds.current_time, ds.current_time.to('Myr')
        # ad = ds.all_data()

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(args.xlim,args.ylim,args.zlim))

        x_range_scaled = (0, args.xlim) 
        y_range_scaled = (0, args.ylim)  
        z_range_scaled = (0, args.zlim)
        # velz_cube, dens_cube, temp_cube = get_velz_dens(obj, x_range_scaled, y_range_scaled, z_range_scaled)
        modality_cube = get_cube(obj, x_range_scaled, y_range_scaled, z_range_scaled, args.modality)
        # Saving img
        for z in range(int(args.zlim)):
            if args.modality in ['dens', 'temp']:
                img = np.log10(modality_cube[:,:,z])
                normalizedImg = ((img - np.min(img)) / (np.max(img) - np.min(img)) ) * 255 
            
            elif args.modality == 'velz':
                img = modality_cube[:, :, z]
                normalizedImg = (abs(img - np.min(img)) / (np.max(img) - np.min(img)) ) * 255


            cv.imwrite(os.path.join(ensure_dir(os.path.join(args.output_root, args.modality, str(timestamp))), f'{z}{args.extension}'), normalizedImg)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", help="The root directory for the hdf5 dataset")              # "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original"
    parser.add_argument("--output_root", help="The root directory for the img output")          # "/home/joy0921/Desktop/Dataset/200_360/200_360_png"
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_")                                                   # "sn34_smd132_bx5_pe300_hdf5_plt_cnt_"
    parser.add_argument("--start_timestamp", help="The starting timestamp for data range", type = int)          # 206  
    parser.add_argument("--end_timestamp", help="The end timestamp for data range", type = int)                 # 230
    parser.add_argument("--offset", help="The offset for incrementing tiemstamps", type = int, default = 1)             # 1   
    parser.add_argument("--xlim", help="Input xlim", type = int, default = 256)                                         # 256 
    parser.add_argument("--ylim", help="Input ylim", type = int, default = 256)                                         # 256  
    parser.add_argument("--zlim", help="Input zlim", type = int, default = 256)                                         # 256
    parser.add_argument("--extension", help="Input the image extension (.jpg, .png)", default=".jpg")       # ".jpg"
    parser.add_argument("--modality", help="Input the image modality (dens, temp, velz)", default="dens")

    args = parser.parse_args()
    main(args)

# python hdf5tojpg.py --hdf5_root /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/imgs --start_timestamp 206 --end_timestamp 359 --offset 1 --modality temp
