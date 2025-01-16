import os
import yt
import numpy as np
import k3d
import argparse

from utils import *

DEBUG = True
hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'

# python mag3d_SN_visualization.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -k /home/joy0921/Desktop/Dataset/MHD-3DIS/htmls -st 390 -et 520 -i 10

def get_magnetic_data(obj, x_range, y_range, z_range):
    """
    Retrieve magx, magy, magz, density, and temperature in a single function.
    """
    magx = obj["flash", "magx"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].value
    magy = obj["flash", "magy"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].value
    magz = obj["flash", "magz"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].value
    dens = obj["flash", "dens"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('g/cm**3').value
    temp = obj["flash", "temp"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('K').value

    dz = obj['flash', 'dz'][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value  # Proton mass
    coldens = dens * dz / (1.4 * mp)

    print(f"magx.shape: {magx.shape}, magy.shape: {magy.shape}, magz.shape: {magz.shape}")
    return magx, magy, magz, coldens, temp


def visualize_magnetic_field(magx, magy, magz, mask_cube, converted_points, html_root, time_Myr, mag_stride=40):
    """
    Visualize magnetic field using k3d vectors.
    """
    # Masking the magnetic components
    magx_masked = np.where(mask_cube, magx, np.nan)
    magy_masked = np.where(mask_cube, magy, np.nan)
    magz_masked = np.where(mask_cube, magz, np.nan)

    # Flatten and create 3D grid
    indices = np.argwhere(~np.isnan(magx_masked))
    origins = indices.astype(np.float32)

    if DEBUG:
        print(f"Original indices: {indices.shape}")

    # Randomly reduce the number of vectors to half
    if len(indices) > 1:
        selected_indices = np.random.choice(len(indices), size=len(indices) // mag_stride, replace=False)
        indices = indices[selected_indices]
        origins = origins[selected_indices]

    vectors = np.array([magx_masked[indices[:, 0], indices[:, 1], indices[:, 2]],
                        magy_masked[indices[:, 0], indices[:, 1], indices[:, 2]],
                        magz_masked[indices[:, 0], indices[:, 1], indices[:, 2]]]).T

    # Normalize vectors to the range [-1, 1]
    max_vals = np.abs(vectors).max(axis=0)
    vectors = vectors / max_vals

    if DEBUG:
        print(f"New vectors: {vectors.shape}")
        print(f"Normalized vectors[0]: {vectors[0]}")

    scale = 20.0  # Adjust as needed
    magnitude = np.linalg.norm(vectors, axis=1)
    colors = k3d.helpers.map_colors(magnitude, k3d.matplotlib_color_maps.Spectral, [])
    vec_colors = np.zeros(2 * len(colors))
    for i, c in enumerate(colors):
        vec_colors[2 * i] = c
        vec_colors[2 * i + 1] = c
    vec_colors = vec_colors.astype(np.uint32)

    fig = k3d.plot()
    vec = k3d.vectors(
        origins=origins - vectors / 2,
        vectors=vectors * scale,
        colors=vec_colors,
        use_head=True,
        head_size=8,
        line_width=0.1
    )
    fig += vec

    # Add explosion points
    SB_center = k3d.points(positions=np.array(converted_points, dtype=np.float32),
                           point_size=1.0,
                           shader='3d',
                           opacity=1.0,
                           color=0xc30010)
    fig += SB_center

    # fig.display()
    with open(os.path.join(html_root, f'{time_Myr}_mag.html'),'w') as fp:
        fp.write(fig.get_snapshot())

    if(DEBUG):
        print("Done. Plot file stored at {}".format(f'{html_root}/{time_Myr}_mag.html'))

def main(args):
    for timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.incr):
        time_Myr = timestamp2Myr(timestamp)
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(hdf5_prefix, timestamp)))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(args.pixel_boundary,) * 3)

        x_range_scaled = (args.lower_bound, args.upper_bound)
        y_range_scaled = (args.lower_bound, args.upper_bound)
        z_range_scaled = (args.lower_bound, args.upper_bound)

        magx, magy, magz, dens_cube, temp_cube = get_magnetic_data(obj, x_range_scaled, y_range_scaled, z_range_scaled)

        all_data = read_SNfeedback(hdf5_root=args.hdf5_root, filename=args.dat_filename)
        filtered_data = filter_data(all_data, range_coord=(-500, -500, 1000, 1000, -500, 500))
        converted_points, filtered_data = update_pos_pix256(filtered_data)

        mask_cube = np.zeros((args.pixel_boundary, args.pixel_boundary, args.upper_bound - args.lower_bound), dtype=bool)
        mask_cube = segment_cube_roi(args, dens_cube, temp_cube, mask_cube)

        visualize_magnetic_field(magx, magy, magz, mask_cube, converted_points, args.html_root, time_Myr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='wholeCube_SN_target_k3d.py',
        description='Model the low density area for the entire cube, label the SN, and point out the target bubble',
        epilog='Contact Joycelyn if you dont understand')

    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.')
    # parser.add_argument('-m', '--mask_root', help='Input the root path to where mask files are stored.')
    parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type=int)
    parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type=int)
    parser.add_argument('-i', '--incr', help='The timestamp increment unit', default=1, type=int)
    parser.add_argument('-df', '--dat_filename', help='Input the .dat filename', default="SNfeedback.dat")
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default=256, type=int)
    parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default=0, type=int)
    parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default=256, type=int)
    parser.add_argument('-k', '--html_root', help='Input the root path to where the k3d plots should be stored')
    parser.add_argument('-Tt', '--temp_thresh', help='The power of temperature threshold for region segmentation. (hot gas)', default=5.5, type=float)
    

    args = parser.parse_args()

    main(args)
