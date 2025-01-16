import yt
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.stats import linregress

# python slice_da_chimney.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -cz 170 -t 420 -o /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/chimneys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.')
    parser.add_argument('-cz', '--center_z', help='The center z coordinate for the target bubble of interest.', default=0, type=int)
    parser.add_argument('-t', '--timestamp', help='Input the timestamp', type=int)
    parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default=0, type=int)
    parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default=256, type=int)
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default=256, type=int)
    parser.add_argument('-imgc', '--image_channel', help='Input the interested image channel to show.', default='dens', type=str)
    parser.add_argument('-o', '--output_root', help='Input the root path to where the plots should be stored')
    parser.add_argument("--save", action="store_true", help="Saving the results or not")
    return parser.parse_args()

# Velocity data retrieval function
def get_velocity_data(obj, x_range, y_range, z_range):
    velx = obj["flash", "velx"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    vely = obj["flash", "vely"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    velz = obj["flash", "velz"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    dens = obj["flash", "dens"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('g/cm**3').value
    temp = obj["flash", "temp"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('K').value
    dz = obj['flash', 'dz'][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value  # Proton mass
    coldens = dens * dz / (1.4 * mp)
    return velx, vely, velz, coldens, temp

# Main script
def main():
    args = parse_args()
    
    # Load dataset
    hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'
    ds = yt.load(os.path.join(args.hdf5_root, f"{hdf5_prefix}{args.timestamp}"))

    # Define grid parameters
    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(args.pixel_boundary,) * 3)

    # Retrieve data
    velx_cube, vely_cube, velz_cube, dens_cube, temp_cube = get_velocity_data(
        obj,
        (args.lower_bound, args.upper_bound),
        (args.lower_bound, args.upper_bound),
        (args.lower_bound, args.upper_bound)
    )

    # Extract center slice
    channel_data = {
        'velx': velx_cube,
        'vely': vely_cube,
        'velz': velz_cube,
        'dens': dens_cube,
        'temp': temp_cube
    }
    img = channel_data[args.image_channel][:, :, args.center_z]

    # Show the image and select points
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv.circle(img_display, (x, y), 5, (255, 0, 0), -1)
            cv.imshow("Image", img_display)

    while True:
        img_display = (img - img.min()) / (img.max() - img.min()) * 255
        img_display = img_display.astype(np.uint8)
        cv.imshow("Image", img_display)
        cv.setMouseCallback("Image", click_event)
        cv.waitKey(0)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            slope, intercept = linregress([x1, x2], [y1, y2])[:2]
            x_vals = np.arange(img.shape[1])
            y_vals = slope * x_vals + intercept
            for x, y in zip(x_vals.astype(int), y_vals.astype(int)):
                cv.circle(img_display, (x, y), 1, (0, 255, 0), -1)
            cv.imshow("Image", img_display)
            key = input("Are you satisfied with the line? (y/n): ")
            if key.lower() == 'y':
                break
            else:
                points = []

    # Expand line to plane and filter cube
    plane_mask = np.abs(np.outer(np.arange(img.shape[0]), slope) - np.arange(img.shape[1]) + intercept < 1)
    plane_mask = np.repeat(plane_mask[:, :, np.newaxis], dens_cube.shape[2], axis=2)

    # Filtered results
    filtered_dens = dens_cube[plane_mask]
    filtered_velz = velz_cube[plane_mask]

    # Plot results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, origin='lower', cmap='viridis')
    plt.plot([x1, x2], [y1, y2], color='red')
    plt.title(f'{args.image_channel} slice at z={args.center_z}')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_dens, origin='lower', cmap='viridis')
    plt.quiver(
        np.arange(filtered_dens.shape[0]),
        np.arange(filtered_dens.shape[1]),
        np.zeros_like(filtered_velz),
        filtered_velz,
        scale=1
    )
    plt.title('Filtered Plane with Velocity Z Arrows')
    
    if args.save:
        save_file = os.path.join(args.output_root, f"{args.image_channel}_chimney_z{args.center_z}.jpg")
        plt.savefig(save_file)
        print(f"Done! Plot saved at: {save_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
