import yt
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
# from scipy.stats import linregress

# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

# python slice_da_chimney.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -cz 170 -t 420 -o /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/chimneys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.')
    parser.add_argument('-mr', '--mask_root', help='Input the root path to where mask files are stored.')
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


def scale_down_velocity(velocity_plane, stride=40):
    """
    Reduce the effective number of points in the velocity_plane while keeping the output size the same.
    Randomly select points and set the rest to zero.
    """
    rows, cols = velocity_plane.shape
    num_points = velocity_plane.size // stride  # Adjust fraction as needed
    rand_indices = np.random.choice(rows * cols, size=num_points, replace=False)

    # Convert flattened indices to 2D coordinates
    rand_row_indices, rand_col_indices = np.unravel_index(rand_indices, (rows, cols))

    # Create a new velocity_plane of the same size filled with zeros
    reduced_velocity_plane = np.zeros_like(velocity_plane)

    # Assign selected points to the reduced_velocity_plane
    reduced_velocity_plane[rand_row_indices, rand_col_indices] = velocity_plane[rand_row_indices, rand_col_indices]

    # Normalize the reduced velocity plane
    max_val = np.abs(reduced_velocity_plane).max()
    if max_val != 0:
        reduced_velocity_plane /= max_val

    return reduced_velocity_plane, rand_row_indices, rand_col_indices

def read_mask_slices(mask_root, cube_shape):
    """
    Reads the mask cube from `mask_root` and saves each slice along the z-axis as an image.
    Filenames are the z-coordinates (e.g., '0.png', '1.png', ...).
    """
    mask_cube = np.zeros(cube_shape)
    
    for z in range(cube_shape[2]):
        mask_cube[z] = cv.imread(os.path.join(mask_root, f"{z}.png"), cv.IMREAD_GRAYSCALE)

    return mask_cube

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
    velx_cube, vely_cube, velz_cube, dens_cube, temp_cube = get_velocity_data(obj, (args.lower_bound, args.upper_bound), (args.lower_bound, args.upper_bound), (args.lower_bound, args.upper_bound))


    # Read the mask center slice
    mask_img = cv.imread(os.path.join(args.mask_root, str(args.timestamp), f"{str(args.center_z)}.png" ), cv.IMREAD_GRAYSCALE)
    mask_cube = read_mask_slices(os.path.join(args.mask_root, str(args.timestamp)), dens_cube.shape)
    
    
    # Extract center slice
    channel_data = {
        'velx': velx_cube,
        'vely': vely_cube,
        'velz': velz_cube,
        'dens': dens_cube,
        'temp': temp_cube
    }
    img = np.log10(channel_data[args.image_channel][:, :, args.center_z])
    
    # Show the image and select points
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            cv.imshow("Image", img_display)

    img_display = (img - img.min()) / (img.max() - img.min()) * 255
    img_display = img_display.astype(np.uint8)
    img_display = cv.merge([img_display, img_display, img_display])  # Duplicate to 3 channels (BGR format)
    img_copy = img_display.copy()  # Create a copy for interactive modifications
    
    redImg = np.zeros(img_copy.shape, img_copy.dtype)
    redImg[:,:] = (0, 0, 255)
    
    # DBUG
    print(f"mask max: {np.max(mask_cube[args.center_z])}\n\n")
    print(f"mask cube shape: {mask_cube.shape}")
    
    redMask = cv.bitwise_and(redImg, redImg, mask=mask_img )#(mask_cube[args.center_z]/255))
    cv.addWeighted(redMask, 0.7, img_copy, 1, 0, img_copy)

    while True:
        cv.imshow("Image", img_copy)
        cv.setMouseCallback("Image", click_event)
        cv.waitKey(0)

        if len(points) == 2:
            cv.destroyAllWindows()
            x1, y1 = points[0]
            x2, y2 = points[1]
            # slope, intercept = linregress([x1, x2], [y1, y2])[:2]

            # Draw the confirmed line
            img_copy = img_display.copy()  # Reset the display to the original image
            cv.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv.imshow("Image", img_copy)
            cv.waitKey(0)
            
            key = input("Are you satisfied with the line? (y/n): ")
            if key.lower() == 'y':
                cv.destroyAllWindows()
                break
            else:
                points = []  # Clear points to allow retry

    
    # Plot results
    fig = plt.figure(figsize =(16, 10))
    # fig, axs = plt.subplots(1, 3, figsize =(24, 16))

    # Subplot 1: Slice with the confirmed line
    ax = fig.add_subplot(2, 3, 1)
    
    cv.addWeighted(redMask, 0.7, img_copy, 1, 0, img_copy)
    # im = ax.imshow(img_copy[:, :, 2], origin='lower', cmap='viridis')
    im = ax.imshow(np.log10(channel_data[args.image_channel][:,:,args.center_z]), origin='lower', cmap='viridis')
    
             
    ax.plot([x1, x2], [y1, y2], color='red', label='Selected Line')
    ax.set_title(f'{args.image_channel} slice at z={args.center_z}')
    ax.legend()
    fig.colorbar(im, label="Density (g/cm³)") #, shrink=0.75)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    

    # ------------------------------------------------------------------------------------------------------------------------
    # Subplot 2: Filtered density and velocity
    ax2 = fig.add_subplot(2, 3, 2)
    
    z_range = (args.lower_bound, args.upper_bound)
    
    # ---------------------back projection -----------------------
    img_d = np.zeros((np.abs(x2 - x1), 256))
    img_v = np.zeros((np.abs(x2 - x1), 256))
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1 

    for z in range(args.lower_bound, args.upper_bound):
        for x in range(x1, x2):
            # y = int(slope * x + intercept)  # nearest neighbor
            y = 128
            img_d[x - x1, z] = np.log10(dens_cube[x, y, z])
            img_v[x - x1, z] = velz_cube[x, y, z]


    # --------------------------------------without mask -------------------------------------------------
    # Plot the density plane
    im2 = ax2.imshow(img_d.T[::], origin="lower", cmap="viridis", extent=(x1, x2, z_range[0], z_range[1]),    # [:,::-1]
                    vmin=np.min(img_d), vmax=np.max(img_d)) # y1, y2))
    fig.colorbar(im2, label="Density (g/cm³)") #, shrink=0.75)
    ax2.set_title("Sliced Density Plane")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Z (pixels)")
    ax2.axhline(y=args.center_z)

    # ------------------------------------------------------------------------------------------------------------------------
    
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(img_v.T[::], origin="lower", cmap="RdBu_r",   # [:,::-1]
                    extent=(x1, x2, z_range[0], z_range[1]),
                    vmin=-600, vmax=600)# y1, y2))
    fig.colorbar(im3, label="Velocity ($cm/s$)") #, shrink=0.5)
    ax3.set_title("Velocity profile")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Z (pixels)")
    ax3.axhline(y=args.center_z)
    
    
    ax5 = fig.add_subplot(2,3,5)
    # ax5.plot(dens_cube[x_plane_idx, y_plane_idx[0], args.center_z])
    
    ax5.plot(img_d[:, args.center_z])
    ax5.set_yscale('log')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Density ($g/cm^3$)')

    
    if args.save:
        save_file = os.path.join(args.output_root, f"{args.image_channel}_chimney_z{args.center_z}.jpg")
        plt.savefig(save_file)
        print(f"Done! Plot saved at: {save_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
