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

def create_2d_plane_mask(x1, y1, x2, y2, z_range, resolution):
    """
    Create a 2D plane based on the user-defined line (x1, y1 -> x2, y2) and z-axis.
    Returns 2D arrays of X, Y, and Z coordinates.
    """
    # Generate points along the user-defined line
    
    x_vals = np.linspace(x1, x2, resolution, dtype=int)# np.abs(x2 - x1))
    y_vals = np.linspace(y1, y2, resolution, dtype=int) #np.abs(y2 - y1))

    # Create a meshgrid for Z-axis and line points
    z_vals = np.linspace(z_range[0], z_range[1], resolution, dtype=int)
    z_plane, line_grid = np.meshgrid(z_vals, np.arange(resolution), indexing='ij')
    

    # Repeat X and Y for each Z coordinate
    x_plane = np.tile(x_vals, (len(z_vals), 1))
    y_plane = np.tile(y_vals, (len(z_vals), 1))

    return x_plane, y_plane, z_plane

    
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
    fig = plt.figure(figsize =(18, 12))
    # fig, axs = plt.subplots(1, 3, figsize =(24, 16))

    # Subplot 1: Slice with the confirmed line
    ax = fig.add_subplot(2, 3, 1)
    
    cv.addWeighted(redMask, 0.7, img_copy, 1, 0, img_copy)
    # im = ax.imshow(img_copy[:, :, 2], origin='lower', cmap='viridis')
    im = ax.imshow(np.log10(channel_data[args.image_channel][:,:,args.center_z]), origin='lower', cmap='viridis')
    print(f"img shape: {img.shape}")
    print(f"img_copy shape: {img_copy.shape}")
             
    ax.plot([x1, x2], [y1, y2], color='red', label='Selected Line')
    ax.set_title(f'{args.image_channel} slice at z={args.center_z}')
    ax.legend()
    fig.colorbar(im, label="Density (g/cm³)") #, shrink=0.75)

    # ------------------------------------------------------------------------------------------------------------------------
    # Subplot 2: Filtered density and velocity
    ax2 = fig.add_subplot(2, 3, 2)
    
    z_range = (args.lower_bound, args.upper_bound)
    resolution = args.pixel_boundary

    # Create 2D plane mask
    x_plane, y_plane, z_plane = create_2d_plane_mask(x1, y1, x2, y2, z_range, resolution)

    # Ensure integer indices for indexing the cube
    x_plane_idx = np.clip(x_plane.astype(int), 0, dens_cube.shape[0] - 1)
    y_plane_idx = np.clip(y_plane.astype(int), 0, dens_cube.shape[1] - 1)
    z_plane_idx = np.clip(z_plane.astype(int), 0, dens_cube.shape[2] - 1)

    # Extract 2D density and velocity planes
    density_plane = np.log10(channel_data[args.image_channel][x_plane_idx, y_plane_idx, z_plane_idx]) #[::-1, :])
    velocity_plane, Y, X = scale_down_velocity(velz_cube[x_plane_idx, y_plane_idx, z_plane_idx], stride=40)
    mask_plane = mask_cube[x_plane_idx, y_plane_idx, z_plane_idx]

    # --------------------------------------wuthout mask -------------------------------------------------
    # Plot the density plane
    im2 = ax2.imshow(density_plane, origin="lower", cmap="viridis", extent=(x1, x2, z_range[0], z_range[1]),    # [:,::-1]
                    vmin=np.min(img), vmax=np.max(img)) # y1, y2))
    fig.colorbar(im2, label="Density (g/cm³)") #, shrink=0.75)
    ax2.set_title("Sliced Density Plane")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Z (pixels)")
    ax2.axhline(y=args.center_z)
    
    # -------------------------------------------with mask ----------------------------------------------
    # Convert the single-channel density slice to RGB
    # density_rgb = np.stack([density_plane, density_plane, density_plane], axis=-1)

    # print(f"density shape: {np.max(density_rgb)}")
    # # Colorize the mask (set red channel where mask is non-zero)
    # red_overlay = np.zeros_like(density_rgb, dtype=np.float32)
    # red_overlay[:, :, 0] = (mask_plane > 0).astype(np.float32) * np.max(density_plane)  # Red channel for mask

    # # Blend the density and the red mask directly without normalizing the density slice
    # overlay = density_rgb.copy()
    # overlay[mask_plane > 0, 0] += np.max(density_plane) * 0.5  # Zero out blue channel in masked areas
    # overlay[mask_plane > 0, 1] = 0  # Zero out green channel in masked areas
    # overlay[mask_plane > 0, 2]  = 0  # Enhance red channel for masked areas

    # # Plot the blended image

    # im2 = ax2.imshow(overlay[:, :, 0], origin="lower")
    # ax2.set_title("Density Slice with Mask Overlay")
    # ax2.set_xlabel("X (pixels)")
    # ax2.set_ylabel("Z (pixels)")
    # fig.colorbar(im2, label="Density (g/cm³)")

    # ------------------------------------------------------------------------------------------------------------------------
    # Overlay velocity arrows
    Y, X = np.meshgrid(
        np.linspace(y1, y2, velocity_plane.shape[0]),
        np.linspace(x1, x2, velocity_plane.shape[1]),
    )

    # Filter out zero values in velocity_plane
    non_zero_mask = velocity_plane != 0
    X_nonzero = X[non_zero_mask]
    Y_nonzero = Y[non_zero_mask]
    U_nonzero = velocity_plane[non_zero_mask]  # Horizontal displacements (non-zero)
    V_nonzero = np.zeros_like(U_nonzero)  # No vertical displacement (still non-zero)

    
    # plt.quiver(X_nonzero, Y_nonzero,
    #     V_nonzero, # np.zeros_like(velocity_plane),  # No X-component for the velocity arrows
    #     U_nonzero, # velocity_plane * 100,
    #     angles="xy",
    #     scale_units="xy",
    #     scale=10,
    #     color="red",
    #     alpha=0.7,
    #     width=10
    # )
    
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(velz_cube[x_plane_idx, y_plane_idx, z_plane_idx], origin="lower", cmap="RdBu_r",   # [:,::-1]
                    extent=(x1, x2, z_range[0], z_range[1]),
                    vmin=-600, vmax=600)# y1, y2))
    fig.colorbar(im3, label="Velocity ($cm/s$)") #, shrink=0.5)
    ax3.set_title("Velocity profile")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Z (pixels)")
    ax3.axhline(y=args.center_z)
    
    
    ax5 = fig.add_subplot(2,3,5)
    # ax5.plot(dens_cube[x_plane_idx, y_plane_idx[0], args.center_z])
    
    ax5.plot(img[:, y_plane_idx[0][0]])
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
