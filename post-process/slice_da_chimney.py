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

def create_2d_plane_mask(x1, y1, x2, y2, z_range, resolution):
    """
    Create a 2D plane based on the user-defined line (x1, y1 -> x2, y2) and z-axis.
    Returns 2D arrays of X, Y, and Z coordinates.
    """
    # Generate points along the user-defined line
    
    x_vals = np.linspace(x1, x2, resolution)# np.abs(x2 - x1))
    y_vals = np.linspace(y1, y2, resolution) #np.abs(y2 - y1))

    # Create a meshgrid for Z-axis and line points
    z_vals = np.linspace(z_range[0], z_range[1], resolution)
    z_grid, line_grid = np.meshgrid(z_vals, np.arange(resolution), indexing='ij')

    # Repeat X and Y for each Z coordinate
    x_plane = np.tile(x_vals, (len(z_vals), 1))
    y_plane = np.tile(y_vals, (len(z_vals), 1))

    return x_plane, y_plane, z_grid

    
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
        
    # DEBUG
    print(f"Max velocity: {np.max(reduced_velocity_plane)}")

    return reduced_velocity_plane, rand_row_indices, rand_col_indices

    
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

    while True:
        cv.imshow("Image", img_copy)
        cv.setMouseCallback("Image", click_event)
        cv.waitKey(0)

        if len(points) == 2:
            cv.destroyAllWindows()
            x1, y1 = points[0]
            x2, y2 = points[1]
            slope, intercept = linregress([x1, x2], [y1, y2])[:2]

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
    plt.figure(figsize =(24, 16))

    # Subplot 1: Slice with the confirmed line
    plt.subplot(1, 3, 1)
    plt.imshow(img, origin='lower', cmap='viridis')
    plt.plot([x1, x2], [y1, y2], color='red', label='Selected Line')
    plt.title(f'{args.image_channel} slice at z={args.center_z}')
    plt.legend()

    # Subplot 2: Filtered density and velocity
    plt.subplot(1, 3, 2)
    
    z_range = (args.lower_bound, args.upper_bound)
    resolution = args.pixel_boundary

    # Create 2D plane mask
    x_plane, y_plane, z_plane = create_2d_plane_mask(x1, y1, x2, y2, z_range, resolution)

    # Ensure integer indices for indexing the cube
    x_plane_idx = np.clip(x_plane.astype(int), 0, dens_cube.shape[0] - 1)
    y_plane_idx = np.clip(y_plane.astype(int), 0, dens_cube.shape[1] - 1)
    z_plane_idx = np.clip(z_plane.astype(int), 0, dens_cube.shape[2] - 1)

    # Extract 2D density and velocity planes
    density_plane = np.log10(channel_data[args.image_channel][x_plane_idx, y_plane_idx, z_plane_idx])
    velocity_plane, Y, X = scale_down_velocity(velz_cube[x_plane_idx, y_plane_idx, z_plane_idx], stride=40)
    
    
    # DEBUG
    print(f"Non zero in velocity: {np.count_nonzero(velocity_plane)}")
    print(f"velocity shape: {velocity_plane[20] * 100}")


    # Plot the density plane
    plt.imshow(density_plane, origin="lower", cmap="viridis", extent=(x1, x2, y1, y2)) #z_range[0], z_range[1])) # y1, y2))
    plt.colorbar(label="Density (g/cm³)")
    plt.title("Density Plane with Velocity Arrows")
    plt.xlabel("X (pixels)")
    plt.ylabel("Z (pixels)")

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

    # DEBUG
    print(f"Before filter: {X.shape}")
    print(f"After filter: {X_nonzero.shape}")
    
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
    
    plt.subplot(1, 3, 3)
    plt.imshow(velz_cube[x_plane_idx, y_plane_idx, z_plane_idx], origin="lower", cmap="RdBu", extent=(x1, x2, y1, y2))
    plt.colorbar(label="Velocity ($m/s^2$)")
    plt.title("Velocity profile")
    plt.xlabel("X (pixels)")
    plt.ylabel("Z (pixels)")


    
    if args.save:
        save_file = os.path.join(args.output_root, f"{args.image_channel}_chimney_z{args.center_z}.jpg")
        plt.savefig(save_file)
        print(f"Done! Plot saved at: {save_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
