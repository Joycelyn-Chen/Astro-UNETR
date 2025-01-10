import os
import cv2
import numpy as np
from skimage import measure
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser(description="Converting semantic output from Swin-UNETR to instance level segmentation.")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--start_timestamp", type=int, required=True, help="Lower limit for timestamp range")
parser.add_argument("--end_timestamp", type=int, required=True, help="Upper limit for timestamp range")

# python semantic2instance.py --data_dir --start_timestamp --end_timestamp

def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two 3D binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def process_and_track_instances(masks_dir, tracks_dir, lower_limit, upper_limit):
    """
    Perform 3D connected component analysis and track instances over time.

    Args:
        masks_dir (str): Path to the directory containing timestamp folders with slices.
        tracks_dir (str): Path to save instance tracklets.
        start_timestamp (int): Lower bound of timestamp range to process.
        end_timestamp (int): Upper bound of timestamp range to process.
    """
    # Ensure output directory exists
    os.makedirs(tracks_dir, exist_ok=True)

    # Sorted list of timestamps
    timestamps = sorted([ts for ts in os.listdir(masks_dir) if lower_limit <= int(ts) <= upper_limit], key=lambda x: int(x))

    # Dictionary to hold active tracks
    active_tracks = {}

    for t, timestamp in enumerate(timestamps):
        print(f"Processing timestamp: {timestamp}")
        timestamp_path = os.path.join(masks_dir, timestamp)
        
        # Load the 3D segmentation cube from slices
        slices = [
            cv2.imread(os.path.join(timestamp_path, f"{z}.png"), cv2.IMREAD_GRAYSCALE)
            for z in range(256)
        ]
        seg_cube = np.stack(slices, axis=0)

        # Perform 3D connected component analysis
        labeled_cube, num_instances = measure.label(seg_cube, return_num=True, connectivity=1)

        # Track linkage for current timestamp
        used_instances = set()

        if t == 0:  # Initialize tracks for the first timestamp
            for instance_id in range(1, num_instances + 1):
                track_id = f"SB{timestamp}_{instance_id}"
                active_tracks[track_id] = (labeled_cube == instance_id).astype(np.uint8)

        else:  # Link instances to existing tracks
            new_tracks = {}

            for track_id, track_mask in active_tracks.items():
                max_iou = 0
                best_instance_id = None
                
                for instance_id in range(1, num_instances + 1):
                    if instance_id in used_instances:
                        continue

                    instance_mask = (labeled_cube == instance_id).astype(np.uint8)
                    iou = compute_iou(track_mask, instance_mask)

                    if iou > max_iou:
                        max_iou = iou
                        best_instance_id = instance_id

                if max_iou > 0:  # Link track to the best match
                    used_instances.add(best_instance_id)
                    new_tracks[track_id] = (labeled_cube == best_instance_id).astype(np.uint8)
                else:  # Close track if no match is found
                    track_output_dir = os.path.join(tracks_dir, track_id)
                    os.makedirs(track_output_dir, exist_ok=True)
                    for z in range(track_mask.shape[0]):
                        slice_mask = track_mask[z, :, :] * 255
                        slice_filename = os.path.join(track_output_dir, f"{z}.png")
                        imsave(slice_filename, slice_mask.astype(np.uint8))
                    print(f"Closed track: {track_id}")

            # Add new instances as new tracks
            for instance_id in range(1, num_instances + 1):
                if instance_id not in used_instances:
                    track_id = f"SB{timestamp}_{instance_id}"
                    new_tracks[track_id] = (labeled_cube == instance_id).astype(np.uint8)

            active_tracks = new_tracks

    # Save remaining active tracks
    for track_id, track_mask in active_tracks.items():
        track_output_dir = os.path.join(tracks_dir, track_id)
        os.makedirs(track_output_dir, exist_ok=True)
        for z in range(track_mask.shape[0]):
            slice_mask = track_mask[z, :, :] * 255
            slice_filename = os.path.join(track_output_dir, f"{z}.png")
            imsave(slice_filename, slice_mask.astype(np.uint8))
        print(f"Saved track: {track_id}")

if __name__ == "__main__":
    args = parser.parse_args()

    masks_dir = os.path.join(args.data_dir, 'masks')  # Directory containing semantic segmentation slices by timestamp
    tracks_dir = os.path.join(args.data_dir, 'SB_tracks')  # Directory to save tracked instances
    
    process_and_track_instances(masks_dir, tracks_dir, args.start_timestamp, args.end_timestamp)
    print("Processing and tracking complete.")
