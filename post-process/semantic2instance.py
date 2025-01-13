import os
import numpy as np
from skimage import measure
from skimage.io import imsave
import cv2
import argparse

parser = argparse.ArgumentParser(description="Converting semantic output from Swin-UNETR to instance level segmentation.")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--start_timestamp", type=int, required=True, help="Lower limit for timestamp range")
parser.add_argument("--end_timestamp", type=int, required=True, help="Upper limit for timestamp range")

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
        lower_limit (int): Lower bound of timestamp range to process.
        upper_limit (int): Upper bound of timestamp range to process.
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

        # Filter small instances
        instance_sizes = [(labeled_cube == i).sum() for i in range(1, num_instances + 1)]
        valid_instances = [i for i, size in enumerate(instance_sizes, start=1) if size >= 80000]

        # Track linkage for current timestamp
        used_tracks = set()
        new_tracks = {}

        if t == 0:  # Initialize tracks for the first timestamp
            for instance_id in valid_instances:
                track_id = f"SB{timestamp}_{instance_id}"
                active_tracks[track_id] = (labeled_cube == instance_id).astype(np.uint8)
                track_output_dir = os.path.join(tracks_dir, track_id, timestamp)
                os.makedirs(track_output_dir, exist_ok=True)
                
                # Save slices for the track
                track_mask = active_tracks[track_id]
                for z in range(track_mask.shape[0]):
                    slice_mask = track_mask[z, :, :] * 255
                    slice_filename = os.path.join(track_output_dir, f"{z}.png")
                    imsave(slice_filename, slice_mask.astype(np.uint8))

        else:  # Link instances to existing tracks
            for instance_id in valid_instances:
                instance_mask = (labeled_cube == instance_id).astype(np.uint8)

                # Find the best matching track
                max_iou = 0
                best_track_id = None

                for track_id, track_mask in active_tracks.items():
                    iou = compute_iou(track_mask, instance_mask)
                    if iou > max_iou:
                        max_iou = iou
                        best_track_id = track_id

                if max_iou > 0:  # Link to existing track
                    used_tracks.add(best_track_id)
                    new_tracks[best_track_id] = instance_mask

                    # Save slices for the track
                    track_output_dir = os.path.join(tracks_dir, best_track_id, timestamp)
                    os.makedirs(track_output_dir, exist_ok=True)
                    for z in range(instance_mask.shape[0]):
                        slice_mask = instance_mask[z, :, :] * 255
                        slice_filename = os.path.join(track_output_dir, f"{z}.png")
                        imsave(slice_filename, slice_mask.astype(np.uint8))

            # Close tracks with no connections
            for track_id in set(active_tracks.keys()) - used_tracks:
                print(f"Closing track: {track_id}")

            # Open new tracks for unmatched instances
            for instance_id in valid_instances:
                instance_mask = (labeled_cube == instance_id).astype(np.uint8)

                # Check overlap with all existing tracks
                overlaps = [compute_iou(track_mask, instance_mask) for track_mask in new_tracks.values()]
                if all(overlap < 0.5 for overlap in overlaps):
                    track_id = f"SB{timestamp}_{instance_id}"
                    new_tracks[track_id] = instance_mask
                    track_output_dir = os.path.join(tracks_dir, track_id, timestamp)
                    os.makedirs(track_output_dir, exist_ok=True)

                    # Save slices for the new track
                    for z in range(instance_mask.shape[0]):
                        slice_mask = instance_mask[z, :, :] * 255
                        slice_filename = os.path.join(track_output_dir, f"{z}.png")
                        imsave(slice_filename, slice_mask.astype(np.uint8))

            active_tracks = new_tracks

if __name__ == "__main__":
    args = parser.parse_args()

    masks_dir = os.path.join(args.data_dir, 'masks')  # Directory containing semantic segmentation slices by timestamp
    tracks_dir = os.path.join(args.data_dir, 'SB_tracks')  # Directory to save tracked instances
    
    process_and_track_instances(masks_dir, tracks_dir, args.start_timestamp, args.end_timestamp)
    print("Processing and tracking complete.")
