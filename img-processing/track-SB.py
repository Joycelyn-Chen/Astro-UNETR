#!/usr/bin/env python3
"""
track_superbubble.py

Script to segment a specified dark region (bubble) and track it through image cubes across multiple timesteps.

Inputs:
  - dataset_root: root directory containing timestep subfolders, each with slice images (*.jpg)
  - output_root: root directory to write output masks (one folder per timestep, one .png per slice)
  - mid_slice: index (0-based) of the reference (middle) slice in each cube
  - init_x, init_y: pixel coordinates in the middle slice to select the target component
  - start_t, end_t: optional numeric timestep bounds to process

Usage:
  python track_superbubble.py --mid_slice 128 --init_x 100 --init_y 120
"""
import os
import cv2
import numpy as np
import argparse


from utils import apply_otsus_thresholding, find_connected_components

# python track-SB.py --dataset_root /home/joy0921/Desktop/Dataset/img_pix256/img --output_root /home/joy0921/Desktop/Dataset/Img_processing_output/SB230 --mid_slice 141 --init_x 178 --init_y 149 --start_t 380 --end_t 400

def segment_and_select(image_path, point):
    """
    Segment dark regions via Otsu and select the component containing `point`.
    Returns: (mask (uint8 0/255), centroid tuple or None)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    binary_mask, _ = apply_otsus_thresholding(image)
    # connectedComponentsWithStats returns (num, labels, stats, centroids)
    num_labels, labels, stats, centroids = find_connected_components(binary_mask)
    x, y = point
    for lbl in range(1, num_labels):
        left, top, w, h, area = stats[lbl]
        if left <= x < left + w and top <= y < top + h:
            comp_mask = (labels == lbl).astype(np.uint8) * 255
            return comp_mask, tuple(centroids[lbl])
    # no component found at point
    return np.zeros_like(binary_mask), None


def segment_and_associate(image_path, prev_mask):
    """
    Segment dark regions, then pick the component with maximal overlap with prev_mask.
    Returns: (mask, centroid)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    binary_mask, _ = apply_otsus_thresholding(image)
    num_labels, labels, stats, centroids = find_connected_components(binary_mask)
    best_lbl, best_ov = 0, 0
    # compute overlap in pixel count
    prev_bin = (prev_mask > 0)
    for lbl in range(1, num_labels):
        curr_bin = (labels == lbl)
        overlap = int(np.sum(prev_bin & curr_bin))
        if overlap > best_ov:
            best_ov, best_lbl = overlap, lbl
    if best_lbl > 0:
        mask = (labels == best_lbl).astype(np.uint8) * 255
        return mask, tuple(centroids[best_lbl])
    # fallback: empty mask
    return np.zeros_like(binary_mask), None


def track_cube(slice_paths, mid_idx, init_point):
    """
    For a single cube (list of slice image paths), segment the target at mid_idx,
    then propagate up and down.
    Returns dict: {slice_index: mask_image}
    """
    masks = {}
    # reference slice
    mask_mid, centroid = segment_and_select(slice_paths[mid_idx], init_point)
    masks[mid_idx] = mask_mid
    # upward (higher indices)
    prev = mask_mid
    for i in range(mid_idx + 1, len(slice_paths)):
        m, cent = segment_and_associate(slice_paths[i], prev)
        masks[i] = m
        prev = m
    # downward (lower indices)
    prev = mask_mid
    for i in range(mid_idx - 1, -1, -1):
        m, cent = segment_and_associate(slice_paths[i], prev)
        masks[i] = m
        prev = m
    return masks



def segment_and_track(dataset_root, output_root, mid_slice, init_point, start_t=None, end_t=None):
    """
    Loop over timesteps, track bubble in each cube, and save masks.
    """
    # list and sort timestep folders numerically
    tids = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    tids = sorted(tids, key=lambda x: int(x))
    # apply bounds
    if start_t is not None:
        tids = [t for t in tids if int(t) >= start_t]
    if end_t is not None:
        tids = [t for t in tids if int(t) <= end_t]
    current_pt = init_point
    for ts in tids:
        ts_path = os.path.join(dataset_root, ts)
        # gather and sort slice filenames
        files = [f for f in os.listdir(ts_path) if f.lower().endswith('.jpg')]
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        slice_paths = [os.path.join(ts_path, f) for f in files]
        # ensure mid_slice in range
        if not (0 <= mid_slice < len(slice_paths)):
            raise IndexError(f"mid_slice {mid_slice} out of range for timestep {ts}")
        # track within this cube
        masks = track_cube(slice_paths, mid_slice, current_pt)
        # write out
        out_ts = os.path.join(output_root, ts, 'mask')
        os.makedirs(out_ts, exist_ok=True)
        for idx, m in masks.items():
            out_name = os.path.splitext(files[idx])[0] + '.png'
            out_path = os.path.join(out_ts, out_name)
            cv2.imwrite(out_path, m)
        
        print(f"Done with t = {ts}, masks saved at: {out_ts}")

        # update init_point for next timestep based on centroid in mid slice
        mid_m = masks.get(mid_slice)
        M = cv2.moments(mid_m)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_pt = (cx, cy)
        # else keep previous point
    print("Tracking complete across all timesteps.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment and track a bubble through image cubes.")
    parser.add_argument('--dataset_root', type=str,
                        default='/home/joy0921/Dataset/img_pix256/img',
                        help='Root folder containing timestep subfolders of .jpg slices')
    parser.add_argument('--output_root', type=str,
                        default='/home/joy0921/Desktop/Dataset/Img_processing_output/SB230',
                        help='Where to save segmented masks')
    parser.add_argument('--mid_slice', type=int, required=True,
                        help='0-based index of the reference (middle) slice')
    parser.add_argument('--init_x', type=int, required=True,
                        help='X coordinate in middle slice to pick target')
    parser.add_argument('--init_y', type=int, required=True,
                        help='Y coordinate in middle slice to pick target')
    parser.add_argument('--start_t', type=int, default=None,
                        help='Optional start timestep (inclusive)')
    parser.add_argument('--end_t', type=int, default=None,
                        help='Optional end timestep (inclusive)')
    args = parser.parse_args()
    segment_and_track(
        args.dataset_root,
        args.output_root,
        args.mid_slice,
        (args.init_x, args.init_y),
        args.start_t,
        args.end_t
    )
 