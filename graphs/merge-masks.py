import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse


# python merge-masks.py --mask_root ~/Desktop/Dataset/MHD-3DIS/mask/ --output_root ~/Desktop/Dataset/MHD-3DIS/mask/small-bubble-merged


parser = argparse.ArgumentParser(description="Merging all masks for small bubble cases at the same z height to facilitate k3d html 3D visualization plotting.")
parser.add_argument("--mask_root", default="./Dataset", type=str, help="input mask directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")



def merge_mask_slices(root_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all object instance directories
    object_instances = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    # Iterate over each object instance to collect available timesteps
    timestep_dict = {}
    for obj in object_instances:
        obj_path = os.path.join(root_dir, obj)
        timesteps = [t for t in os.listdir(obj_path) if os.path.isdir(os.path.join(obj_path, t))]
        
        for t in timesteps:
            timestep_path = os.path.join(obj_path, t)
            image_files = sorted(os.listdir(timestep_path))  # Collect all z-slice filenames
            
            if t not in timestep_dict:
                timestep_dict[t] = set(image_files)
            else:
                timestep_dict[t].intersection_update(image_files)
    
    # Process each timestep and merge masks
    for t, image_filenames in tqdm(timestep_dict.items(), desc="Processing Timesteps"):
        timestep_output_dir = os.path.join(output_dir, t)
        os.makedirs(timestep_output_dir, exist_ok=True)
        
        for img_filename in tqdm(image_filenames, desc=f"Merging {t}", leave=False):
            merged_mask = None
            
            for obj in object_instances:
                timestep_path = os.path.join(root_dir, obj, t)
                img_path = os.path.join(timestep_path, img_filename)
                
                if not os.path.exists(img_path):
                    continue  # Skip if the file doesn't exist for this object
                
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                
                if merged_mask is None:
                    merged_mask = np.zeros_like(mask, dtype=np.uint8)
                
                merged_mask = cv2.bitwise_or(merged_mask, mask)  # OR operation
            
            if merged_mask is not None:
                output_path = os.path.join(timestep_output_dir, img_filename)
                cv2.imwrite(output_path, merged_mask)
                
    print("Merging complete. Merged masks saved to:", output_dir)

if __name__ == "__main__":
    args = parser.parse_args()

    merge_mask_slices(args.mask_root, args.output_root)

