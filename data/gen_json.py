import os
import json
import argparse

parser = argparse.ArgumentParser(description="Generating json file linking all the input data as input to Swin-UNETR")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--output_file", default="test.json", type=str, help="output filename")

# python gen_json.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_file MHD-NII.json

def generate_mhd_json(data_dir, output_file):
    data_structure = {"training": []}

    # Define paths for imgs and masks directories
    imgs_dir = os.path.join(data_dir, "imgs")
    masks_dir = os.path.join(data_dir, "masks")

    fold_counts = {0: 0, 1: 0}  # Dictionary to track fold counts

    # Create the list of timesteps:
    # Range 206 to 359 (inclusive) with step 1
    # Range 380 to 1900 (inclusive) with step 10
    timesteps = list(range(206, 360)) + list(range(380, 1901, 10))

    for t in timesteps:
        ts = str(t)
        # Construct full paths for the three modalities
        density_file = os.path.join(imgs_dir, f"{ts}_d.nii.gz")
        temperature_file = os.path.join(imgs_dir, f"{ts}_t.nii.gz")
        velocity_file = os.path.join(imgs_dir, f"{ts}_v.nii.gz")
        
        # Check if all three modality files exist
        if not (os.path.exists(density_file) and os.path.exists(temperature_file) and os.path.exists(velocity_file)):
            print(f"Warning: One or more modalities for timestep {ts} not found in imgs directory.")
            continue

        # Construct full path for the label (mask)
        label_file = os.path.join(masks_dir, f"{ts}.seg.nii.gz")
        if not os.path.exists(label_file):
            print(f"Warning: Corresponding mask for timestep {ts} not found in masks directory.")
            continue

        # Determine fold assignment based on timestamp (fold=1 if timestep is between 380 and 780, else fold=0)
        fold = 1 if 380 <= t <= 780 else 0
        fold_counts[fold] += 1

        # Construct relative paths (to be stored in the json file)
        rel_density = os.path.join("imgs", f"{ts}_d.nii.gz")
        rel_temperature = os.path.join("imgs", f"{ts}_t.nii.gz")
        rel_velocity = os.path.join("imgs", f"{ts}_v.nii.gz")
        rel_label = os.path.join("masks", f"{ts}.seg.nii.gz")

        data_structure["training"].append({
            "fold": fold,
            "image": [rel_density, rel_density, rel_temperature, rel_velocity],
            "label": rel_label
        })

    # Write the data structure to the output JSON file
    with open(output_file, "w") as json_file:
        json.dump(data_structure, json_file, indent=4)

    # Print fold counts
    print(f"Fold 1 has {fold_counts[1]} entries.")
    print(f"Fold 0 has {fold_counts[0]} entries.")

if __name__ == "__main__":
    args = parser.parse_args()
    generate_mhd_json(args.data_dir, os.path.join(args.data_dir, args.output_file))
    print(f"Generated {args.output_file} successfully."

