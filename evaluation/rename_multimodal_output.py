import os
import argparse


def rename_files(target_dir):
    # Iterate over every file in the target directory
    for filename in os.listdir(target_dir):
        # Check if the filename contains an underscore
        if '_' in filename:
            # Split the filename at '_' and use the first part plus the extension
            new_name = filename.split('_')[0] + ".nii.gz"
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == '__main__':
    # Specify your target directory here or prompt the user
    parser = argparse.ArgumentParser(description="Renaming multimodal output, to match the evaluation code input")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing predicted images (.nii.gz)")
    args = parser.parse_args()
    
    
    # Check if the given directory exists
    if os.path.isdir(args.target_dir):
        rename_files(args.target_dir)
    else:
        print("The specified directory does not exist.")

# python rename_multimodal_output.py --target_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/astro-unetr-multimodal-epoch100/masks-output