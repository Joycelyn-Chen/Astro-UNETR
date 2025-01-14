import os
import numpy as np
import torch
import cv2 as cv
import argparse
import hydra
from sam2.build_sam import build_sam2_video_predictor
from utils import save_mask

# python instance-seg-point-prompt.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS --timestamp 410

def show_point(img, point, color=(0, 255, 0)):
    cv.circle(img, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)

def show_box(img, bbox, color=(0, 0, 255)):
    cv.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

def initialize_predictor(args, device):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=f"{args.sam2_root}/sam2/configs/sam2.1", version_base='1.2')

    sam2_checkpoint = os.path.join(args.sam2_root, "checkpoints/sam2.1_hiera_large.pt")
    model_cfg = "sam2.1_hiera_l.yaml"

    return build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def get_bounding_boxes(mask_dir):
    bbox_points = {}
    for mask_file in sorted(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        z = int(os.path.splitext(mask_file)[0])
        points = []
        if contours:
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                points.append((center_x, center_y))
            bbox_points[z] = points
    return bbox_points

def extract_random_slices(mask_dir):
    bbox_points = get_bounding_boxes(mask_dir)
    return bbox_points

def annotate_points_and_confirm(frame, z_coord, points):
    # for point in points:
        # show_point(frame, point)
    # cv.imshow(f"Annotated Frame - Slice {z_coord}", frame)
    # cv.waitKey(32)

    while True:
        key = input("Confirm points for slice {}? (y/n): ".format(z_coord))
        if key.lower() == 'y':
            # cv.destroyAllWindows()
            return points
        elif key.lower() == 'n':
            print(f"Current points: {points}")
            updated_points = []
            num_points = int(input("Enter the number of points to update: "))
            for i in range(num_points):
                x = int(input(f"Enter x coordinate for point {i+1}: "))
                y = int(input(f"Enter y coordinate for point {i+1}: "))
                updated_points.append((x, y))
            points = updated_points
            # frame_copy = frame.copy()
            # for point in points:
            #     show_point(frame_copy, point)
            # cv.imshow(f"Updated Annotated Frame - Slice {z_coord}", frame_copy)
            # cv.waitKey(32)

def main():
    parser = argparse.ArgumentParser(description="3D Instance Tracking using SAM2")
    parser.add_argument("--data_dir", default="/home/joycelyn/Desktop/Dataset/MHD-3DIS", type=str, help="Input data directory")
    parser.add_argument("--SB_ID", default=230, type=int, help="SB ID for the target bubble")
    parser.add_argument("--center_x", default=150, type=int, help="Center x of the target bubble")
    parser.add_argument("--center_y", default=178, type=int, help="Center y of the target bubble")
    parser.add_argument("--center_z", default=130, type=int, help="Center z of the target bubble")
    parser.add_argument("--timestamp", default=380, type=int, help="Timestamp of interest")
    parser.add_argument("--time_interval", default=10, type=int, help="Timestamp increment interval")
    parser.add_argument("--sam2_root", default="/home/joycelyn/Desktop/sam2", type=str, help="Path to SAM2 root directory")
    parser.add_argument("--interactive_confirmation", action="store_true", help="Enable interactive point confirmation prompting")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    predictor = initialize_predictor(args, device)
    

    current_dir = os.path.join(args.data_dir, f"masks-jpg/{args.timestamp}")
    previous_dir = os.path.join(args.data_dir, "SB_tracks", str(args.SB_ID), f"{args.timestamp - args.time_interval}")
    bbox_points = extract_random_slices(previous_dir)

    inference_state = predictor.init_state(video_path=current_dir)

    # Add negative click at (5, 5)
    points = np.array([[5, 5]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels
    )

    # for z_coord, points in bbox_points.items():
    #     frame_path = os.path.join(current_dir, f"{z_coord}.jpg")
    #     frame = cv.imread(frame_path)
    #     #DEBUG
    #     print(f"Reading from {frame_path}, points: {points}")
    #     if frame is None:
    #         print(f"Frame {z_coord} not found.")
    #         continue

    #     confirmed_points = annotate_points_and_confirm(frame, z_coord, points)

    #     for point in confirmed_points:
    #         predictor.add_new_points_or_box(
    #             inference_state=inference_state,  
    #             frame_idx=z_coord,
    #             obj_id=args.SB_ID,
    #             points=np.array([point], dtype=np.float32),
    #             labels=np.array([1], dtype=np.int32)
    #         )
    # Updated part of the code
    for z_coord, points in bbox_points.items():
        # Only process every other 10 z slices
        if z_coord % 10 != 0:  # Skip slices that are not multiples of 20
            continue

        frame_path = os.path.join(current_dir, f"{z_coord}.jpg")
        frame = cv.imread(frame_path)
        # DEBUG
        print(f"Reading from {frame_path}, points: {points}")
        
        if frame is None:
            print(f"Frame {z_coord} not found.")
            continue

        # Check if interactive confirmation is enabled
        if args.interactive_confirmation:
            confirmed_points = annotate_points_and_confirm(frame, z_coord, points)
        else:
            confirmed_points = points  # Skip confirmation if disabled

        for point in confirmed_points:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_coord,
                obj_id=args.SB_ID,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32)
            )


    print("All points confirmed. Proceeding with propagation...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    output_dir = os.path.join(args.data_dir, "SB_tracks", str(args.SB_ID), str(args.timestamp))
    os.makedirs(output_dir, exist_ok=True)

    for out_frame_idx, segment_data in video_segments.items():
        for _, out_mask in segment_data.items():
            save_mask(out_mask, os.path.join(output_dir, f"{out_frame_idx}.png"))

    print(f"Segmentation saved to {output_dir}")

if __name__ == "__main__":
    main()
