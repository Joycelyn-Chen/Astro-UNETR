import os
import numpy as np
import torch
import cv2 as cv
import argparse
import hydra
from sam2.build_sam import build_sam2_video_predictor
from utils import save_mask

# python instance_tracking.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS --timestamp 390

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
    bboxes = []
    for mask_file in sorted(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        z = int(os.path.splitext(mask_file)[0])  # Extract z coordinate from filename
        if contours:
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            bboxes.append([x_min, y_min, x_max, y_max, z])
        else:
            bboxes.append([])  # Add an empty list to preserve z structure
    
    return np.array(bboxes, dtype=object)


def extract_random_slices(bboxes, num_slices=3):
    non_zero_slices = []

    # Collect all non-zero bounding box slices
    for z, bbox in enumerate(bboxes):
        if len(bbox) > 0:
            non_zero_slices.append((bbox, bbox[4]))

    if len(non_zero_slices) < num_slices:
        raise ValueError("Not enough non-empty bounding box slices available.")

    # Calculate specific slices
    first_slice_idx = len(non_zero_slices) // 3
    middle_slice_idx = len(non_zero_slices) // 2
    last_slice_idx = middle_slice_idx + len(non_zero_slices) // 6

    selected_slices = [
        non_zero_slices[first_slice_idx],
        non_zero_slices[middle_slice_idx],
        non_zero_slices[last_slice_idx]
    ]

    return selected_slices


def annotate_bbox_and_points(frame, bbox, center_point):
    show_box(frame, bbox)
    show_point(frame, center_point)
    cv.imshow("Annotated Frame", frame)
    cv.waitKey(32)


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
    parser.add_argument("--bbox_expand_rate", default=0.1, type=float, help="Bbox expand rate from previous timestamp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    predictor = initialize_predictor(args, device)

    current_dir = os.path.join(args.data_dir, f"masks-jpg/{args.timestamp}")
    previous_dir = os.path.join(args.data_dir, "SB_tracks", str(args.SB_ID), f"{args.timestamp - args.time_interval}")

    bboxes = get_bounding_boxes(previous_dir)


    frame_names = sorted([
        p for p in os.listdir(current_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".png"]
    ], key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=current_dir)
    # predictor.reset_state(inference_state)

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

    # Add center point and bounding boxes
    center_point = np.array([[args.center_y, args.center_x]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    slices = extract_random_slices(bboxes)

    # for bbox, z_coord in slices:
    # # for bbox in bboxes[np.random.choice(len(bboxes), 3, replace=False)]:

    #     bbox[:2] = [int(x - x * args.bbox_expand_rate) for x in bbox[:2]] #bbox[:2] * args.bbox_expand_rate
    #     bbox[2:4] = [int(x + x * args.bbox_expand_rate) for x in bbox[2:4]] #bbox[2:4] * args.bbox_expand_rate
    #     # z_coord = int(bbox[4])  # Extract z coordinate from the bounding box
    #     bbox_tensor = np.array(bbox[:4], dtype=np.float32)  # Convert bbox to tensor with dtype=float32
        
        
    #     predictor.add_new_points_or_box(
    #         inference_state=inference_state,
    #         frame_idx=int(z_coord),  # Use the z coordinate from the bounding box
    #         obj_id=args.SB_ID,
    #         points=center_point,
    #         labels=labels,
    #         box=bbox_tensor  # Exclude the z coordinate from the bounding box input
    #     )
    confirmed_bboxes = []

    for bbox, z_coord in slices:
        prev_masks = os.path.join('/'.join(current_dir.split('/')[:-1]), str(args.timestamp - args.time_interval))
        frame_path = os.path.join(prev_masks, f"{z_coord}.jpg")
        
        # DEBUG
        print(f"Reading frame from: {frame_path}")
        frame = cv.imread(frame_path) #, cv.IMREAD_GRAYSCALE)

        
        if frame is None:
            print(f"Frame {z_coord} not found.")
            continue

        bbox[:2] = [int(x - x * args.bbox_expand_rate) for x in bbox[:2]]
        bbox[2:4] = [int(x + x * args.bbox_expand_rate) for x in bbox[2:4]]

        center_point = [args.center_y, args.center_x]

        while True:
            annotate_bbox_and_points(frame, bbox, center_point)
            key = input("Confirm annotation? (y/n): ")
            cv.destroyAllWindows()

            if key.lower() == 'y':
                confirmed_bboxes.append((bbox, center_point, z_coord))
                # cv.destroyAllWindows()
                break
            elif key.lower() == 'n':
                print(f"Current bounding box: {bbox}")
                print(f"Current center point: {center_point}")
                
                new_x = int(input("Enter new center x: "))
                new_y = int(input("Enter new center y: "))
                bbox = [
                    int(input("Enter new bbox x_min: ")),
                    int(input("Enter new bbox y_min: ")),
                    int(input("Enter new bbox x_max: ")),
                    int(input("Enter new bbox y_max: "))
                ]
                center_point = [new_y, new_x]


    for bbox, center_point, z_coord in confirmed_bboxes:
        bbox_tensor = np.array(bbox[:4], dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,  # Replace with actual inference state
            frame_idx=z_coord,
            obj_id=args.SB_ID,
            points=np.array([center_point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            box=bbox_tensor
        )
        # DEBUG
        print(f"Added box for frame {z_coord}")

    print("All bounding boxes confirmed. Proceeding with propagation...")

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
