import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import hydra

from utils import *
import argparse

from sam2.build_sam import build_sam2_video_predictor

parser = argparse.ArgumentParser(description="Tracking semantic segmentation output from Swin-UNETR with SAM2 to become instance segmentation")
parser.add_argument("--data_dir", default="./Dataset", type=str, help="input data directory")
parser.add_argument("--SB_ID", default=230, type=int, help="SB ID for the target bubble")
parser.add_argument("--center_x", default=150, type=int, help="center x for the target bubble")
parser.add_argument("--center_y", default=178, type=int, help="center y for the target bubble")
parser.add_argument("--center_z", default=130, type=int, help="center z for the target bubble")
parser.add_argument("--bbox_x1", default=100, type=int, help="bounding box x1 coord for the target bubble")
parser.add_argument("--bbox_x2", default=200, type=int, help="bounding box x2 coord for the target bubble")
parser.add_argument("--bbox_y1", default=128, type=int, help="bounding box y1 coord for the target bubble")
parser.add_argument("--bbox_y2", default=228, type=int, help="bounding box y2 coord for the target bubble")
parser.add_argument("--timestamp", default=380, type=int, help="Timestamp of interest")
parser.add_argument("--sam2_root", default="/home/joycelyn/Desktop/sam2", type=str, help="path to sam2 root directory")


args = parser.parse_args()

# python video-inference.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/masks 

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# Select an video input
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`

video_dir = os.path.join(args.data_dir, str(args.timestamp)) 

# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
# hydra.initialize_config_module("configs", version_base='1.2')
hydra.initialize_config_dir(config_dir="/home/joycelyn/Desktop/sam2/sam2/configs/sam2.1", version_base='1.2')


sam2_checkpoint = os.path.join(args.sam2_root, "checkpoints/sam2.1_hiera_large.pt") 
# model_cfg = os.path.join(args.sam2_root, "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")  
model_cfg = "sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", "png"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

print("Initializing inference state...")
# Initialize the inference state
inference_state = predictor.init_state(video_path=video_dir)

# clean the inference state if necessary
# predictor.reset_state(inference_state)


# Add 1st click
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers)



# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[5, 5]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

ann_frame_idx = args.center_z  # the frame index we interact with
ann_obj_id = args.SB_ID   # give a unique id to each object we interact with (it can be any integers)


print("Adding point and bbox prompt to predictor...")
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[args.center_y, args.center_x]], dtype=np.float32)
box = np.array([args.bbox_x1, args.bbox_y1, args.bbox_x2, args.bbox_y2], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    box=box,
)

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_box(box, plt.gca())
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


print("Propagating the video...")
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


instance_out_dir = os.path.join(args.data_dir, '..', 'SB_tracks', str(args.SB_ID), str(args.timestamp))
os.makedirs(instance_out_dir, exist_ok=True)

print("Saving the inference output...")
for out_frame_idx in range(len(frame_names)):
    # for out_obj_id, out_mask in video_segments[out_frame_idx].items():  # should be able to remove this loop
    out_mask = list(video_segments[out_frame_idx].values())[0]  # Directly access the single segment
    save_mask(out_mask, os.path.join(instance_out_dir, f"{out_frame_idx}.png"))

print(f"Done! Outputs saved at {instance_out_dir}")
