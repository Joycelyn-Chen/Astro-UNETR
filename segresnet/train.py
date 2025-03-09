import glob
import os
import sys
import time
import logging
import argparse
from pathlib import Path
import numpy as np

import torch
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    RandSpatialCrop,
    Resize,
    Activations,
    AsDiscrete,
)
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric
from monai.handlers import StatsHandler, TensorBoardStatsHandler, MLFlowHandler, MeanDice
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint






# python train.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs --exp_name segresnet-test --max_epochs 10 



# ------------------------------------------------------------------------------
# Helper transform: duplicate single channel to 4 channels.
def duplicate_channels(x):
    """
    If x has one channel, duplicate it along the channel axis to create a 4-channel input.
    Supports both numpy arrays and torch tensors.
    """
    if x.shape[0] == 1:
        if isinstance(x, np.ndarray):
            return np.repeat(x, 4, axis=0)
        elif isinstance(x, torch.Tensor):
            return x.repeat(4, 1, 1, 1)
    return x

# ------------------------------------------------------------------------------
# Custom conversion function for segmentation labels to multi-channel format.
def convert_to_multichannel(label):
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    # Squeeze out the extra channel if present (i.e. from shape (1, H, W, D) to (H, W, D))
    if label.ndim == 4 and label.shape[0] == 1:
        label = label.squeeze(0)
    # Tumor Core (TC): merge label 2 and 3
    tc = torch.logical_or(label == 2, label == 3)
    # Whole Tumor (WT): merge labels 1, 2 and 3
    wt = torch.logical_or(torch.logical_or(label == 1, label == 2), label == 3)
    # Enhancing Tumor (ET): label 2 only
    et = (label == 2)
    multi = torch.stack([tc, wt, et], dim=0).float()  # resulting shape: (3, H, W, D)
    return multi


# A simple lambda wrapper for label conversion.
def convert_seg(x):
    return convert_to_multichannel(x)

# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SegResNet 3D Segmentation Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training images and segmentations (under train/imgs and train/masks)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs (checkpoints, logs, etc.)")
    parser.add_argument("--exp_name", type=str, default="segresnet-exp", help="Experiment name for saving outputs")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train")
    args = parser.parse_args()

    output_directory = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_directory, exist_ok=True)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Starting SegResNet training...")

    # Read training images and segmentation masks.
    train_img_paths = sorted(glob.glob(os.path.join(args.data_dir, "train", "imgs", "*.nii.gz")))
    train_seg_paths = sorted(glob.glob(os.path.join(args.data_dir, "train", "masks", "*.seg.nii.gz")))
    if len(train_img_paths) == 0 or len(train_seg_paths) == 0:
        logging.error("No training images or segmentations found in the specified directory.")
        sys.exit(1)
    
    # For validation, use an 80/20 split.
    num_train = len(train_img_paths)
    split_idx = int(num_train * 0.8)
    train_imgs = train_img_paths[:split_idx]
    train_segs = train_seg_paths[:split_idx]
    val_imgs = train_img_paths[split_idx:]
    val_segs = train_seg_paths[split_idx:]

    # Define transforms for images and segmentation masks.
    # Note: Crop/resize is applied first, then we duplicate the channel.
    train_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),             # now shape: (1, H, W, D)
        RandSpatialCrop((96, 96, 96), random_size=False),  # crop spatial dims (H, W, D)
        duplicate_channels,               # duplicate channel -> (4, 96, 96, 96)
    ])
    train_segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),             # segmentation remains (1, H, W, D)
        # lambda x: convert_seg(x),         # convert to multi-channel label: (3, H, W, D)
        RandSpatialCrop((96, 96, 96), random_size=False),
        lambda x: convert_seg(x),         # convert to multi-channel label: (3, H, W, D)
    ])
    val_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
        duplicate_channels,
    ])
    val_segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        lambda x: convert_seg(x),
        Resize((96, 96, 96)),
    ])

    # Create datasets and data loaders.
    train_ds = ArrayDataset(train_imgs, train_imtrans, train_segs, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_ds = ArrayDataset(val_imgs, val_imtrans, val_segs, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create the SegResNet model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,   # 4-channel input (after duplication)
        out_channels=3,  # 3 segmentation channels (TC, WT, ET)
        dropout_prob=0.2,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=False, softmax=True)  # to_onehot_y=True
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    # Create the trainer using Ignite.
    trainer = create_supervised_trainer(model, optimizer, loss_function, device=device)

    # Set up the evaluator using MeanDice from monai.handlers.
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    evaluator = create_supervised_evaluator(
        model,
        metrics={"Mean_Dice": MeanDice()},
        device=device,
        non_blocking=False,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)]
        ),
    )

    # Set up checkpoint saving.
    log_dir = os.path.join(output_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_handler = ModelCheckpoint(log_dir, filename_prefix="segresnet", n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model, "optimizer": optimizer})

    # Attach stats handlers to the trainer.
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)
    mlflow_dir = os.path.join(log_dir, "mlruns")
    train_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: x)
    train_mlflow_handler.attach(trainer)

    # Run evaluation at the end of each epoch.
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def run_validation(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        dice = metrics["Mean_Dice"]
        logging.info(f"Validation Mean Dice: {dice:.4f}")

    trainer.run(train_loader, max_epochs=args.max_epochs)
    logging.info("Training completed.")

if __name__ == "__main__":
    main()