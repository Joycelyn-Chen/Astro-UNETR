#!/usr/bin/env python

import glob
import logging
import os
import sys
from pathlib import Path
import argparse

import torch
# from monai.config import print_config
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.handlers import MeanDice, MLFlowHandler, StatsHandler, TensorBoardImageHandler, TensorBoardStatsHandler
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator


# python train.py --data_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/ --output_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs --exp_name unet-test --max_epochs 10 

def main():
    parser = argparse.ArgumentParser(description="UNet 3D Segmentation Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training images and segmentations (*.nii.gz and *.seg.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs (checkpoints, logs, etc.)")
    parser.add_argument("--exp_name", type=str, default="unet-test", help="Experiment name for saving outputs")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs to train")
    args = parser.parse_args()

    output_directory = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_directory, exist_ok=True)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Starting training...")

    # Use provided data_dir for images and segmentations
    data_dir = args.data_dir
    images = sorted(glob.glob(os.path.join(data_dir, 'train', 'imgs', "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(data_dir, 'train', 'masks', "*.seg.nii.gz")))
    if len(images) == 0 or len(segs) == 0:
        logging.error("No images or segmentations found in the data directory.")
        sys.exit(1)

    # Define transforms for training and validation
    train_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        RandSpatialCrop((96, 96, 96), random_size=False),
    ])
    train_segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        RandSpatialCrop((96, 96, 96), random_size=False),
    ])
    val_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
    ])
    val_segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
    ])

    # Create datasets and dataloaders.
    # Here we assume a simple split: first 20 samples for training, rest for validation.
    train_ds = ArrayDataset(images[:20], train_imtrans, segs[:20], train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    val_ds = ArrayDataset(images[20:], val_imtrans, segs[20:], val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=5, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create the UNet model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Create the trainer using Ignite.
    trainer = create_supervised_trainer(net, optimizer, loss_function, device=device)

    # Set up checkpoint saving.
    log_dir = os.path.join(output_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_handler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # Optional scheduler
    from ignite.handlers import ModelCheckpoint
    checkpoint_handler = ModelCheckpoint(log_dir, filename_prefix="net", n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"net": net, "optimizer": optimizer})

    # Attach stats handlers.
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)
    mlflow_dir = os.path.join(log_dir, "mlruns")
    train_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: x)
    train_mlflow_handler.attach(trainer)

    # Create evaluator for validation.
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    val_metrics = {"Mean_Dice": MeanDice()}
    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device=device,
        non_blocking=False,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)]
        ),
    )
    val_stats_handler = StatsHandler(name="evaluator", output_transform=lambda x: None, global_epoch_transform=lambda engine: trainer.state.epoch)
    val_stats_handler.attach(evaluator)
    val_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None, global_epoch_transform=lambda engine: trainer.state.epoch)
    val_tensorboard_stats_handler.attach(evaluator)
    val_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: None, global_epoch_transform=lambda engine: trainer.state.epoch)
    val_mlflow_handler.attach(evaluator)
    val_tensorboard_image_handler = TensorBoardImageHandler(
        log_dir=log_dir,
        batch_transform=lambda batch: (batch[0], batch[1]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda engine: trainer.state.epoch,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, val_tensorboard_image_handler)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def run_validation(engine):
        evaluator.run(val_loader)

    trainer.run(train_loader, max_epochs=args.max_epochs)
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
