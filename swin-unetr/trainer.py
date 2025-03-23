# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

import torch.nn.functional as F

def morphological_difference(mask):
    """
    Extracts the exterior ring from a binary segmentation mask.
    
    The exterior ring is computed as the difference between the mask and its eroded version.
    Erosion is performed using a 3x3 kernel. A pixel is considered eroded (i.e., kept)
    only if all of its 3x3 neighborhood pixels are 1.
    
    Args:
        mask (torch.Tensor): Binary mask tensor of shape [N, 1, H, W].
        
    Returns:
        torch.Tensor: Binary mask of the exterior ring of shape [N, 1, H, W].
    """
    # Define a 3x3 kernel filled with ones.
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)
    # Perform convolution to simulate erosion (padding=1 to maintain size).
    eroded = F.conv2d(mask, kernel, padding=1)
    # Keep pixels where the full 3x3 neighborhood is ones.
    eroded_binary = (eroded == 9).float()
    # The ring is the difference between the original mask and its eroded version.
    ring = mask - eroded_binary
    # Ensure the ring mask is binary.
    ring = (ring > 0).float()
    return ring

def r_loss(pred_mask, temp_cube):
    """
    Computes the ratio loss (r_loss) that forces the segmented bubble to be as discrete as possible.
    
    The steps are as follows:
      1. Use the predicted binary segmentation mask.
      2. Extract the exterior ring mask from the segmentation mask.
      3. Apply (cast) the ring mask onto the temperature cube.
      4. Count the number of ring pixels with temperature values over 125.
      5. Calculate the r_loss as the ratio of the count in step 4 to the total number of ring pixels.
    
    Args:
        pred_mask (torch.Tensor): Binary segmentation mask of shape [N, 1, H, W].
        temp_cube (torch.Tensor): Temperature cube tensor of shape [N, 1, H, W].
        
    Returns:
        torch.Tensor: A scalar tensor representing the ratio loss.
    """
    # 2. Get the exterior ring.
    ring_mask = morphological_difference(pred_mask)
    
    # 3. Apply the ring mask to the temperature cube.
    ring_temp = temp_cube * ring_mask
    
    # 4. Count the number of ring pixels with temperature > 125.
    count_over = (ring_temp > 125).float().sum()
    
    # 5. Count the total number of ring pixels.
    total_ring = ring_mask.sum()
    
    # Avoid division by zero.
    epsilon = 1e-6
    ratio_loss = count_over / (total_ring + epsilon)
    
    # Return as a tensor (the result of torch operations is already a tensor).
    return ratio_loss


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        # Retrieve data and target from batch
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        temp_cube = data.cuda(2) # rank 2 is the temperature cube
        
        # Zero gradients
        for param in model.parameters():
            param.grad = None
        
        with autocast(enabled=args.amp):
            # Forward pass
            logits = model(data)
            
            # Compute the original dice loss
            dice_loss_val = loss_func(logits, target)
            
            # Convert logits to a binary segmentation mask
            pred_mask = (torch.sigmoid(logits) > 0.5).float()
            
            # Compute the r_loss (ratio loss) using the predicted mask and the temperature cube
            # Here we assume that the input data contains the temperature cube information.
            r_loss_val = r_loss(pred_mask, temp_cube)
            r_loss_val = r_loss_val.to(dtype=dice_loss_val.dtype, device=dice_loss_val.device)
            
            # Total loss is the sum of dice loss and ratio loss
            total_loss = dice_loss_val + r_loss_val
        
        # Backward pass and optimizer step
        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        # Distributed loss aggregation (if applicable)
        if args.distributed:
            loss_list = distributed_all_gather(
                [total_loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(total_loss.item(), n=args.batch_size)
        
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    
    # Clear gradients after epoch
    for param in model.parameters():
        param.grad = None
        
    return run_loss.avg


# Original code
# def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
#     model.train()
#     start_time = time.time()
#     run_loss = AverageMeter()
#     for idx, batch_data in enumerate(loader):
#         if isinstance(batch_data, list):
#             data, target = batch_data
#         else:
#             data, target = batch_data["image"], batch_data["label"]
#         data, target = data.cuda(args.rank), target.cuda(args.rank)
#         for param in model.parameters():
#             param.grad = None
#         with autocast(enabled=args.amp):
#             logits = model(data)

#             # joy's doing
#             # print(f"logits type: {type(logits)}, shape: {logits.shape}")
#             # print(f"{logits[150]}")
            

#             loss = loss_func(logits, target)
#         if args.amp:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
#         if args.distributed:
#             loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
#             run_loss.update(
#                 np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
#             )
#         else:
#             run_loss.update(loss.item(), n=args.batch_size)
#         if args.rank == 0:
#             print(
#                 "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
#                 "loss: {:.4f}".format(run_loss.avg),
#                 "time {:.2f}s".format(time.time() - start_time),
#             )
#         start_time = time.time()
#     for param in model.parameters():
#         param.grad = None
#     return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                Dice_TC = val_acc[0]
                Dice_WT = val_acc[1]
                Dice_ET = val_acc[2]
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
