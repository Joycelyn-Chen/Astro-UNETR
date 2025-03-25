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

import json
import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.transforms import Transform

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            tr.append(d)
        else:
            val.append(d)

    return tr, val


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


# Joy's doing
# def load_slices(folder, ext="jpg"):
#     """Helper function to load slices and stack them into a 3D array."""
#     slice_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(f".{ext}")])
#     data = np.stack([transforms.LoadImage(image_only=True)(f) for f in slice_files], axis=0)

#     # DEBUG
#     print(f"reading images from {folder}, shape: ", data.shape)

#     return data

class LoadSlices(Transform):
    def __init__(self, image_only=True, dtype=np.float32, ext='jpg', reader='PILReader'):
        self.image_only = image_only
        self.dtype = dtype
        self.ext = ext
        self.reader = reader

    def __call__(self, data):
        # `data` should be the path to the folder containing the image slices
        folder_path = data

        # DEBUG
        print(f"Processing folder: {folder_path}")

        loader = transforms.LoadImage(image_only=self.image_only, reader=self.reader)
        slice_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(f".{self.ext}")])
        slices = [loader(f) for f in slice_files]
        stacked_slices = np.stack(slices, axis=0).astype(self.dtype)
        
        print(f"Stacked slices, shape: {stacked_slices.shape}")

        return stacked_slices

def datafold_read_astro(datalist, basedir, fold=0, key="training"):
    """Read JSON file and return data for the specified fold."""
    with open(datalist, 'r') as f:
        json_data = json.load(f)
    
    data = json_data[key]
    train_files = []
    val_files = []
    test_files = []

    for entry in data:
        # # DEBUG
        # print(f"Entry in data: {entry}")

        entry["image"] = os.path.join(basedir, entry["image"])
        entry["label"] = os.path.join(basedir, entry["label"])
        if entry.get("fold") == fold:
            val_files.append(entry)
        elif entry.get("type") == "test":
            test_files.append(entry)
        else:
            train_files.append(entry)
    
    
    return train_files, val_files, test_files

from torch.utils.data import DataLoader

def get_astro_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    fold = args.fold

    if args.testing:
        key = "testing"

    # Read dataset for the given fold
    train_files, val_files, test_files = datafold_read_astro(datalist=datalist_json, basedir=data_dir, fold=fold, key=key)

    # Define transformations for training, validation, and testing
    def load_transform(data):
        # DEBUG
        print(f"load_transform being called: reading {data}")
        # assert False, "load_transform has been called!"

        image_data = load_slices(data["image"], ext="jpg")
        label_data = load_slices(data["label"], ext="png")
        return {"image": image_data, "label": label_data}

    train_transform = transforms.Compose(
        [
            # load_transform,
            LoadSlices(),
            transforms.NormalizeIntensityd(keys="image"),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.ScaleIntensityd(keys="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            # load_transform,
            LoadSlices(),
            transforms.NormalizeIntensityd(keys="image"),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            # load_transform(test_files[0]),      # Joy: magic [0] within, not good
            LoadSlices(),
            transforms.NormalizeIntensityd(keys="image"),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    img_transform = transforms.Compose([
        LoadSlices(ext='jpg'),
        transforms.NormalizeIntensityd(keys="image"),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    seg_transform = transforms.Compose([
        LoadSlices(ext='png'),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ToTensord(keys=["image", "label"])
    ])


    # print(f"In data loader, test_files: {test_files}")
    # result = load_transform(test_files[0])
    # print("Test load_transform output:", result)

    # Original:
    # # Create data loaders
    # train_ds = data.Dataset(data=train_files, transform=train_transform)
    # val_ds = data.Dataset(data=val_files, transform=val_transform)
    # test_ds = data.Dataset(data=test_files, transform=test_transform)


    # DEBUG
    # print(f"Passing test_files into data.ImageDataset: {test_files}")

    train_ds = data.ImageDataset(
        image_files=[f['image'] for f in train_files],
        seg_files=[f['label'] for f in train_files],
        transform=img_transform,
        seg_transform=seg_transform
    )
    val_ds = data.ImageDataset(
        image_files=[f['image'] for f in val_files],
        seg_files=[f['label'] for f in val_files],
        transform=img_transform,
        seg_transform=seg_transform
    )
    test_ds = data.ImageDataset(
        image_files=[f['image'] for f in test_files],
        seg_files=[f['label'] for f in test_files],
        transform=img_transform,
        seg_transform=seg_transform
    )

    # train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Original:
    # test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    return test_loader  #train_loader, val_loader, 

