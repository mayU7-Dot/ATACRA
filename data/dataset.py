"""Dataset classes for Brixia chest X-ray severity scoring."""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BrixiaDataset(Dataset):
    """Brixia CXR dataset with 6-region severity labels (0-3 per region).

    Parameters
    ----------
    filename_list : list[str]
        List of image filenames.
    label_list : list[list[int]]
        Corresponding severity labels per region, shape (N, 6).
    label_df : pd.DataFrame
        Dataframe mapping filename to metadata.
    prefix : str
        Path prefix where images are stored.
    transform : nn.Module or None
        Base transform (resize, normalise, etc.).
    augmentation : callable or None
        Albumentations-style augmentation for training.
    train : bool
        If True, strong data augmentation is applied.
    """

    def __init__(self, filename_list, label_list, label_df, prefix,
                 transform=None, augmentation=None, train=True):
        self.filename_list = filename_list
        self.label = label_list
        self.label_df = label_df
        self.prefix = prefix
        self.train = train
        self.transform = transform
        self.augmentation = augmentation

        self.strong_transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomRotation(10)], p=0.8
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.5, contrast=0.5,
                    saturation=0.5, hue=0.5
                )], p=0.8
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=0.0, p=0.8),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))],
                p=0.8
            ),
        ])

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        full_filename = self.label_df[
            self.label_df["Filename"] == self.filename_list[idx]
        ]["Filename"].values[0]
        full_filename = full_filename[:-4] + ".png"

        file_path = self.prefix + full_filename
        image = Image.open(file_path)

        now_label = torch.tensor(self.label[idx], dtype=torch.int64)
        now_label_float = torch.tensor(self.label[idx], dtype=torch.float32)

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        image = transforms.ToTensor()(image)

        if self.transform:
            image = self.transform(image).type(torch.float32)
            if self.train:
                image = self.strong_transform(image)

        if image.shape[0] != 3:
            image = image.expand(3, -1, -1)

        return image, now_label, now_label_float, self.filename_list[idx]


class PneumoniaBinaryDataset(Dataset):
    """Binary pneumonia classification dataset (Normal vs Pneumonia).

    Compatible with the public Kaggle pediatric pneumonia dataset:
    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    """

    def __init__(self, image_paths, labels, transform=None, train=True,
                 pil_aug=None, tensor_aug=None):
        self.image_paths = image_paths
        self.labels = labels             # 0 = Normal, 1 = Pneumonia
        self.transform = transform
        self.train = train

        # PIL-based augmentations — applied BEFORE ToTensor so ColorJitter
        # operates on real pixel values.
        if pil_aug is not None:
            self.pil_aug = pil_aug
        elif train:
            self.pil_aug = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.7),
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))],
                    p=0.5
                ),
            ])
        else:
            self.pil_aug = None

        # Tensor-level augmentations (applied AFTER Normalize for valid
        # photometric augmentations that expect zero-centred inputs).
        if tensor_aug is not None:
            self.tensor_aug = tensor_aug
        elif train:
            self.tensor_aug = transforms.Compose([
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
                ),
            ])
        else:
            self.tensor_aug = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # PIL augmentations on raw pixels
        if self.train and self.pil_aug is not None:
            image = self.pil_aug(image)

        image = transforms.ToTensor()(image)

        if self.transform:
            image = self.transform(image)

        # Tensor augmentations after Normalize
        if self.train and self.tensor_aug is not None:
            image = self.tensor_aug(image)

        if image.shape[0] != 3:
            image = image.expand(3, -1, -1)

        return image, label
