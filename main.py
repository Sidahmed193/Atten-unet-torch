# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:27:44 2023

@author: Sid Ahmed Hamdad
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset_utils import dataset_check
from torch_cuda_test import lookforcuda
import os
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from unet import AttentionUNet
from PIL import Image

# verify that there is cuda device available
lookforcuda()

# path to dataset directory with train test and val folders inside
dataset_path = "F:\\PFE_dataset\\2D_AXE_Z_50_50_split_90_5_5\\working"

# check that dataset is good
dataset_check(dataset_path)


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = torch.load(self.image_paths[idx])  # Load images directly
        # mask = torch.load(self.mask_paths[idx])  # Load masks directly
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def trainGenerator(
    batch_size, train_path, image_folder, mask_folder, target_size=(128, 128)
):
    # Create image and mask paths
    image_paths = [
        os.path.join(train_path, image_folder, filename)
        for filename in tqdm(os.listdir(os.path.join(train_path, image_folder)))
    ]
    mask_paths = [
        os.path.join(train_path, mask_folder, filename)
        for filename in tqdm(os.listdir(os.path.join(train_path, mask_folder)))
    ]
    # Create transformations for data augmentation
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    # Create dataset
    dataset = CustomDataset(image_paths, mask_paths, transform=transform)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


def dice_coeff(prediction, target):
    smooth = 1e-6  # Epsilon for numerical stability
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


# Define Dice Loss
def dice_loss(prediction, target):
    return 1 - dice_coeff(prediction, target)


def train_model(model, train_loader, val_loader, optimizer, num_epochs=5):
    criterion = dice_loss  # Using Dice Loss as the criterion

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, data in tqdm(enumerate(train_loader, 0)):
            image, mask = data
            image, mask = image.to(device), mask.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                val_dice += dice_coeff(
                    val_outputs.cpu().numpy(), val_labels.cpu().numpy()
                )

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Dice Coeff: {val_dice:.4f}"
        )


if __name__ == "__main__":
    # Define train_loader and val_loader
    train_loader = trainGenerator(
        batch_size=32,
        train_path=os.path.join(dataset_path, "train"),
        image_folder="images",
        mask_folder="masks",
    )
    val_loader = trainGenerator(
        batch_size=32,
        train_path=os.path.join(dataset_path, "val"),
        image_folder="images",
        mask_folder="masks",
    )

    # Define UNet model and optimizer
    model = AttentionUNet()
    # print summery of model
    summary(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("start of train")
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, num_epochs=5)
