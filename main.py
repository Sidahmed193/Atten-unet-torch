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
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# verify that there is cuda device available
lookforcuda()

# path to dataset directory with train test and val folders inside
dataset_path = "F:\\PFE_dataset\\2D_AXE_Z_50_50_split_90_5_5\\working"

# check that dataset is good
dataset_check(dataset_path)


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=(128, 128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize
        image = TF.resize(image, self.target_size)
        mask = TF.resize(mask, self.target_size)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        angle = random.uniform(-0.2, 0.2)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Random affine transformation
        affine_params = transforms.RandomAffine.get_params(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            scale_ranges=(0.95, 1.05),
            shears=(-0.05, 0.05, -0.05, 0.05),
            img_size=self.target_size,
        )
        image = TF.affine(image, *affine_params)
        mask = TF.affine(mask, *affine_params)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask


def trainGenerator(
    batch_size, train_path, image_folder, mask_folder, target_size=(128, 128)
):
    # Create image and mask paths
    image_paths = [
        os.path.join(train_path, image_folder, filename)
        for filename in sorted(tqdm(os.listdir(os.path.join(train_path, image_folder))))
    ]
    mask_paths = [
        os.path.join(train_path, mask_folder, filename)
        for filename in sorted(tqdm(os.listdir(os.path.join(train_path, mask_folder))))
    ]
    # Create dataset
    dataset = CustomDataset(image_paths, mask_paths)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


# Function to show an image and mask side by side
def show_image_mask(image, mask):
    # Convert tensors to numpy arrays
    image_np = image.squeeze(0).numpy()  # Squeeze to remove batch dimension
    mask_np = mask.squeeze(0).numpy()  # Squeeze to remove batch dimension

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Display the image
    ax1.imshow(image_np, cmap="gray")
    ax1.set_title("Image")
    ax1.axis("off")

    # Display the mask as an overlay on the image
    ax2.imshow(image_np, cmap="gray")
    ax2.imshow(mask_np, cmap="jet", alpha=0.5)  # Overlay mask with transparency
    ax2.set_title("Overlay")
    ax2.axis("off")

    # Display only the mask
    ax3.imshow(mask_np, cmap="gray")
    ax3.set_title("Mask")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


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
        train_loss = 0.0
        val_loss = 0.0
        val_dice = 0.0

        # Training loop
        # Initialize tqdm progress bar
        pbar_train = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
        for i, data in pbar_train:
            image, mask = data
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Update the postifx values on the tqdm progress bar
            pbar_train.set_postfix({"loss": f"{train_loss / (i + 1):.4f}"})

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_data in val_loader:
                val_image, val_mask = val_data
                val_image, val_mask = val_image.to(device), val_mask.to(device)
                val_outputs = model(val_image)
                val_loss += criterion(val_outputs, val_mask).item()
                val_dice += dice_coeff(val_outputs.cpu(), val_mask.cpu())

        # Calculate average losses and dice coefficient
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice Coeff: {val_dice:.4f}"
        )


if __name__ == "__main__":
    train_loader_test = trainGenerator(
        batch_size=10,  # Set batch_size to 1 to fetch one image-mask pair at a time
        train_path=os.path.join(dataset_path, "train"),
        image_folder="images",
        mask_folder="masks",
        target_size=(128, 128),  # Assuming this is your desired target size
    )
    counter = 10
    # Iterate through the DataLoader to get a batch
    for image, mask in train_loader_test:
        show_image_mask(image[0], mask[0])  # Indexing at 0 since batch_size=1
        counter -= 1
        if counter <= 0:
            break  # Only show 10 pairs

    # Define train_loader and val_loader
    train_loader = trainGenerator(
        batch_size=32,
        train_path=os.path.join(dataset_path, "train"),
        image_folder="images",
        mask_folder="masks",
        target_size=(128, 128),
    )
    val_loader = trainGenerator(
        batch_size=32,
        train_path=os.path.join(dataset_path, "val"),
        image_folder="images",
        mask_folder="masks",
        target_size=(128, 128),
    )

    # Define UNet model and optimizer
    model = AttentionUNet()
    # print summery of model
    summary(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("start of training")
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, num_epochs=5)
