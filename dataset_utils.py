# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:46:56 2023

@author: Sid Ahmed Hamdad
"""
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def dataset_check(dataset_path):
    print(len(os.listdir(os.path.join(dataset_path, "train\\images"))))
    print(len(os.listdir(os.path.join(dataset_path, "train\\masks"))))
    print(len(os.listdir(os.path.join(dataset_path, "val\\images"))))
    print(len(os.listdir(os.path.join(dataset_path, "val\\masks"))))
    print(len(os.listdir(os.path.join(dataset_path, "test\\images"))))
    print(len(os.listdir(os.path.join(dataset_path, "test\\masks"))))

    # Set directory paths
    image_dir = os.path.join(dataset_path, "train\\images")
    mask_dir = os.path.join(dataset_path, "train\\masks")

    # Get list of image and mask filenames
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    # Check that the number of images and masks match
    assert len(image_filenames) == len(mask_filenames)

    # Print 5 random image and mask pairs
    for i in range(5):
        index = random.randint(0, len(image_filenames) - 1)
        image_filename = image_filenames[index]
        mask_filename = mask_filenames[index]
        print(mask_filename)
        print(image_filename)
        image = Image.open(os.path.join(image_dir, image_filename))
        mask = Image.open(os.path.join(mask_dir, mask_filename))
        print("Maximum pixel value of image: ", np.max(np.array(image)))
        print("Maximum pixel value of mask: ", np.max(np.array(mask)))

        print(f"Image {i+1}: {image_filename}")
        print(f"Mask {i+1}: {mask_filename}")

        mask = np.array(mask)
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)

        # Show images using Matplotlib
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Mask")
        plt.show()
