import matplotlib.pyplot as plt

import numpy as np

import torchvision
import torch


def plot_grid(dataloader):
    """A utility function to plot grid of images from dataloader
    Args:
        dataloader: A dataloader object
    """
    images, labels = next(iter(dataloader))
    grid = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16, 16))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
