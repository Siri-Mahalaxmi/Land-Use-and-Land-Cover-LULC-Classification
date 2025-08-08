import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(image, ground_truth, prediction=None, title="Sample"):
    if prediction is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(ground_truth.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    if prediction is not None:
        axes[2].imshow(prediction.squeeze(), cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Dataset class to load images and masks
class LULCDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert to PyTorch tensor
        image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).long()  # Convert mask to binary (1 for buildings, 0 for others)

        return image, mask
