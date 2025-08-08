import os
import random
import rasterio
import matplotlib.pyplot as plt
from glob import glob

# Path to mask images
train_masks_path = "vegetation_dataset/train/masks/"
mask_files = sorted(glob(os.path.join(train_masks_path, "*.tif")))

# Select 15 random masks
selected_masks = random.sample(mask_files, min(15, len(mask_files)))

# Plot the masks
plt.figure(figsize=(15, 10))  # Adjusted size to fit 15 images
for i, file in enumerate(selected_masks):
    with rasterio.open(file) as src:
        mask = src.read(1)  # Assuming single-channel mask

    plt.subplot(3, 5, i + 1)  # 3 rows and 5 columns to fit 15 images
    plt.imshow(mask, cmap='gray')
    plt.title(os.path.basename(file), fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
