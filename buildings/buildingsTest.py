import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import LULCDataset  # make sure this returns image, mask in _getitem_
from unet import UNet
from glob import glob
from sklearn.model_selection import train_test_split

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset paths
building_images = sorted(glob("building_dataset/train/images/*.tif"))
building_masks = sorted(glob("building_dataset/train/masks/*.tif"))

# Split dataset
_, test_imgs, _, test_masks = train_test_split(building_images, building_masks, test_size=0.2, random_state=42)
test_dataset = LULCDataset(test_imgs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load("unet_buildings.pth"))
model.eval()

# Function to show image, GT and predicted mask
def visualize_prediction(image, ground_truth, prediction, idx):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0).cpu())
    axs[0].set_title("Input Image")
    axs[1].imshow(ground_truth.squeeze().cpu(), cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(prediction.squeeze().cpu(), cmap='gray')
    axs[2].set_title("Predicted Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize predictions for 5–10 images
num_to_visualize = 7  # Change to 5 or 10 as needed
with torch.no_grad():
    for idx, (image, mask) in enumerate(test_loader):
        if idx >= num_to_visualize:
            break
        image = image.to(device)
        output = model(image)
        pred_mask = torch.sigmoid(output).cpu()
        pred_mask = (pred_mask > 0.5).float()

        visualize_prediction(image[0], mask[0], pred_mask[0], idx)
