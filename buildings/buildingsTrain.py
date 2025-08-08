import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob

from utils import LULCDataset, visualize_sample  # Keeping utils as is
from unet import UNet  # Using the same U-Net model

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Update dataset paths to your built-up areas dataset
building_images = sorted(glob("building_dataset/train/images/*.tif"))
building_masks = sorted(glob("building_dataset/train/masks/*.tif"))

# Split the dataset into training and testing sets
train_imgs, test_imgs, train_masks, test_masks = train_test_split(building_images, building_masks, test_size=0.2, random_state=42)

# Use the LULCDataset for loading images and masks
train_dataset = LULCDataset(train_imgs, train_masks)
test_dataset = LULCDataset(test_imgs, test_masks)

# Dataloaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Function to calculate accuracy and IoU
def compute_metrics(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()  # Convert output to binary (0 or 1)
    
    intersection = (preds * targets).sum(dim=(1,2))
    union = ((preds + targets) > 0).float().sum(dim=(1,2))
    iou = (intersection / (union + 1e-6)).mean().item()
    
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    acc = (correct / total).item()
    
    return acc, iou

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, masks.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy and IoU
        acc, iou = compute_metrics(outputs, masks)
        total_acc += acc
        total_iou += iou

    # Average loss, accuracy, and IoU for the epoch
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | IoU: {avg_iou:.4f}")

# Save the trained model
torch.save(model.state_dict(), "unet_buildings.pth")
print("Model saved as unet_buildings.pth")
