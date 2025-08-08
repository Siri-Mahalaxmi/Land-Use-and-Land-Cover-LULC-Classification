import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
from PIL import Image
import glob
import random
from sklearn.model_selection import train_test_split

PATCH_SIZE = 256
DATA_DIR = "satellite_zips/sentinel2_data/sentinel2_data"
OUTPUT_DIR = "vegetation_dataset"  # Updated output folder

def read_and_pad(src, window, patch_size=256):
    data = src.read(1, window=window).astype(np.float32)
    h, w = data.shape
    padded = np.zeros((patch_size, patch_size), dtype=np.float32)
    padded[:h, :w] = data
    return padded

def get_band_paths(safe_folder):
    try:
        img_data_path = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA"))
        if not img_data_path:
            raise FileNotFoundError(f"IMG_DATA folder not found in {safe_folder}")
        img_data_path = img_data_path[0]

        band_paths = {
            "B02": glob.glob(os.path.join(img_data_path, "*_B02.jp2")),
            "B03": glob.glob(os.path.join(img_data_path, "*_B03.jp2")),
            "B04": glob.glob(os.path.join(img_data_path, "*_B04.jp2")),
            "B08": glob.glob(os.path.join(img_data_path, "*_B08.jp2")),
        }

        missing_bands = [band for band, paths in band_paths.items() if len(paths) == 0]
        if missing_bands:
            raise ValueError(f"Missing bands: {', '.join(missing_bands)} in {safe_folder}")

        return {band: paths[0] for band, paths in band_paths.items()}
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None

def process_safe_folder(safe_path, image_mask_pairs, prefix):
    band_paths = get_band_paths(safe_path)
    
    if band_paths is None:
        print(f"Skipping {safe_path} due to missing bands.")
        return

    try:
        with rasterio.open(band_paths["B04"]) as red_src, \
             rasterio.open(band_paths["B03"]) as green_src, \
             rasterio.open(band_paths["B02"]) as blue_src, \
             rasterio.open(band_paths["B08"]) as nir_src:

            height, width = red_src.height, red_src.width
            patch_id = 0

            for top in range(0, height, PATCH_SIZE):
                for left in range(0, width, PATCH_SIZE):
                    window = Window(left, top, PATCH_SIZE, PATCH_SIZE)

                    red = read_and_pad(red_src, window)
                    green = read_and_pad(green_src, window)
                    blue = read_and_pad(blue_src, window)
                    nir = read_and_pad(nir_src, window)

                    # NDVI Calculation
                    ndvi = (nir - red) / (nir + red + 1e-6)

                    # Create vegetation mask using NDVI threshold
                    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                    mask[ndvi > 0.2] = 1  # NDVI > 0.3 indicates healthy vegetation

                    print(f"Max NDVI in this patch: {np.max(ndvi)}")  # Debugging

                    if np.any(mask == 1):  # Save only relevant patches
                        img_rgb = np.stack([red, green, blue], axis=-1)
                        img_rgb = np.clip((img_rgb / 3000) * 255, 0, 255).astype(np.uint8)

                        binary_mask = (mask == 1).astype(np.uint8) * 255

                        image_mask_pairs.append((
                            Image.fromarray(img_rgb),
                            Image.fromarray(binary_mask),
                            f"{prefix}_{patch_id}"
                        ))

                        print(f"Vegetation patch found: {prefix}_{patch_id}")

                    patch_id += 1

    except Exception as e:
        print(f"Error processing {safe_path}: {e}")

def run_for_all_safe_folders(base_folder=DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/test/images", exist_ok=True)
    os.makedirs(f"{output_dir}/test/masks", exist_ok=True)

    safe_folders = glob.glob(os.path.join(base_folder, "*.SAFE"))
    print(f"Found {len(safe_folders)} SAFE folders.")
    image_mask_pairs = []

    for idx, safe_folder in enumerate(safe_folders):
        print(f"Processing {safe_folder}...")
        process_safe_folder(safe_folder, image_mask_pairs, prefix=f"tile{idx}")

    print(f"Total image-mask pairs found: {len(image_mask_pairs)}")

    if len(image_mask_pairs) == 0:
        print("No valid image-mask pairs found. Exiting.")
        return

    train_data, test_data = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)

    for split, dataset in [('train', train_data), ('test', test_data)]:
        for img, mask, name in dataset:
            img.save(f"{output_dir}/{split}/images/{name}.tif")
            mask.save(f"{output_dir}/{split}/masks/{name}.tif")

run_for_all_safe_folders()
