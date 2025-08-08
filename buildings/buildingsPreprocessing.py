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
OUTPUT_DIR = "building_dataset"

def read_and_pad(src, window, patch_size=256):
    data = src.read(1, window=window).astype(np.float32)
    h, w = data.shape
    padded = np.zeros((patch_size, patch_size), dtype=np.float32)
    padded[:h, :w] = data
    return padded

def get_band_paths(safe_folder):
    try:
        # Find the image data path
        img_data_path = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA"))
        
        if not img_data_path:
            raise FileNotFoundError(f"IMG_DATA folder not found in {safe_folder}")

        img_data_path = img_data_path[0]
        print(f"Image data path: {img_data_path}")
        
        # Try fetching each band path, raise error if not found
        band_paths = {
            "B02": glob.glob(os.path.join(img_data_path, "*_B02.jp2")),
            "B03": glob.glob(os.path.join(img_data_path, "*_B03.jp2")),
            "B04": glob.glob(os.path.join(img_data_path, "*_B04.jp2")),
            "B08": glob.glob(os.path.join(img_data_path, "*_B08.jp2")),
            "B11": glob.glob(os.path.join(img_data_path, "*_B11.jp2"))
        }

        # Check if any band is missing
        missing_bands = [band for band, paths in band_paths.items() if len(paths) == 0]
        if missing_bands:
            raise ValueError(f"Missing bands: {', '.join(missing_bands)} in {safe_folder}")

        # Return the band paths
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
             rasterio.open(band_paths["B08"]) as nir_src, \
             rasterio.open(band_paths["B11"]) as swir_src:

            height, width = red_src.height, red_src.width
            patch_id = 0

            for top in range(0, height, PATCH_SIZE):
                for left in range(0, width, PATCH_SIZE):
                    window = Window(left, top, PATCH_SIZE, PATCH_SIZE)

                    red = read_and_pad(red_src, window)
                    green = read_and_pad(green_src, window)
                    blue = read_and_pad(blue_src, window)
                    nir = read_and_pad(nir_src, window)

                    # SWIR band resizing (half the resolution)
                    swir_window = Window(left // 2, top // 2, PATCH_SIZE // 2, PATCH_SIZE // 2)
                    swir_raw = swir_src.read(1, window=swir_window).astype(np.float32)
                    h, w = swir_raw.shape
                    swir_padded = np.zeros((PATCH_SIZE // 2, PATCH_SIZE // 2), dtype=np.float32)
                    swir_padded[:h, :w] = swir_raw
                    swir_resized = resize(swir_padded, (PATCH_SIZE, PATCH_SIZE), order=1,
                                          preserve_range=True, anti_aliasing=True)

                    # Built-up Index (NDBI)
                    ndbi = (swir_resized - nir) / (swir_resized + nir + 1e-6)

                    # Create mask: built-up area detection using NDBI threshold
                    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                    mask[ndbi > 0.1] = 1  # Adjusted threshold for built-up area

                    print(f"Max NDBI in this patch: {np.max(ndbi)}")  # Debugging line

                    if np.any(mask == 1):  # Only save patches that contain built-up area
                        img_rgb = np.stack([red, green, blue], axis=-1)
                        img_rgb = np.clip((img_rgb / 3000) * 255, 0, 255).astype(np.uint8)

                        binary_mask = (mask == 1).astype(np.uint8) * 255

                        image_mask_pairs.append((
                            Image.fromarray(img_rgb),
                            Image.fromarray(binary_mask),
                            f"{prefix}_{patch_id}"
                        ))

                        print(f"Valid patch found for {prefix}_{patch_id}")  # Debugging line

                    patch_id += 1

    except Exception as e:
        print(f"Error processing {safe_path}: {e}")

def run_for_all_safe_folders(base_folder=DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/test/images", exist_ok=True)
    os.makedirs(f"{output_dir}/test/masks", exist_ok=True)

    safe_folders = glob.glob(os.path.join(base_folder, "*.SAFE"))
    print(f"Found {len(safe_folders)} SAFE folders.")  # Debugging line
    image_mask_pairs = []

    for idx, safe_folder in enumerate(safe_folders):
        print(f"Processing {safe_folder}...")
        process_safe_folder(safe_folder, image_mask_pairs, prefix=f"tile{idx}")

    # Check if there are any image-mask pairs before splitting
    print(f"Total image-mask pairs found: {len(image_mask_pairs)}")

    if len(image_mask_pairs) == 0:
        print("No valid image-mask pairs found. Exiting.")
        return

    # Split dataset: 80% train, 20% test
    train_data, test_data = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)

    for split, dataset in [('train', train_data), ('test', test_data)]:
        for img, mask, name in dataset:
            img.save(f"{output_dir}/{split}/images/{name}.tif")
            mask.save(f"{output_dir}/{split}/masks/{name}.tif")

run_for_all_safe_folders()
