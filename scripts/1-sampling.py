"""Sampling EUNIS ground truth data for CNN segmentation"""

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

from setup import setup, DATASET_IDS

parser = argparse.ArgumentParser(description="Sampling EUNIS ground truth data")
parser.add_argument("--data_folder", required=True, choices=list(DATASET_IDS.keys()), 
                    help=f"Folder containing data. Must be one of: {list(DATASET_IDS.keys())}")
parser.add_argument("--patch_size", type=int, default=100, help="Size of patches to extract")
parser.add_argument("--epsg_code", type=int, default=32632, help="EPSG code for output raster")
args = parser.parse_args()

# load input raster and shapefile data
setup(directory_name=args.data_folder)
data_folder = Path(f"deep-dunes-data/{args.data_folder}")
img_rgb_path = data_folder / "rgb.tif"
img_multi_path = data_folder / "multi.tif"
shapefile_path = data_folder / "ground_truth.shp"

# Split ground truth points into train, validation, and test sets
points_gdf = gpd.read_file(shapefile_path)

# Lists to store dataframes
train_list = []
valid_list = []
test_list = []

for class_value, group in points_gdf.groupby('class'):
    # split into 60% training and 40% testing
    train, temp = train_test_split(group, test_size=0.4, random_state=42, stratify=None)

    # split the testing set into 50% testing and 50% validation
    valid, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=None)

    train_list.append(train)
    valid_list.append(valid)
    test_list.append(test)

# Merge groups
shp_train = pd.concat(train_list).reset_index(drop=True)
shp_valid = pd.concat(valid_list).reset_index(drop=True)
shp_test = pd.concat(test_list).reset_index(drop=True)

splits = {
    "train": shp_train,
    "valid": shp_valid,
    "test": shp_test
}

# Check
print("Train:\n", shp_train.pivot_table(index='class', aggfunc='size'))
print("Valid:\n", shp_valid.pivot_table(index='class', aggfunc='size'))
print("Test:\n", shp_test.pivot_table(index='class', aggfunc='size'))

# export vector files
shp_train.to_file(data_folder / "vec_train.geojson", driver="GeoJSON")
shp_valid.to_file(data_folder / "vec_valid.geojson", driver="GeoJSON")
shp_test.to_file(data_folder / "vec_test.geojson", driver="GeoJSON")

print("Training, test and validation shapefiles saved.")


# Function to extract patches from ground truth points
def extract_patches_from_centroids(gdf_points, img_path, patch_size=args.patch_size):
    patch_imgs = []
    patch_labels = []
    with rasterio.open(img_path) as src:
        transform = src.transform
        inv_transform = ~transform
        for idx, row in gdf_points.iterrows():
            label = row['class']
            point = row.geometry

            col, row_px = inv_transform * (point.x, point.y)
            col, row_px = int(round(col)), int(round(row_px))

            half = patch_size // 2
            row_start = row_px - half
            col_start = col - half

            if (row_start < 0 or col_start < 0 or
                row_start + patch_size > src.height or
                col_start + patch_size > src.width):
                continue  # skip patches that go out of bounds

            window = rasterio.windows.Window(col_start, row_start, patch_size, patch_size)
            patch = src.read(window=window)

            label_mask = np.full((1, patch_size, patch_size), label, dtype=np.uint8)

            patch_imgs.append(patch)
            patch_labels.append(label_mask)

    return patch_imgs, patch_labels

# Function to stack patches into a single image
def stack_patches(patch_list, label_list, out_img_path, out_lbl_path):
    # Extract patch dimension
    patch_h, patch_w = patch_list[0].shape[1:]
    bands = patch_list[0].shape[0]
    total_patches = len(patch_list)

    # Create grid dimensions
    grid_cols = 1
    grid_rows = total_patches

    # Empty arrays for stacked patches
    stacked_img = np.zeros((bands, grid_rows * patch_h, grid_cols * patch_w), dtype=patch_list[0].dtype)
    stacked_lbl = np.zeros((1, grid_rows * patch_h, grid_cols * patch_w), dtype=label_list[0].dtype)

    for idx, (patch, label) in enumerate(zip(patch_list, label_list)):
        row = idx // grid_cols
        col = idx % grid_cols
        y0, y1 = row * patch_h, (row + 1) * patch_h
        x0, x1 = col * patch_w, (col + 1) * patch_w

        stacked_img[:, y0:y1, x0:x1] = patch
        stacked_lbl[:, y0:y1, x0:x1] = label

    # Write output files
    meta = {
        "driver": "GTiff",
        "height": stacked_img.shape[1],
        "width": stacked_img.shape[2],
        "count": bands,
        "dtype": stacked_img.dtype,
        "crs": f"EPSG:{args.epsg_code}",
        "transform": rasterio.transform.from_origin(0, 0, 1, 1)
    }

    with rasterio.open(out_img_path, "w", **meta) as dst:
        dst.write(stacked_img)

    if out_lbl_path is not None:
        meta["count"] = 1
        meta["dtype"] = stacked_lbl.dtype
        with rasterio.open(out_lbl_path, "w", **meta) as dst:
            dst.write(stacked_lbl)


# Apply to each split
for split_name, split_gdf in splits.items():
    print(f"Processing {split_name}...")

    if img_rgb_path.exists():
        patches, labels = extract_patches_from_centroids(split_gdf, img_path=img_rgb_path)
        if patches:
            stack_patches(
                patches,
                labels,
                out_img_path=data_folder / f"{split_name}_rgb_patches.tif",
                out_lbl_path=data_folder / f"{split_name}_labels.tif"
            )
    if img_multi_path.exists():
        patches, labels = extract_patches_from_centroids(split_gdf, img_path=img_multi_path)
        if patches:
            stack_patches(
                patches,
                labels,
                out_img_path=data_folder / f"{split_name}_multi_patches.tif",
                out_lbl_path=None # Labels are already saved
            )

print("Done.")
