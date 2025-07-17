"""Sampling EUNIS ground truth data for CNN segmentation"""

import pyotb
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Sampling EUNIS ground truth data")
parser.add_argument("--data_folder", required=True, help="Folder containing data")
parser.add_argument("--patch_size", type=int, default=100, help="Size of patches to extract")
args = parser.parse_args()

# load input raster and shapefile data
img_rgb = args.data_folder + "rgb.tif"  
img_multi = args.data_folder + "multi.tif"  
shapefile_path = args.data_folder + "ground_truth.shp"

# check if files exist
exist_rgb = Path(img_rgb).exists()
exist_multi = Path(img_multi).exists()

# define output folders
out_pth = args.data_folder
output_folder = Path(out_pth)

prefix_train = out_pth + "train_eunis_"
prefix_valid = out_pth + "valid_eunis_"
prefix_test = out_pth + "test_eunis_"


# split into training, validation, and test sets -----------------
# load terrain truth data points

points_gdf = gpd.read_file(shapefile_path)

# Lists to store dataframes
train_list = []
valid_list = []
test_list = []

for class_value, group in points_gdf.groupby('class'):
    # split into 80% training and 20% testing
    train, temp = train_test_split(group, test_size=0.4, random_state=42, stratify=None)
    
    # split the training set into 50% testing and 50% validation
    valid, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=None)
    
    train_list.append(train)
    valid_list.append(valid)
    test_list.append(test)

# Merge groups
shp_train = pd.concat(train_list).reset_index(drop=True)
shp_valid = pd.concat(valid_list).reset_index(drop=True)
shp_test = pd.concat(test_list).reset_index(drop=True)

# Check
print("Train:\n", shp_train.pivot_table(index='class', aggfunc='size'))
print("Valid:\n", shp_valid.pivot_table(index='class', aggfunc='size'))
print("Test:\n", shp_test.pivot_table(index='class', aggfunc='size'))

# export vector files
shp_train.to_file(output_folder / "vec_train.geojson", driver="GeoJSON")
shp_valid.to_file(output_folder / "vec_valid.geojson", driver="GeoJSON")
shp_test.to_file(output_folder / "vec_test.geojson", driver="GeoJSON")

print("Training, test and validation shapefiles saved.")


# Patches extraction application of OTB -----------------
n_sources = str(2) if exist_rgb and exist_multi else str(1)
os.environ["OTB_TF_NSOURCES"] = n_sources

# Load input vectors
vec_train = output_folder / "vec_train.geojson"
vec_valid = output_folder / "vec_valid.geojson"
vec_test = output_folder / "vec_test.geojson"

for vec, prefix in zip([vec_train, vec_valid, vec_test], [prefix_train, prefix_valid, prefix_test]):

    if n_sources == "1":
        app_extract = pyotb.PatchesExtraction({
            "source1.il": img_rgb,
            "source1.patchsizex": args.patch_size,
            "source1.patchsizey": args.patch_size,
            "source1.nodata": 0,
            "source1.out": prefix + "rgb_patches.tif",
            "vec": vec,
            "field": "class",
            "outlabels": prefix + "labels.tif",
        })

    else:
        app_extract = pyotb.PatchesExtraction({
            "source1.il": img_rgb,
            "source1.patchsizex": args.patch_size,
            "source1.patchsizey": args.patch_size,
            "source1.nodata": 0,
            "source1.out": prefix + "rgb_patches.tif",
            "source2.il": img_multi,
            "source2.patchsizex": args.patch_size,
            "source2.patchsizey": args.patch_size,
            "source2.nodata": 0,
            "source2.out": prefix + "multi_patches.tif",
            "vec": vec,
            "field": "class",
            "outlabels": prefix + "labels.tif",
    })

