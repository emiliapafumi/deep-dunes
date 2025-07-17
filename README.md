# dune-cnn

This repository contains data and code for semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure

- `scripts/sampling.py` — python script for sampling patches from raster datasets.
- `scripts/training.py` — python script for training a CNN for semantic segmentation.
- `scripts/inference.py` — python script for performing inference with a trained CNN model.

- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data.

## Use
1) patch extraction
   python scripts/sampling.py --data_folder path/to/data --patch_size 100 --epsg_code 32632
    
3) model training
   python scripts/training.py --data_folder path/to/data --model_name mymodel --img_type rgb --class_nb 5 --epochs 50
   
5) inference

## Example 
As an example, the code is applied here to RGB imagery from UAV dataset.
