# dune-cnn

This repository contains data and code for semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure of the project

- `scripts/` — Directory for python scripts to use for sampling remote sensing data, training a CNN model and performing inference.
- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data.

## Usage
Steps to produce habitat maps:
1) sampling: to extract patches corresponding to the ground truth squares (2 m x 2 m) from each remote sensing dataset;
3) model training: to train a CNN model with U-Net architecture;
5) inference: to produce the final habitat map using a trained CNN model.

Example of application to RGB imagery from UAV dataset:
- `python scripts/sampling.py --data_folder data/dune-uav/ --patch_size 100 --epsg_code 32632` 
- `python scripts/training.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --class_nb 4 --epochs 50`

## Requirements
- Python 3.8+
- tensorflow
- keras
- otbtf
- rasterio
- geopandas
- scikit-learn
- numpy

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Notes
The processing is based on the OTBTF/keras tutorial: https://otb-keras-tutorial.readthedocs.io/en/latest/ 
