# dune-cnn

This repository contains data and code for semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure of the project

- `scripts/` — Directory for python code to use for sampling remote sensing data, training a CNN model and performing inference.
- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data.

## Use
Here is an example of use on RGB imagery from UAV dataset. 
1) patch extraction: patches corresponding to the ground truth squares (2 m x 2 m) are extracted from each remote sensing dataset.  \n
   `python scripts/sampling.py --data_folder data/dune-uav/ --patch_size 100 --epsg_code 32632`
    
3) model training: a CNN model is trained. \n
   `python scripts/training.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --class_nb 4 --epochs 50`
   
5) inference


## Notes
The processing is based on the OTBTF/keras tutorial available here: https://otb-keras-tutorial.readthedocs.io/en/latest/ 
