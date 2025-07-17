# dune-cnn

This repository contains data and code for semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure of the project

- `scripts/` — Directory for python code to use for sampling remote sensing data, training a CNN model and performing inference.
- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data.

## Use
1) patch extraction
   python scripts/sampling.py --data_folder path/to/data --patch_size 100 --epsg_code 32632
    
3) model training
   python scripts/training.py --data_folder path/to/data --model_name mymodel --img_type rgb --class_nb 5 --epochs 50
   
5) inference


## Notes
The processing is based on the OTBTF/keras tutorial available here: https://otb-keras-tutorial.readthedocs.io/en/latest/ 
