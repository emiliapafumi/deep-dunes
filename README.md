# deep-dunes

This repository contains code to perform semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure of the project

- `scripts/` — Directory for python scripts to use for sampling remote sensing data, training a CNN model and performing inference.
- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data. This folder is available at this link: https://drive.google.com/drive/folders/1krYQ6T-wg3J54ZcwJ8tcnwlp4-mPcgFa?usp=share_link 

## Usage
Steps to produce habitat maps:
1) sampling: to extract patches corresponding to the ground truth squares (2 m x 2 m) from each remote sensing dataset;
2) model training: to train a CNN model with U-Net architecture;
3) inference: to produce the final habitat map using a trained CNN model.

Example of use on RGB imagery from UAV dataset (model CNN-01):
```bash
python scripts/1-sampling.py --data_folder data/dune-uav/ --patch_size 100 --epsg_code 32632
script models/terminal_logs/log_cnn_01.txt
python scripts/2-training.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --class_nb 4 --epochs 50
python scripts/3-inference.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --ext_fname box=3000:3000:5000:5000
```

### Requirements
Orfeo ToolBox Tensor Flow (OTBTF) is available on Docker. 
Steps for using OTBTF from Docker:
1) start a container from the OTBTF image, create a persistent volume for python libraries and install the required libraries:
```bash
docker volume create python_packages
docker run -it --platform=linux/amd64 -v ~/Desktop/deep-dunes:/data -v python_packages:/Users/emilpaf/Library/Python/3.9/lib/python/site-packages mdl4eo/otbtf:latest /bin/bash
cd /data/
pip install -r scripts/requirements.txt
pip list
```
Libraries needed:
- Python 3.8+
- tensorflow
- keras
- otbtf
- rasterio
- geopandas
- scikit-learn
- numpy

2) start the container you have created using its name (es. priceless_goodall):
```bash
docker ps -a
docker start priceless_goodall
docker exec -it priceless_goodall /bin/bash
cd /data/
```

## Notes
The processing is based on the OTBTF/keras tutorial: https://otb-keras-tutorial.readthedocs.io/en/latest/ 
