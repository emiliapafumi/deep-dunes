# deep-dunes

This repository contains code to perform semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.

## Structure of the project

- `scripts/` — Directory for python scripts to use for sampling remote sensing data, training a CNN model and performing inference.
- `models/` — Directory for saved models, logs and checkpoints.
- `data/` — Directory for input and output data. This folder is available at this link: https://drive.google.com/drive/folders/1krYQ6T-wg3J54ZcwJ8tcnwlp4-mPcgFa?usp=share_link 

## Usage
### Pre-requisites
Orfeo ToolBox Tensor Flow (OTBTF) is available on Docker. 
Steps for using OTBTF from Docker:
1) pull the latest CPU docker image:
```bash
docker pull mdl4eo/otbtf:latest
```

2) start a container from the OTBTF image, create a persistent volume for python libraries and install the required libraries:
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

3) start the container you have created using its name (es. priceless_goodall):
```bash
docker ps -a
docker start priceless_goodall
docker exec -it priceless_goodall /bin/bash
cd /data/
```

### Steps to produce habitat maps:
1) sampling: to extract patches corresponding to the ground truth squares (2 m x 2 m) from each remote sensing dataset;
2) model training: to train a CNN model with U-Net architecture;
3) inference: to produce the final habitat map using a trained CNN model;
4) validation: to compute accuracy metrics.

Example of use on RGB imagery from UAV dataset (model CNN-01):
```bash
python deep-dunes/scripts/1-sampling.py --data_folder data/dune-uav/ --patch_size 100
script models/terminal_logs/log_cnn_01.txt
python deep-dunes/scripts/2-training.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --class_nb 4
python deep-dunes/scripts/3-inference.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb --ext_fname box=3000:3000:5000:5000
python deep-dunes/scripts/4-validation.py --data_folder data/dune-uav/ --model_name cnn-01 --img_type rgb
```
note: script is used to save a log.  
Models created with these scripts:
- CNN-01: input = dune-uav/rgb.tif
- CNN-02: input = dune-uav/multi.tif
- CNN-03: input = dune-air/rgb.tif
- CNN-04: input = dune-air/multi.tif
- CNN-05: input = dune-ge/rgb.tif
- CNN-06: input = dune-wv/rgb.tif
- CNN-07: input = dune-wv/multi.tif
 


## Notes
The processing is based on the OTBTF/keras tutorial: https://otb-keras-tutorial.readthedocs.io/en/latest/ 
