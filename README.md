# deep-dunes

This repository contains code to perform semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.
<br><br>

## Structure of the project

- `scripts/` — Directory for python scripts to use for sampling remote sensing data, training a CNN model, performing inference and validation.
- `models/` — Directory for saved models, logs and checkpoints.
- `deep-dunes-data/` — Directory for input and output data. This folder is available at this link: https://drive.google.com/drive/folders/1krYQ6T-wg3J54ZcwJ8tcnwlp4-mPcgFa?usp=share_link 
<br><br>
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

3) start the container you have created using its name (es. name_container):
```bash
docker ps -a
docker start name_container
docker exec -it name_container /bin/bash
cd /data/
```
   
### Steps to produce habitat maps:
1) sampling: to extract patches corresponding to the ground truth squares (2 m x 2 m) from each remote sensing dataset;
2) model training: to train a CNN model for image segmentation;
3) inference: to produce the final habitat map using a trained CNN model;
4) validation: to compute accuracy metrics.
<br><br>
  
## Example of application
In this example, the RGB imagery from airborne dataset is used (model CNN-03).  
Note: for the other CNNs, different datasets are used:  
- CNN-01: input = dune-uav/rgb.tif  
- CNN-02: input = dune-uav/multi.tif  
- CNN-03: input = dune-air/rgb.tif  
- CNN-04: input = dune-air/multi.tif  
- CNN-05: input = dune-ge/rgb.tif  
- CNN-06: input = dune-wv/rgb.tif  
- CNN-07: input = dune-wv/multi.tif
<br>
Before proceeding, download the deep-dunes-data/ folder and save it in deep-dunes/deep-dunes-data.
  
### Step 1:  
```bash
cd deep-dunes/
python scripts/1-sampling.py --data_folder dune-air/ --patch_size 10
```
Run the first script to extract patches from the input image using the ground truth points, divided in training, validation and testing datasets.  
Close-up view on the input RGB image:  
<br>
<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/c734c658-06a6-43da-9d6a-06f0055a78b0" />  
<br>
Output: a set of GeoTIFF files containing 2m x 2m patches for training, validation and testing.  
<br>
Example of patches extracted from RGB image (files _rgb_patches.tif):
<img width="1663" height="28" alt="Image" src="https://github.com/user-attachments/assets/8500e6ac-e925-427b-b17c-a6b088a9dc64" />
Example of patches with labels (files _labels.tif):
<img width="1663" height="28" alt="Image" src="https://github.com/user-attachments/assets/19e6c3b5-71c4-4857-9066-9143cba05485" />
<br>
### Step 2:  
```bash
script models/terminal_logs/log_cnn_03.txt
python scripts/2-training.py --data_folder dune-air/ --model_name cnn-03 --img_type rgb --class_nb 5
```
Run the second script to train a CNN model. Learning rates, number of epochs and batch size are set by default but can be adjusted if needed.  
Note: script is used to save a log (inside models/terminal_logs/ folder).  
  
Output: the trained CNN model is saved in models/output/savedmodel_cnn-03/.  
<br>
### Step 3:  
```bash
python scripts/3-inference.py --data_folder dune-air/ --model_name cnn-03 --img_type rgb
```
Run the third script to apply the trained CNN model and produce the habitat map.  
<br>
Output: a GeoTIFF file (map_rgb.tif) representing the habitat map, with colors corresponding to habitats.  
Close-up view on the map:  
<br>
<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/468ef8e6-6598-4379-ad28-7e5b204f68fc" />
<br>
### Step 4:  
```bash
python scripts/4-validation.py --data_folder dune-air/ --model_name cnn-03 --img_type rgb
```
Run the fourth script to perform accuracy assessment (calculate overall accuracy, kappa, precision, recall, F-Score).  
    
Output: table containing accuracy values, both for the overall classification and for the single classes (models/accuracy_metrics.csv).   
  
| Metric              | Class   | Value | CNN    | Image Type |
|---------------------|---------|-------|--------|------------|
| Overall Accuracy    | overall | 0.86  | cnn-03 | rgb        |
| Cohen's Kappa       | overall | 0.83  | cnn-03 | rgb        |
| Average Precision   | overall | 0.89  | cnn-03 | rgb        |
| Average Recall      | overall | 0.86  | cnn-03 | rgb        |
| Average F-Score     | overall | 0.84  | cnn-03 | rgb        |
| Precision           | 0       | 0.71  | cnn-03 | rgb        |
| Recall              | 0       | 1.00  | cnn-03 | rgb        |
| F-Score             | 0       | 0.83  | cnn-03 | rgb        |
| Precision           | 1       | 1.00  | cnn-03 | rgb        |
| Recall              | 1       | 0.40  | cnn-03 | rgb        |
| F-Score             | 1       | 0.57  | cnn-03 | rgb        |
| Precision           | 2       | 0.76  | cnn-03 | rgb        |
| Recall              | 2       | 0.95  | cnn-03 | rgb        |
| F-Score             | 2       | 0.84  | cnn-03 | rgb        |
| Precision           | 3       | 1.00  | cnn-03 | rgb        |
| Recall              | 3       | 0.95  | cnn-03 | rgb        |
| F-Score             | 3       | 0.97  | cnn-03 | rgb        |
| Precision           | 4       | 1.00  | cnn-03 | rgb        |
| Recall              | 4       | 1.00  | cnn-03 | rgb        |
| F-Score             | 4       | 1.00  | cnn-03 | rgb        |
<br>
  
## Notes
The processing is based on the OTBTF/keras tutorial: https://otb-keras-tutorial.readthedocs.io/en/latest/ 



