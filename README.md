# deep-dunes

This repository contains code to perform semantic segmentation of remote sensing imagery for habitat mapping on coastal dunes.
<br><br>

## Structure of the project

- `scripts/` — Directory for python scripts to use for sampling remote sensing data, training a CNN model, performing inference and validation.
- `models/` — Directory for saved models, logs and checkpoints.
- `deep-dunes-data/` — Directory for input and output data.
- `deep-dunes-graphs.md` — Code for graphs in R.
  
## Usage
### Pre-requisites
Orfeo ToolBox Tensor Flow (OTBTF) is available on Docker. 
Steps for using OTBTF from Docker:
1) install OTBTF from docker by pulling the latest CPU docker image:
```bash
docker pull mdl4eo/otbtf:latest
```

2) create a persistent volume for python libraries:
```bash
docker volume create python_packages
```

3) start a new container from the OTBTF image
```bash
docker run -it --name otbtf_container --platform=linux/amd64 -v "$(pwd):/data" -v python_packages:/usr/local/lib/python3.8/site-packages mdl4eo/otbtf:latest /bin/bash
```
In detail: 
* `docker run`: starts a new container from the specified image (`mdl4eo/otbtf:latest`)
* `-it`: runs an interactive shell session with a usable terminal
* `--name otbtf_container`: assigns a custom name to the container
* `--platform=linux/amd64`: forces Docker to emulate an image for the specified CPU platform
* `-v "$(pwd):/data"`: creates a bind mount, mapping your current working directory on the host machine to the `/data` folder inside the container
* `-v python_packages:/usr/local/lib/python3.8/site-packages`: mounts the named volume `python_packages` to persist installed Python libraries across container restarts
* `/bin/bash`: executes the interactive Bash shell upon startup


4) install the required libraries
```bash
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

5) to start the container once it is created
```bash
docker ps -a
docker start otbtf_container
docker exec -it otbtf_container /bin/bash
cd /data/
```
   
6) other useful functions for docker containers   
`docker restart otbtf_container` to restart the container   
`docker stop otbtf_container` to stop the container without erasing it   
   
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
  
### Step 1:  
```bash
python scripts/1-sampling.py --data_folder dune-air --patch_size 10
```
Run the first script to extract patches from the input image using the ground truth points, divided in training, validation and testing datasets. Patch size should be adjusted according to the spatial resolution of each remote sensing dataset, to represent the 2 m x 2 m ground truth plots.
<br><br>
Close-up view on the input RGB image:  
<br>
<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/c734c658-06a6-43da-9d6a-06f0055a78b0" />  
<br>
Output: a set of GeoTIFF files containing 2m x 2m patches for training, validation and testing.  
```text
dune-air/
├── vec_train.geojson
├── vec_valid.geojson
├── vec_test.geojson
├── train_rgb_patches.tif
├── valid_rgb_patches.tif
├── test_rgb_patches.tif
├── train_labels.tif
├── valid_labels.tif
└── test_labels.tif
```
Example of patches extracted from RGB image (files "_rgb_patches.tif"):
<img width="1663" height="28" alt="Image" src="https://github.com/user-attachments/assets/8500e6ac-e925-427b-b17c-a6b088a9dc64" />
Example of patches with labels (files "_labels.tif"):
<img width="1663" height="28" alt="Image" src="https://github.com/user-attachments/assets/19e6c3b5-71c4-4857-9066-9143cba05485" />
<br><br>

### Step 2:  
```bash
script models/terminal_logs/log_cnn_03.txt
python scripts/2-training.py --data_folder dune-air --model_name cnn-03 --img_type rgb --class_nb 5
```
Run the second script to train a CNN model. Learning rates, number of epochs and batch size are set by default but can be adjusted if needed. Training logs are saved in `models/terminal_logs/` folder.  
  
Output: the trained CNN model is saved in `models/output/savedmodel_cnn-03/`.  
<br>
### Step 3:  
```bash
python scripts/3-inference.py --data_folder dune-air/ --model_name cnn-03 --img_type rgb
```
Run the third script to apply the trained CNN model and produce the habitat map.  
<br>
Output: a GeoTIFF file (`dune_air/map_rgb.tif`) representing the habitat map, with colors corresponding to habitats.  
Close-up view on the map:  
<br>
<img width="300" height="300" alt="Image" src="https://github.com/user-attachments/assets/468ef8e6-6598-4379-ad28-7e5b204f68fc" />
<br>
### Step 4:  
```bash
python scripts/4-assessment.py --data_folder dune-air/ --model_name cnn-03 --img_type rgb
```
Run the fourth script to perform accuracy assessment (calculate overall accuracy, kappa, precision, recall, F-Score, Intersection over Union).  
    
Output: table containing accuracy values, both for the overall classification and for the single classes (`models/accuracy_metrics.csv`).   
  
| Metric              | Class   | Value | CNN    | Image Type |
|---------------------|---------|-------|--------|------------|
| Overall Accuracy    | overall | 0.86  | cnn-03 | rgb        |
| Cohen's Kappa       | overall | 0.83  | cnn-03 | rgb        |
| Average Precision   | overall | 0.89  | cnn-03 | rgb        |
| Average Recall      | overall | 0.86  | cnn-03 | rgb        |
| Average F-Score     | overall | 0.84  | cnn-03 | rgb        |
| Mean IoU            | overall	| 0.71	| cnn-03 | rgb        |
| Precision           | 0       | 0.71  | cnn-03 | rgb        |
| Recall              | 0       | 1.00  | cnn-03 | rgb        |
| F-Score             | 0       | 0.83  | cnn-03 | rgb        |
| Class IoU           | 0       | 0.68  | cnn-03 | rgb        |
| Precision           | 1       | 1.00  | cnn-03 | rgb        |
| Recall              | 1       | 0.40  | cnn-03 | rgb        |
| F-Score             | 1       | 0.57  | cnn-03 | rgb        |
| Class IoU           | 1       | 0.31  | cnn-03 | rgb        |
| Precision           | 2       | 0.76  | cnn-03 | rgb        |
| Recall              | 2       | 0.95  | cnn-03 | rgb        |
| F-Score             | 2       | 0.84  | cnn-03 | rgb        |
| Class IoU           | 2       | 0.70  | cnn-03 | rgb        |
| Precision           | 3       | 1.00  | cnn-03 | rgb        |
| Recall              | 3       | 0.95  | cnn-03 | rgb        |
| F-Score             | 3       | 0.97  | cnn-03 | rgb        |
| Precision           | 4       | 1.00  | cnn-03 | rgb        |
| Recall              | 4       | 1.00  | cnn-03 | rgb        |
| F-Score             | 4       | 1.00  | cnn-03 | rgb        |
| Class IoU           | 4       | 0.99  | cnn-03 | rgb        |
<br>
  
## Notes
The processing is based on the OTBTF/keras tutorial: https://otb-keras-tutorial.readthedocs.io/en/latest/ 






