"""Semantic segmentation of remote sensing images for EUNIS classification"""

import pyotb
import argparse

from setup import setup

parser = argparse.ArgumentParser(description="Apply the CNN model")
parser.add_argument("--data_folder", required=True, help="Folder containing data")
parser.add_argument("--model_name", required=True, help="model name")
parser.add_argument("--img_type", choices=["rgb", "multi"], default="rgb", help="Type of input image")
parser.add_argument("--ext_fname", required=False, help="subset of the output image")
params = parser.parse_args()

# define directories
setup(directory_name=params.data_folder)
data_folder = f"deep-dunes-data/{params.data_folder}"

model_dir = "deep-dunes/models/output/savedmodel_" + params.model_name
input_file = params.data_folder + params.img_type + ".tif"
output_file = params.data_folder + "map_" + params.img_type + ".tif"

infer = pyotb.TensorflowModelServe(
  n_sources=1,
  source1_il=input_file,
  source1_rfieldx=100, 
  source1_rfieldy=100,
  source1_placeholder="input_img",
  model_dir=model_dir,
  model_fullyconv=True,
  output_efieldx=68, 
  output_efieldy=68,
  output_names="argmax_layer_crop16"
)
infer.write(
    output_file,  # output image filename
    pixel_type="uint8",  # output image encoding
    ext_fname=params.ext_fname,  # subset of the output image
)
