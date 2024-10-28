import pyotb
import argparse

parser = argparse.ArgumentParser(description="Apply the model")
parser.add_argument("--model_dir", required=True, help="model directory")
parser.add_argument("--output_name", required=True, help="output file name")
params = parser.parse_args()

# Generate the classification map
infer = pyotb.TensorflowModelServe(
  n_sources=1,
  source1_il="/data/GE_aoi2.tif",
  source1_rfieldx=128,
  source1_rfieldy=128,
  source1_placeholder="input_ge",
  model_dir=params.model_dir,
  model_fullyconv=True,
  output_efieldx=64,
  output_efieldy=64,
  output_names="softmax_layer_crop32"
)

infer.write(params.output_name)
