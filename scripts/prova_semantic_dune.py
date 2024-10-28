# PROVA SEMANTIC SEGMENTATION su dune 


# verify installation of packages

import pyotb
import pystac_client
import planetary_computer

import argparse
import otbtf
import tensorflow as tf


# list files in the data folder
import os
os.getcwd()
os.listdir("/data/")

# patches selection in QGIS

# patches extraction

import pyotb

vec_train = "/data/vec_train.geojson"
vec_valid = "/data/vec_valid.geojson"
vec_test = "/data/vec_test.geojson"

for vec in [vec_train, vec_valid, vec_test]:
    name = vec.split('/')[-1].replace("vec_", "").replace(".geojson", "")
    
    app_extract = pyotb.PatchesExtraction(
        n_sources=2,  
        source1_il="/data/GE_aoi1.tif",
        source1_patchsizex=64,
        source1_patchsizey=64,
        source1_nodata=0,
        source2_il="/data/tt.tif",
        source2_patchsizex=64,
        source2_patchsizey=64,
        source2_nodata=0,
        vec=vec,
        field="id"
    )
    
    out_dict = {
        "source1.out": name + "_ge_patches.tif",
        "source2.out": name + "_labels_patches.tif",
    }
    
    pixel_type = {
        "source1.out": "int16",
        "source2.out": "uint8",
    }
    
    ext_fname = "gdal:co:COMPRESS=DEFLATE"
    app_extract.write(out_dict, pixel_type=pixel_type, ext_fname=ext_fname)


# define some constants
class_nb = 3             # number of classes
inp_key_ge = "input_ge"    # model input ge
tgt_key = "estimated"    # model target

# helper to create otbtf dataset from lists of patches
def create_otbtf_dataset(ge, labels):
    return otbtf.DatasetFromPatchesImages(
        filenames_dict={
            "ge": ge,
            "labels": labels
        }
    )

# dataset preprocessing function
def dataset_preprocessing_fn(sample):
    return {
        inp_key_ge: sample["ge"],
        tgt_key: otbtf.ops.one_hot(labels=sample["labels"], nb_classes=class_nb)
    }

# TensorFlow dataset creation from lists of patches
def create_dataset(ge, labels, batch_size=8):
    otbtf_dataset = create_otbtf_dataset(ge, labels)
    return otbtf_dataset.get_tf_dataset(
        batch_size=batch_size,
        preprocessing_fn=dataset_preprocessing_fn,
        targets_keys=[tgt_key]
    )

# define convolution operator
def conv(inp, depth, name, strides=2):
    conv_op = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=3,
        strides=strides,
        activation="relu",
        padding="same",
        name=name
    )
    return conv_op(inp)

# define transposed convolution operator
def tconv(inp, depth, name, activation="relu"):
    tconv_op = tf.keras.layers.Conv2DTranspose(
        filters=depth,
        kernel_size=3,
        strides=2,
        activation=activation,
        padding="same",
        name=name
    )
    return tconv_op(inp)

# build the model
import otbtf
class FCNNModel(otbtf.ModelBase):
    
    def normalize_inputs(self, inputs):
        return {
            inp_key_ge: tf.cast(inputs[inp_key_ge], tf.float32) * 0.01,
        }
    
    def get_outputs(self, normalized_inputs):
        norm_inp_ge = normalized_inputs[inp_key_ge]
                
        cv1 = conv(norm_inp_ge, 16, "conv1")
        cv2 = conv(cv1, 32, "conv2")
        cv3 = conv(cv2, 64, "conv3")
        cv4 = conv(cv3, 64, "conv4")
        cv1t = tconv(cv4, 64, "conv1t") + cv3
        cv2t = tconv(cv1t, 32, "conv2t") + cv2
        cv3t = tconv(cv2t, 16, "conv3t") + cv1
        cv4t = tconv(cv3t, class_nb, "softmax_layer", "softmax")
        
        argmax_op = otbtf.layers.Argmax(name="argmax_layer")
        
        return {tgt_key: cv4t, "estimated_labels": argmax_op(cv4t)}


# custom metric for F1-Score (code from: https://stackoverflow.com/questions/64474463/custom-f1-score-metric-in-tensorflow)

class FScore(tf.keras.metrics.Metric):
        
    def __init__(self, class_id, name=None, **kwargs):
        if not name:
            name = f'f_score_{class_id}'
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(class_id=class_id)
        self.recall_fn = tf.keras.metrics.Recall(class_id=class_id)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))
    
    def result(self):
        return self.f1
    
    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)


# training setup
def train(params, ds_train, ds_valid, ds_test):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = FCNNModel(dataset_element_spec=ds_train.element_spec)
        
        # Precision and recall for each class
        metrics = [
            cls(class_id=class_id)
            for class_id in range(class_nb)
            for cls in [tf.keras.metrics.Precision, tf.keras.metrics.Recall]
        ]
        
        # F1-Score for each class
        metrics += [
            FScore(class_id=class_id, name=f"fscore_cls{class_id}")
            for class_id in range(class_nb)
        ]
        
        model.compile(
            loss={tgt_key: tf.keras.losses.CategoricalCrossentropy()},
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            metrics={tgt_key: metrics}
        )
        model.summary()
        save_best_cb = tf.keras.callbacks.ModelCheckpoint(
            params.model_dir,
            mode="min",
            save_best_only=True,
            monitor="val_loss"
        )
        callbacks = [save_best_cb]
        if params.log_dir:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=params.log_dir))
        if params.ckpt_dir:
            ckpt_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=params.ckpt_dir)
            callbacks.append(ckpt_cb)
        
        # Train the model
        model.fit(
            ds_train,
            epochs=params.epochs,
            validation_data=ds_valid,
            callbacks=callbacks
        )
        
        # Final evaluation on the test dataset
        model.load_weights(params.model_dir)
        values = model.evaluate(ds_test, batch_size=params.batch_size)
        for metric_name, value in zip(model.metrics_names, values):
            print(f"{metric_name}: {100*value:.2f}")


# build a simple parser to provide arguments

import sys
import argparse
import tensorflow as tf

# Simulate the command-line arguments 
sys.argv = [ 
    'part_3_train.py',  # Script name 
    '--model_dir', '/data/models/model3', 
    '--log_dir', '/data/logs/model3',  
    '--epochs', '50',  
    '--ckpt_dir', '/data/ckpts/model3' 
]

# Now parse the arguments using argparse
parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("--model_dir", required=True, help="model directory")
parser.add_argument("--log_dir", help="log directory")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_dir", help="Directory for checkpoints")
params = parser.parse_args()

# Print to verify 
print("Model directory:", params.model_dir)

tf.get_logger().setLevel('ERROR')


# dataset intantiation
ds_train = create_dataset(
    ["/data/train_ge_patches.tif"],
    ["/data/train_labels_patches.tif"],
)
ds_train = ds_train.shuffle(buffer_size=100)

ds_valid = create_dataset(
    ["/data/valid_ge_patches.tif"],
    ["/data/valid_labels_patches.tif"],
)

ds_test = create_dataset(
    ["/data/test_ge_patches.tif"],
    ["/data/test_labels_patches.tif"],
)



# train the model
train(params, ds_train, ds_valid, ds_test)


# inference to observe blocking artifacts

import sys
import argparse
import tensorflow as tf

# Simulate the command-line arguments 
sys.argv = [ 
    'part_3_train.py',  # Script name 
    '--model_dir', '/data/models/model3'
]

import pyotb

# Generate the classification map
infer = pyotb.TensorflowModelServe(
  n_sources=1,
  source1_il="/data/GE_aoi1.tif",
  source1_rfieldx=128,
  source1_rfieldy=128,
  source1_placeholder="input_ge",
  model_dir=params.model_dir,
  model_fullyconv=True,
  output_efieldx=128,
  output_efieldy=128,
  output_names="softmax_layer"
)

infer.write(
  "/data/map_artifacts.tif",
  ext_fname="box=2000:2000:1000:1000"
)


# inference without blocking artifacts

import pyotb
import argparse

parser = argparse.ArgumentParser(description="Apply the model")
parser.add_argument("--model_dir", required=True, help="model directory")
params = parser.parse_args()

# Generate the classification map
infer = pyotb.TensorflowModelServe(
  n_sources=1,
  source1_il="/data/GE_aoi1.tif",
  source1_rfieldx=128,
  source1_rfieldy=128,
  source1_placeholder="input_ge",
  model_dir=params.model_dir,
  model_fullyconv=True,
  output_efieldx=64,
  output_efieldy=64,
  output_names="softmax_layer_crop32"
)

infer.write(
  "/data/map_valid.tif",
  ext_fname="box=2000:2000:1000:1000"
)

infer.write(
  "/data/map_valid_all.tif"
)

# create summary (to visualize in tensorboard)

log_dir = "/data/logs/model3"
tf.summary.create_file_writer(log_dir)

