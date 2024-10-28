"""Semantic segmentation of RGB images"""

from mymetrics import FScore
import otbtf
import tensorflow as tf
import argparse
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import HeNormal
import efficientnet.keras as efn 


class_nb = 3             # number of classes
inp_key_ge = "input_ge"  # model input ge
tgt_key = "estimated"    # model target


def create_otbtf_dataset(ge, labels):
    return otbtf.DatasetFromPatchesImages(
        filenames_dict={
            "ge3": ge,
            "labels3": labels
        }
    )

def dataset_preprocessing_fn(sample):
    return {
        inp_key_ge: sample["ge3"],
        tgt_key: otbtf.ops.one_hot(labels=sample["labels3"], nb_classes=class_nb)
    }

def create_dataset(ge, labels, batch_size=8):
    otbtf_dataset = create_otbtf_dataset(ge, labels)
    return otbtf_dataset.get_tf_dataset(
        batch_size=batch_size,
        preprocessing_fn=dataset_preprocessing_fn,
        targets_keys=[tgt_key]
    )

def normalize_inputs(self, inputs):
        return {
            inp_key_ge: tf.cast(inputs[inp_key_ge], tf.float32) * 0.01
        }

# EfficientNet B0: input shape (224, 224, 3)
model = efn.EfficientNetB0(
    include_top=True, # include final dense layer
    input_tensor=inputs,
    weights='imagenet', # or weights='noisy-student'
    pooling=None,
    classes=3,
    classifier_activation="softmax",
    name="efficientnetb0"
)

# train model
model.fit(
    ds_train,
    epochs=params.epochs,
    validation_data=ds_valid,
    callbacks=callbacks
)

parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("--model_dir", required=True, help="model directory")
parser.add_argument("--log_dir", help="log directory")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_dir", help="Directory for checkpoints")
params = parser.parse_args()
tf.get_logger().setLevel('ERROR')