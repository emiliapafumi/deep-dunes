"""Semantic segmentation of remote sensing images for EUNIS classification"""
from mymetrics import FScore
import otbtf
import tensorflow as tf
import argparse
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import HeNormal
from pathlib import Path
from tensorflow import keras


parser = argparse.ArgumentParser(description="Train a CNN model")
parser.add_argument("--data_folder", required=True, help="Folder containing data")
parser.add_argument("--model_name", required=True, help="model name")
parser.add_argument("--img_type", choices=["rgb", "multi"], default="rgb", help="Type of input image")
parser.add_argument("--class_nb", type=int, default=5, help="Number of classes")
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--epochs", type=int, default=50)
params = parser.parse_args()
tf.get_logger().setLevel('ERROR')

# define directories
model_dir = "models/output/savedmodel_" + params.model_name
log_dir = "models/logs/savedmodel_" + params.model_name
ckpt_dir = "models/ckpts/savedmodel_" + params.model_name

class_nb = params.class_nb  # number of classes
inp_key = "input_img"       # model input
tgt_key = "estimated"       # model target

def create_otbtf_dataset(img, labels):
    return otbtf.DatasetFromPatchesImages(
        filenames_dict={
            "img": img,
            "labels": labels
        }
    )

def dataset_preprocessing_fn(sample):
    return {
        inp_key: sample["img"],
        tgt_key: otbtf.ops.one_hot(labels=sample["labels"], nb_classes=class_nb)
    }

def create_dataset(img, labels, batch_size=8):
    otbtf_dataset = create_otbtf_dataset(img, labels)
    
    return otbtf_dataset.get_tf_dataset(
        batch_size=batch_size,
        preprocessing_fn=dataset_preprocessing_fn,
        targets_keys=[tgt_key]
    )


def conv(inp, depth, name):
    conv_op = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=3,
        strides=1,
        activation="relu",
        padding="same",
        name=name,
        kernel_initializer=HeNormal()
    )
    return conv_op(inp)


def tconv(inp, depth, name, activation="relu"):
    tconv_op = tf.keras.layers.Conv2DTranspose(
        filters=depth,
        kernel_size=3,
        strides=1,
        activation=activation,
        padding="same",
        name=name,
        kernel_initializer=HeNormal()
    )
    return tconv_op(inp)


class FCNNModel(otbtf.ModelBase):
    
    def normalize_inputs(self, inputs):
        return {
            inp_key: tf.cast(inputs[inp_key], tf.float32) * 0.01
        }
    
    def get_outputs(self, normalized_inputs):
        norm_inp = normalized_inputs[inp_key]
        cv1 = conv(norm_inp, 64, "conv1")
        cv1 = BatchNormalization()(cv1)
        cv2 = conv(cv1, 128, "conv2") 
        cv2 = BatchNormalization()(cv2)
        cv3 = conv(cv2, 256, "conv3")
        cv3 = BatchNormalization()(cv3)
        cv1t = tconv(cv3, 128, "conv1t") + cv2
        cv1t = BatchNormalization()(cv1t)
        cv2t = tconv(cv1t, 64, "conv2t") + cv1
        cv2t = BatchNormalization()(cv2t)
        cv3t = tconv(cv2t, class_nb, "softmax_layer", "softmax")
        argmax_op = otbtf.layers.Argmax(name="argmax_layer")
        
        return {tgt_key: cv3t, "estimated_labels": argmax_op(cv3t)}


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
            model_dir,
            mode="min",
            save_best_only=True,
            monitor="val_loss"
        )
        callbacks = [save_best_cb]
        if log_dir:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        if ckpt_dir:
            ckpt_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=ckpt_dir)
            callbacks.append(ckpt_cb)
        # Train the model
        model.fit(
            ds_train,
            epochs=params.epochs,
            validation_data=ds_valid,
            callbacks=callbacks
        )
        # Final evaluation on the test dataset
        model.load_weights(model_dir)
        values = model.evaluate(ds_test, batch_size=params.batch_size)
        for metric_name, value in zip(model.metrics_names, values):
            print(f"{metric_name}: {100*value:.2f}")



patch_folder = params.data_folder 
ds_train = create_dataset(
    [(patch_folder + f"train_{params.img_type}_patches.tif")],
    [(patch_folder + "train_labels.tif")]
)
ds_train = ds_train.shuffle(buffer_size=100)

### define data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

def augment_sample(inputs, labels): 
    img = inputs[inp_key]
    img = data_augmentation(img) # only images are augmented, not labels!
    return {inp_key: img}, labels

# apply augmentation only to training dataset 
ds_train = ds_train.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

ds_valid = create_dataset(
    [(patch_folder + f"valid_{params.img_type}_patches.tif")],
    [(patch_folder + "valid_labels.tif")]
)

ds_test = create_dataset(
    [(patch_folder + f"test_{params.img_type}_patches.tif")],
    [(patch_folder + "test_labels.tif")]
)

train(params, ds_train, ds_valid, ds_test)
