"""Semantic segmentation of RGB images"""
from mymetrics import FScore
import otbtf
import tensorflow as tf
import argparse
import logging
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import HeNormal

parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("--model_dir", required=True, help="model directory")
parser.add_argument("--log_dir", help="log directory")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_dir", help="Directory for checkpoints")
parser.add_argument("--output_file", type=str, default="output.txt", help="Output file for logging")
params = parser.parse_args()
tf.get_logger().setLevel('ERROR')

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(params.output_file),  # Usa il nome del file passato come argomento
        logging.StreamHandler()                    # Output anche nel terminale
    ]
)

class_nb = 3             # number of classes
inp_key_ge = "input_ge"  # model input ge
tgt_key = "estimated"    # model target


def create_otbtf_dataset(ge, labels):
    return otbtf.DatasetFromPatchesImages(
        filenames_dict={
            "ge2": ge,
            "labels2": labels
        }
    )

def dataset_preprocessing_fn(sample):
    return {
        inp_key_ge: sample["ge2"],
        tgt_key: otbtf.ops.one_hot(labels=sample["labels2"], nb_classes=class_nb)
    }

def create_dataset(ge, labels, batch_size=8):
    otbtf_dataset = create_otbtf_dataset(ge, labels)
    return otbtf_dataset.get_tf_dataset(
        batch_size=batch_size,
        preprocessing_fn=dataset_preprocessing_fn,
        targets_keys=[tgt_key]
    )


def conv(inp, depth, name, strides=1):
    conv_op = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=3,
        strides=strides,
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
            inp_key_ge: tf.cast(inputs[inp_key_ge], tf.float32) * 0.01
        }
    
    def get_outputs(self, normalized_inputs):
        norm_inp_ge = normalized_inputs[inp_key_ge]
        
        cv1 = conv(norm_inp_ge, 32, "conv1")
        cv1 = BatchNormalization()(cv1)
        cv2 = conv(cv1, 64, "conv2")
        cv2 = BatchNormalization()(cv2)
        cv3 = conv(cv2, 128, "conv3")
        cv3 = BatchNormalization()(cv3)
        cv4 = conv(cv3, 256, "conv4")
        cv4 = BatchNormalization()(cv4)
        cv1t = tconv(cv4, 128, "conv1t") + cv3
        cv1t = BatchNormalization()(cv1t)
        cv2t = tconv(cv1t, 64, "conv2t") + cv2
        cv2t = BatchNormalization()(cv2t)
        cv3t = tconv(cv2t, 32, "conv3t") + cv1
        cv3t = BatchNormalization()(cv3t)
        cv4t = tconv(cv3t, class_nb, "softmax_layer", "softmax")

        argmax_op = otbtf.layers.Argmax(name="argmax_layer")

        return {tgt_key: cv4t, "estimated_labels": argmax_op(cv4t)}


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

train(params, ds_train, ds_valid, ds_test)