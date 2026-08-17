"""Performance assessment for dune cnn models"""

import os
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.sample
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import argparse
from math import sqrt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tensor_util


# Parameters parser
parser = argparse.ArgumentParser(description="Assess accuracy")
parser.add_argument("--data_folder", required=True, help="Folder containing data. Must be one of: dune-uav, dune-air, dune-ge, dune-wv")
parser.add_argument("--model_name", required=True, help="model name")
parser.add_argument("--img_type", choices=["rgb", "multi"], default="rgb", help="Type of input image")
params = parser.parse_args()

data_folder = f"deep-dunes-data/{params.data_folder}"

model_name = params.model_name
img_type = params.img_type

print(f"Accuracy for CNN: {model_name}")

shapefile_path = f"{data_folder}/vec_test.geojson"
geotiff_path = f"{data_folder}/map_{img_type}.tif"


# Load ground truth shapefile
gdf = gpd.read_file(shapefile_path)
print(f"{len(gdf)} ground truth points found.")

# Load classified GeoTIFF
with rasterio.open(geotiff_path) as src:
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
    resolution = src.transform[0]
    predicted_values = [val[0] for val in src.sample(coords)]
    predicted_mask = src.read(1)  # Read the first band of the raster

# check if crs match
if gdf.crs != src.crs:
    print(f"Warning: CRS mismatch! Ground truth CRS: {gdf.crs}, Raster CRS: {src.crs}")

plot_size = 4 # plot in square meters
plot_side_m = int(sqrt(plot_size))  # plot size in meters
plot_side_pixels = int(plot_side_m / resolution)  # plot size in pixels

# Function to get predicted value for each ground truth patch
def get_predicted_mode(predicted_mask, gdf_points, src, plot_side_pixels):
    side = int(plot_side_pixels / 2)  # window side: Half the side length in pixels

    predicted_values = []
    for i, geom in enumerate(gdf_points.geometry):
        # Calculate the window around the point
        x, y = geom.x, geom.y
        row, col = src.index(x, y)  # Get the row and column indices for the coordinates
        x_min = int(max(row - side, 0))
        x_max = int(min(row + side + 1, predicted_mask.shape[0]))
        y_min = int(max(col - side, 0))
        y_max = int(min(col + side + 1, predicted_mask.shape[1]))
        window = predicted_mask[x_min:x_max, y_min:y_max]

        # Calculate frequency of each class in the window
        vals, counts = np.unique(window, return_counts=True)
        freq_table = dict(zip(vals, counts))
        print(f"Predicted values in window around point {i} (real value: {gdf_points.iloc[i]['class']}): {freq_table}")

        # Calculate the mode of the window, ignoring NaN values
        moda = mode(window, axis=None, keepdims=False, nan_policy='omit').mode
        predicted_values.append(int(moda))
    return np.array(predicted_values, dtype=np.uint8)

# Function to compute IoU for each class 
def compute_pixelwise_iou(predicted_mask, gdf_points, src, plot_side_pixels):

    side = int(plot_side_pixels / 2)
    labels = np.array(sorted(gdf_points["class"].unique()))
    class_to_idx = {c: i for i, c in enumerate(labels)}
    n_classes = len(labels)

    tp = np.zeros(n_classes, dtype=np.uint64)
    fp = np.zeros(n_classes, dtype=np.uint64)
    fn = np.zeros(n_classes, dtype=np.uint64)

    for i, geom in enumerate(gdf_points.geometry):

        ref_class = gdf_points.iloc[i]["class"]

        x, y = geom.x, geom.y
        row, col = src.index(x, y)

        x_min = int(max(row - side, 0))
        x_max = int(min(row + side + 1, predicted_mask.shape[0]))
        y_min = int(max(col - side, 0))
        y_max = int(min(col + side + 1, predicted_mask.shape[1]))

        window = predicted_mask[x_min:x_max, y_min:y_max]

        # flatten
        pred_pixels = window.flatten()

        # remove nodata if needed
        if src.nodata is not None:
            pred_pixels = pred_pixels[pred_pixels != src.nodata]

        if len(pred_pixels) == 0:
            continue

        for cls in labels:
            cls_idx = class_to_idx[cls]
            pred_mask_cls = pred_pixels == cls
            ref_mask_cls = (cls == ref_class)
            tp[cls_idx] += np.sum(pred_mask_cls & ref_mask_cls)
            fp[cls_idx] += np.sum(pred_mask_cls & (~ref_mask_cls))
            fn[cls_idx] += np.sum((~pred_mask_cls) & ref_mask_cls)

    iou = np.divide(
        tp,
        tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp + fn) != 0
    )
    miou = np.nanmean(iou)

    return labels, iou, miou

with rasterio.open(geotiff_path) as src:
    predicted_mask = src.read(1).astype(np.uint8)
    
    # extract predicted values in the center of the ground truth points
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
    predicted_values_point = [val[0] for val in src.sample(coords)]
    #print(f"Predicted values at the ground truth points: {predicted_values_point}.")

    # extract predicted values as mode in the plot size
    predicted_values_mode = get_predicted_mode(predicted_mask, gdf, src, plot_side_pixels)

    # compute IoU
    iou_labels, pixel_iou, pixel_miou = compute_pixelwise_iou(
        predicted_mask,
        gdf,
        src,
        plot_side_pixels
    ) 

# Extract true classes (from 'class' field)
actual_values = gdf["class"].to_numpy()

# Calculate Accuracy
labels = np.unique(np.concatenate([actual_values, predicted_values_mode]))
cm = confusion_matrix(actual_values, predicted_values_mode)
oa = accuracy_score(actual_values, predicted_values_mode)
kappa = cohen_kappa_score(actual_values, predicted_values_mode)

# accuracy per class
precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=0)!=0)
recall = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1)!=0)
FScore = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision, dtype=float), where=(precision + recall)!=0)
producer_accuracy = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros_like(np.diag(cm), dtype=float), where=cm.sum(axis=1)!=0)
user_accuracy = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros_like(np.diag(cm), dtype=float), where=cm.sum(axis=0)!=0)

precision_average = np.nanmean(precision)
recall_average = np.nanmean(recall)
FScore_average = np.nanmean(FScore)


# Print results
print("\nConfusion Matrix:")
print(cm)
print(f"\nOverall Accuracy: {oa:.3f}")
print(f"Cohen's Kappa: {kappa:.3f}")
print(f"Average Precision: {precision_average:.3f}")
print(f"Average Recall: {recall_average:.3f}")
print(f"Average F-Score: {FScore_average:.3f}")
for i, label in enumerate(labels):
    print(f"Class {label} — Producer's: {producer_accuracy[i]:.2f}, User's: {user_accuracy[i]:.2f}, Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F-Score: {FScore[i]:.2f}")
print(f"\nPixel-wise Mean IoU: {pixel_miou:.3f}")
for i, cls in enumerate(iou_labels):
    print(f"Class {cls} — Pixel IoU: {pixel_iou[i]:.3f}")

# Export metrics to csv file
metrics_list = []

# Global metrics
metrics_list.append({"Metric": "Overall Accuracy", "Class": "overall", "Value": oa})
metrics_list.append({"Metric": "Cohen's Kappa", "Class": "overall", "Value": kappa})
metrics_list.append({"Metric": "Average Precision", "Class": "overall", "Value": precision_average})
metrics_list.append({"Metric": "Average Recall", "Class": "overall", "Value": recall_average})
metrics_list.append({"Metric": "Average F-Score", "Class": "overall", "Value": FScore_average})
metrics_list.append({"Metric": "Mean IoU", "Class": "overall", "Value": pixel_miou})

# Class-specific metrics
iou_dict = dict(zip(iou_labels, pixel_iou))
for i, label in enumerate(labels):
    metrics_list.append({"Metric": "Precision", "Class": label, "Value": precision[i]})
    metrics_list.append({"Metric": "Recall", "Class": label, "Value": recall[i]})
    metrics_list.append({"Metric": "F-Score", "Class": label, "Value": FScore[i]})
    metrics_list.append({"Metric": "Producer's Accuracy", "Class": label, "Value": producer_accuracy[i]})
    metrics_list.append({"Metric": "User's Accuracy", "Class": label, "Value": user_accuracy[i]})
    metrics_list.append({"Metric": "Class IoU", "Class": label, "Value": iou_dict[label]})

df_metrics = pd.DataFrame(metrics_list)
df_metrics["CNN"] = model_name
df_metrics["Image Type"] = img_type

# Export to a CSV file (one file for all cnns)
metrics_file = "models/accuracy_metrics.csv"
write_header = not os.path.exists(metrics_file)
df_metrics.to_csv(metrics_file, mode='a', header=write_header, index=False)
print(f"\nMetrics saved to {metrics_file}\n")



# Download data from TensorBoard

log_dir = f"models/logs/savedmodel_{model_name}/validation/"
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

events = ea.Tensors('epoch_loss')
losses = []
for e in events:
    # Decode the tensor value
    value = tensor_util.make_ndarray(e.tensor_proto)
    # If it's a single value, extract it as a float
    losses.append((model_name, e.step, float(value)))

df_loss = pd.DataFrame(losses, columns=['CNN', 'step', 'loss'])

# Export to a CSV file
loss_file = f'models/loss.csv'
write_header = not os.path.exists(loss_file)
df_loss.to_csv(loss_file, mode='a', header=write_header, index=False)
print(f"\nLosses saved to {loss_file}\n")
