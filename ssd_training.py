import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from google.cloud import storage
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import random

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Command-line arguments
if len(sys.argv) != 3:
    print("Usage: python ssd_training.py <parquet_path> <image_dir>")
    sys.exit(1)

# GCS paths
parquet_path = sys.argv[1]
resized_image_dir = sys.argv[2]

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SSD Training") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Function to read Parquet data and filter rows with existing images
def read_parquet_with_existing_images(parquet_path, image_dir):
    try:
        df = spark.read.parquet(parquet_path)
        bucket_name, prefix = image_dir.split("/", 3)[2:]
        bucket = storage_client.bucket(bucket_name)
        image_blobs = bucket.list_blobs(prefix=prefix)
        gcs_images = {blob.name.split("/")[-1].replace(".jpg", "") for blob in image_blobs if blob.name.endswith(".jpg")}
        df = df.filter(col("ImageID").isin(*list(gcs_images)))
        return df.toPandas()
    except Exception as e:
        print(f"Error reading or filtering Parquet data: {e}")
        sys.exit(1)

# Load the Parquet data and filter missing images
data_df = read_parquet_with_existing_images(parquet_path, resized_image_dir)

# Create a label mapping dictionary
unique_labels = set(ann["LabelName"] for anns in data_df["Annotations"] for ann in anns)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Helper function to validate bounding boxes
def validate_boxes(boxes):
    valid_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        if x_max > x_min and y_max > y_min:
            valid_boxes.append([x_min, y_min, x_max, y_max])
    return valid_boxes

# Custom Dataset for Object Detection
class ObjectDetectionDataset(Dataset):
    def __init__(self, data_df, image_dir, label_mapping):
        self.data_df = data_df
        self.image_dir = image_dir
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        try:
            row = self.data_df.iloc[idx]
            image_id = row["ImageID"]
            annotations = row["Annotations"]

            bucket_name, prefix = self.image_dir.split("/", 3)[2:]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"{prefix}{image_id}.jpg")
            image_data = np.frombuffer(blob.download_as_bytes(), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {image_id}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = []
            labels = []
            for ann in annotations:
                boxes.append([ann["XMin"], ann["YMin"], ann["XMax"], ann["YMax"]])
                labels.append(self.label_mapping[ann["LabelName"]])
            valid_boxes = validate_boxes(boxes)

            if not valid_boxes:
                raise ValueError(f"No valid bounding boxes for image {image_id}")

            boxes = torch.tensor(valid_boxes, dtype=torch.float32)
            labels = torch.tensor(labels[:len(valid_boxes)], dtype=torch.int64)
            image = F.to_tensor(image)

            return image, {"boxes": boxes, "labels": labels}
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None

# Filter out None results and cache the dataset
dataset = [
    item
    for item in map(ObjectDetectionDataset(data_df, resized_image_dir, label_mapping).__getitem__, range(len(data_df)))
    if item
]
del data_df

# Partition the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=os.cpu_count(), pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=os.cpu_count(), pin_memory=True)

# Initialize SSD model with weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
model.train()

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
num_epochs = 5
checkpoint_path = "/tmp/ssd_checkpoint.pth"

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    scheduler.step()
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation
metrics = {"IoU": [], "Precision": [], "Recall": [], "F1": []}       ##############

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

model.eval()
with torch.no_grad():
    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].detach().cpu().numpy()
            pred_scores = output["scores"].detach().cpu().numpy()
            pred_labels = output["labels"].detach().cpu().numpy()
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()

            for gt_box in gt_boxes:
                ious = [calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
                metrics["IoU"].append(max(ious) if ious else 0)

            tp = sum(1 for iou in metrics["IoU"] if iou > 0.5)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            metrics["Precision"].append(precision)
            metrics["Recall"].append(recall)
            metrics["F1"].append(f1_score)

print(f"Average IoU: {np.mean(metrics['IoU']):.4f}")
print(f"Average Precision: {np.mean(metrics['Precision']):.4f}")
print(f"Average Recall: {np.mean(metrics['Recall']):.4f}")
print(f"Average F1 Score: {np.mean(metrics['F1']):.4f}")

# Visualization function with better readability
def visualize_predictions(image, gt_boxes, pred_boxes, pred_scores, pred_labels, output_path, confidence_threshold=0.5):
    """Visualize ground truth and predicted bounding boxes with confidence filtering."""
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot ground truth boxes in green
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor="green",
                fill=False,
                linewidth=2,
                label="Ground Truth"
            )
        )

    # Filter predictions based on confidence threshold
    high_conf_indices = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[high_conf_indices]
    pred_labels = pred_labels[high_conf_indices]
    pred_scores = pred_scores[high_conf_indices]

    # Plot predicted boxes in red with scores
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor="red",
                fill=False,
                linewidth=2
            )
        )
        plt.text(
            x_min,
            y_min - 10,
            f"{label} ({score:.2f})",
            color="red",
            fontsize=10,
            backgroundcolor="white",
        )

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

# Updated visualization process
random_idx = random.randint(0, len(test_loader) - 1)
for idx, (images, targets) in enumerate(test_loader):
    if idx == random_idx:
        image = images[0].to(device)
        gt_boxes = targets[0]["boxes"].cpu().numpy()
        outputs = model([image])
        pred_boxes = outputs[0]["boxes"].detach().cpu().numpy()
        pred_scores = outputs[0]["scores"].detach().cpu().numpy()
        pred_labels = outputs[0]["labels"].detach().cpu().numpy()
        break

# Improved visualization and saving the output
local_image_path = "/tmp/visualized_image.jpg"
visualize_predictions(image, gt_boxes, pred_boxes, pred_scores, pred_labels, local_image_path)

# Upload the image to GCS
bucket_name, prefix = "met777_term_project", "output"
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(f"{prefix}/visualized_image.jpg")
blob.upload_from_filename(local_image_path)

print(f"Visualization saved to GCS: gs://{bucket_name}/{prefix}/visualized_image.jpg")
