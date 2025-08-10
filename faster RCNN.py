# Import necessary libraries
import findspark
findspark.init()  # Initialize findspark to locate Spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, monotonically_increasing_id
import pandas as pd
import cv2
import sys
import numpy as np
import os
from google.cloud import storage
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize Spark session
spark = SparkSession.builder.appName("CS777_Term_Project").getOrCreate()

# Check for correct command-line arguments
if len(sys.argv) != 3:
    print("Usage: faster_rcnn.py <parquet_path> <save_model_path>", file=sys.stderr)
    sys.exit(1)

# Get paths from command-line arguments
parquet_path = sys.argv[1]
save_model_path = sys.argv[2]

# Read the processed Parquet file into a Spark DataFrame
processed_df = spark.read.option("compression", "snappy").parquet(parquet_path)
processed_df.show(5)

# Extract unique labels from the dataset and create a mapping from label name to index
unique_labels = processed_df.select("Annotations.LabelName").distinct().rdd.flatMap(lambda x: x[0]).distinct().collect()
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
print("Label to index mapping:", label_to_index)

# PyTorch Dataset for object detection
class ObjectDetectionDataset(Dataset):
    def __init__(self, dataframe, label_to_index, transform=None):
        self.dataframe = dataframe
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row["ImageID"]
        
        # Initialize GCS client and read image from GCS
        storage_client = storage.Client()
        bucket_name, image_path = "deepak-cs777-fall2024", f"Open-Images/resized/{image_id}.jpg"
        
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(image_path)
            image_data = blob.download_as_bytes()
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return None

        if image is None:
            return None

        # Prepare bounding boxes and labels
        annotations = row['Annotations']
        boxes, labels = [], []
        for ann in annotations:
            if all(ann[i] is not None for i in range(4)) and ann[2] > ann[0] and ann[3] > ann[1]:
                boxes.append([ann[0], ann[1], ann[2], ann[3]])
                labels.append(self.label_to_index[ann[4]])

        if not boxes:
            return None

        # Convert to PyTorch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)

        return image, {"boxes": boxes, "labels": labels}

# Define image transformations
transform = transforms.Compose([transforms.ToTensor()])

# Custom collate function to handle batches of images and targets
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, targets = zip(*batch)
    return list(images), list(targets)

# Streaming DataLoader using PySpark
class PySparkDataLoader:
    def __init__(self, spark_df, rank, world_size, batch_size, transform=None):
        self.spark_df = spark_df
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.transform = transform
        self.num_batches = (self.spark_df.count() // self.world_size + self.batch_size - 1) // self.batch_size
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        start = self.current_batch * self.batch_size
        
        # Filter data for the current rank and batch
        batch_data = self.spark_df.filter((col("id") % self.world_size) == self.rank).limit(start + self.batch_size).subtract(self.spark_df.limit(start)).collect()
        
        if not batch_data:
            raise StopIteration

        dataset = ObjectDetectionDataset(pd.DataFrame(batch_data, columns=self.spark_df.columns), label_to_index, transform=self.transform)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=custom_collate_fn)
        
        self.current_batch += 1
        return next(iter(dataloader))

# Configuration for training
batch_size = 8
num_epoch = 5

# Setup for distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Cleanup for distributed training
def cleanup():
    dist.destroy_process_group()

# Function to save model to GCS
def save_model_to_gcs(local_model_path, gcs_bucket_name, gcs_blob_path):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_model_path)
    print(f"Model saved to gs://{gcs_bucket_name}/{gcs_blob_path}")

# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    metric_map = MeanAveragePrecision()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in data_loader:
            if images is None or targets is None:
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            metric_map.update(outputs, targets)
            for out in outputs:
                all_preds.extend(out['labels'].cpu().numpy())
            for tgt in targets:
                all_targets.extend(tgt['labels'].cpu().numpy())

    # Compute and print metrics
    map_score = metric_map.compute()
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    print(f"mAP: {map_score['map']:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Main training function
def model_train(rank, world_size, parquet_path, save_model_path, num_epochs=10, batch_size=16):
    print(f"Starting training on rank {rank}.")
    setup(rank, world_size)
    
    # Load and prepare data
    processed_df = spark.read.option("compression", "snappy").parquet(parquet_path)
    processed_df = processed_df.withColumn("id", monotonically_increasing_id()).repartition(world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and wrap with DDP
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Split data and create data loaders
    train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    train_loader = PySparkDataLoader(train_df, rank, world_size, batch_size, transform)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            if images is None or targets is None:
                continue
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        
        print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

        # Save model from the master process
        if rank == 0:
            local_model_path = "/tmp/faster_rcnn_model.pth"
            torch.save(model.module.state_dict(), local_model_path)
            save_model_to_gcs(local_model_path, "deepak-cs777-fall2024", "Open-Images/saved_model/faster_rcnn_model.pth")

    train_df.unpersist()
    
    # Evaluation
    if rank == 0:
        print("Evaluating model...")
        test_df.cache()
        test_loader = PySparkDataLoader(test_df, rank, world_size, batch_size, transform)
        evaluate_model(model.module, test_loader, device)
        test_df.unpersist()

    cleanup()

# Entry point for the script
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        # Spawn multiple processes for distributed training
        mp.spawn(model_train, args=(world_size, parquet_path, save_model_path, num_epoch, batch_size), nprocs=world_size, join=True)
    else:
        # Run on a single device if no multi-GPU setup
        model_train(0, 1, parquet_path, save_model_path, num_epoch, batch_size)

# Stop Spark session
spark.stop()