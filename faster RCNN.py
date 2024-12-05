import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct
from pyspark.sql.functions import monotonically_increasing_id

import pandas as pd
import cv2
import sys
import numpy as np
import multiprocessing
import io

from google.cloud import storage
# from google.cloud import dataproc_v1
# from google.cloud import compute_v1

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision

from sklearn.metrics import precision_score, recall_score, f1_score

spark = SparkSession.builder \
    .appName("CS777_Term_Project") \
    .getOrCreate()

if len(sys.argv) != 3:
    print("Usage: wordcount <file>", file=sys.stderr)

parquet_path = sys.argv[1]
save_model_path = sys.argv[2]

# Read the saved Parquet file
processed_df = spark.read.option("compression", "snappy").parquet(parquet_path)
processed_df.show(5)

# Extract unique labels from your dataset
unique_labels = processed_df.select("Annotations.LabelName").distinct().rdd.flatMap(lambda x: x[0]).distinct().collect()

# Create a label mapping dictionary
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  
print(label_to_index)

# Step 3: PyTorch Dataset
class ObjectDetectionDataset(Dataset):
  def __init__(self, dataframe, label_to_index, transform=None):
    self.dataframe = dataframe
    self.label_to_index = label_to_index
    self.transform = transform

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    row = self.dataframe.iloc[idx]
    image = None
    storage_client = storage.Client(project="deepak-cs777-fall2024")
    image_id = row["ImageID"]
    bucket_name, image_path = "deepak-cs777-fall2024", f"Open-Images/resized/{image_id}.jpg"

    # Read image from GCS
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(image_path)
        image_data = blob.download_as_bytes()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"ImageID": image_id, "Error": f"Image not found or inaccessible: {image_path}"}

    if image is None:
        print(f"ImageID: {image_id}, Error: Image not found: {image_path}")
        return None
    annotations = row['Annotations']

    # Prepare bounding boxes and labels
    boxes = []
    labels = []
    for ann in annotations:
        if ann[0] is not None and ann[1] is not None and ann[2] is not None and ann[3] is not None:
            if ann[2] > ann[0] and ann[3] > ann[1]:
                boxes.append([ann[0], ann[1], ann[2], ann[3]])
                labels.append(self.label_to_index[ann[4]])
            else:
                print(f"Skipping invalid bounding box dimensions: {ann}")
        else:
            print(f"Skipping annotation with None values: {ann}")

    if len(boxes) == 0:  # Skip samples with no valid annotations
      print(f"Skipping sample {row['FilePath']} with no valid bounding boxes.")
      return None

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    image = self.transform(image)

    return image, {"boxes": boxes, "labels": labels}



# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

def custom_collate_fn(batch):
  # Filter out any None values or invalid samples
  batch = [item for item in batch if item is not None and isinstance(item, tuple) and len(item) == 2]

  if len(batch) == 0:  # Check if the batch is empty
     return None, None
  
  images = []
  targets = []
  for item in batch:
     if isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
      images.append(item[0]) # List of images
      targets.append(item[1]) # List of annotations (bounding boxes and labels)
  
  if len(images) == 0:
     return None, None
  
  return images, targets


# Streaming DataLoader
class PySparkDataLoader:
  def __init__(self, spark_df, rank, world_size, batch_size, transform=None):
    self.spark_df = spark_df
    self.rank = rank
    self.world_size = world_size
    self.batch_size = batch_size
    self.transform = transform
    self.num_batches = (self.spark_df.count() + self.batch_size - 1) // self.batch_size
    self.current_batch = 0

  def __iter__(self):
    self.current_batch = 0  # Reset for each iteration
    return self

  def __next__(self):
    if self.current_batch >= self.num_batches:
      raise StopIteration
        
    start = self.current_batch * self.batch_size
    end = start + self.batch_size

    # Select rows assigned to this process
    batch_data = (
    self.spark_df
    .filter((col("id") % self.world_size) == self.rank)
    .limit(end)
    .subtract(self.spark_df.limit(start))
    .collect()
    )
    if not batch_data:
      raise StopIteration

    dataset = ObjectDetectionDataset(pd.DataFrame(batch_data, columns =["ImageID","Annotations","FilePath","id"]), label_to_index, transform=self.transform)
    # Use DistributedSampler for data partitioning
    sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank= self.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, collate_fn=custom_collate_fn)
        
    self.current_batch += 1
    return next(iter(dataloader)) # Return first (and only) DataLoader batch


# Use PySparkDataLoader
batch_size = 8
num_epoch = 5

def setup(rank, world_size):
    """Initialize the process group for distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend= "nccl" ,
        # init_method="tcp://127.0.0.1:12355",
        rank=rank,
        world_size=world_size
        )

def cleanup():
    dist.destroy_process_group()

def save_model_to_gcs(local_model_path, gcs_bucket_name, gcs_blob_path):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_path)

    # Upload the model
    blob.upload_from_filename(local_model_path)
    print(f"Model saved to gs://{gcs_bucket_name}/{gcs_blob_path}")

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model using IoU, mAP, Precision, Recall, and F1-Score.
    """
    model.eval()  # Set model to evaluation mode
    # metric_map = MeanAveragePrecision()  # mAP metric
    iou_list = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            if images is None or targets is None:
                continue

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                # IoU calculation
                for p_box, t_box in zip(pred['boxes'], target['boxes']):
                    iou = torchvision.ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0))
                    iou_list.append(iou.item())

                # For Precision, Recall, F1
                pred_labels = pred['labels'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                all_preds.extend(pred_labels)
                all_targets.extend(true_labels)

                # Update mAP metric
                metric_map.update([pred], [target])

    # Metrics calculation
    average_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    # map_score = metric_map.compute()
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    print(f"IoU: {average_iou:.4f}")
    # print(f"mAP: {map_score['map_50']:.4f}")  # mAP at IoU=0.5
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return average_iou, map_score, precision, recall, f1

# Step 4: Train Faster R-CNN Model
def model_train(rank, world_size, parquet_path, save_model_path, num_epochs=10, batch_size=16):
  print(f"Rank {rank} world {world_size} processed_df_path {parquet_path} epoch {num_epochs} batch {batch_size}")
  setup(rank, world_size)
  processed_df = spark.read.option("compression", "snappy").parquet(parquet_path)
  processed_df = processed_df.withColumn("id", monotonically_increasing_id())
  processed_df = processed_df.repartition(4)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Rank {rank} device {device}")
  model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
  if torch.cuda.is_available():
      print(f"DDP model")
      model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
  else :
      print(f"device {device.type} DataParallel")
      model = DataParallel(model)

  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  # Initialize PySpark DataLoader
  transform = transforms.Compose([transforms.ToTensor()])
  data_loader = PySparkDataLoader(processed_df, rank, world_size, batch_size, transform)
  processed_df_train, processed_df_test = processed_df.randomSplit([0.8, 0.2], seed=42)
  processed_df_train.cache()
  data_loader_train = PySparkDataLoader(processed_df_train, rank, world_size, batch_size, transform)
  model.train()
  grad_accum_steps = 4
  for epoch in range(num_epochs):
    epoch_loss = 0
    # optimizer.zero_grad()
    for i, (images, targets) in enumerate(data_loader_train):
      if images is None or targets is None:  # Skip empty batches
        continue
      if any(len(target['boxes']) == 0 for target in targets):
        continue
      

      images = [image.to(device) for image in images]
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      losses.backward()
      if (i + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

      epoch_loss += losses.item()
      # print(f"epoch {epoch} and epoch loss {epoch_loss}")

         
    print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    if rank == 0:  # Save model only on master node
      # client = storage.Client()
      bucket_name = "deepak-cs777-fall2024"
      destination_blob_name = "Open-Images/saved_model/faster_rcnn_model.pth"
      local_model_path = "tmp/faster_rcnn_model.pth"
      torch.save(model.state_dict(), local_model_path)  # Save locally
      save_model_to_gcs(
        local_model_path,
        "deepak-cs777-fall2024",
        "Open-Images/saved_model/faster_rcnn_model.pth"
      )
      print(f"Model saved to gs://{bucket_name}/{destination_blob_name}")
  processed_df_train.unpersist()
  processed_df_test.cache()
  data_loader_test = PySparkDataLoader(processed_df_test, rank, world_size, batch_size, transform)
  print(f"Evaluating Model...")
  evaluate_model(model, data_loader_test, device)  
  cleanup()



if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs
    if world_size > 1 :
        mp.spawn(model_train, args=(world_size, parquet_path, save_model_path, num_epoch, batch_size), nprocs=world_size, join=True)
    else:
        model_train(0, 1, parquet_path, save_model_path, num_epoch, batch_size)
# Stop Spark session
spark.stop()