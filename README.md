# Object Detection with Open Images Dataset

This project implements an end-to-end object detection pipeline using PySpark and PyTorch. It utilizes a subset of the Google Open Images Dataset to train and evaluate two popular object detection models: Single Shot Detector (SSD) and Faster R-CNN.

The pipeline consists of the following steps:
1.  Downloading images from the Open Images S3 bucket to a Google Cloud Storage (GCS) bucket.
2.  Preprocessing the images and their corresponding bounding box annotations.
3.  Training and evaluating object detection models (SSD and Faster R-CNN) on the preprocessed data.

## File Descriptions

-   `downloading Open Images.py`: A PySpark script to download specified images from the public Open Images S3 bucket and upload them to a Google Cloud Storage (GCS) bucket.
-   `preprocessing.py`: A PySpark script that resizes images, scales the corresponding bounding box annotations, and saves the processed data in Parquet format for efficient use in training.
-   `faster RCNN.py`: A PyTorch script for training and evaluating a Faster R-CNN model with a ResNet-50 backbone using distributed data parallel for multi-GPU training.
-   `ssd_training.py`: A PyTorch script for training and evaluating an SSDLite model with a MobileNetV3 backbone.

## Prerequisites

-   Python 3.x
-   PySpark
-   PyTorch
-   Google Cloud SDK (for GCS access)
-   Boto3 (for S3 access)
-   Pandas, OpenCV, scikit-learn, Matplotlib

## Usage

### 1. Download Images

Run the `downloading Open Images.py` script to download the images listed in a text file from the Open Images dataset on S3 to your GCS bucket.

**Arguments:**
1.  `image_list_path`: Path to a text file containing the list of image IDs to download (e.g., `train/000002b66c9c498e`).
2.  `output_bucket`: The name of your GCS bucket to store the images.
3.  `output_prefix`: The folder path within the GCS bucket where images will be saved.

**Example:**
```bash
python "downloading Open Images.py" image_list.txt your-gcs-bucket-name Open-Images/
```

### 2. Preprocess Data

Run the `preprocessing.py` script to resize the downloaded images and their bounding boxes, and then save this information as a Parquet file.

**Arguments:**
1.  `csv_path`: Path to the CSV file containing bounding box annotations.
2.  `image_dir`: GCS path where the original images are stored.
3.  `resized_image_dir`: GCS path where the resized images will be stored.
4.  `output_parquet_path`: GCS path where the output Parquet file will be stored.

**Example:**
```bash
python preprocessing.py annotations.csv gs://your-gcs-bucket-name/Open-Images/ gs://your-gcs-bucket-name/Open-Images/resized/ gs://your-gcs-bucket-name/Open-Images/processed_data.parquet
```

### 3. Train and Evaluate Models

After preprocessing, you can train either the SSD or Faster R-CNN model.

#### SSD

**Arguments:**
1.  `parquet_path`: GCS path to the preprocessed Parquet file.
2.  `image_dir`: GCS path to the directory containing the resized images.

**Example:**
```bash
python ssd_training.py gs://your-gcs-bucket-name/Open-Images/processed_data.parquet gs://your-gcs-bucket-name/Open-Images/resized/
```

#### Faster R-CNN

**Arguments:**
1.  `parquet_path`: GCS path to the preprocessed Parquet file.
2.  `save_model_path`: GCS path to save the trained model.

**Example:**
```bash
python "faster RCNN.py" gs://your-gcs-bucket-name/Open-Images/processed_data.parquet gs://your-gcs-bucket-name/Open-Images/saved_model/
```
