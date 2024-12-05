import sys
import os
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from google.cloud import storage

# Check if correct number of arguments are provided
if len(sys.argv) != 5:
    print("Usage: python data_preprocessing.py <csv_path> <image_dir> <resized_image_dir> <output_parquet_path>")
    sys.exit(1)

# Parse command-line arguments
csv_path = sys.argv[1]  # Path to the CSV file
image_dir = sys.argv[2]  # Directory containing images
resized_image_dir = sys.argv[3]  # Directory to save resized images
output_parquet_path = sys.argv[4]  # Path to save the output Parquet file
target_size = (299, 299)  # Target resolution for resizing

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("OptimizedDataPreprocessingForML") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "21") \
    .getOrCreate()

# Load bounding box data
data_df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Cache data to avoid recomputation during transformations
data_df.cache()

# Group bounding boxes by ImageID
grouped_df = data_df.groupBy("ImageID").agg(
    collect_list(struct("XMin", "YMin", "XMax", "YMax", "LabelName")).alias("annotations")
)

# Unpersist (uncache) after grouping is done
data_df.unpersist()

# Function to preprocess images and bounding boxes
def preprocess_image(row, image_dir, resized_image_dir, target_size):
    from google.cloud import storage  # Import here to avoid pickling issues

    # Initialize GCS client inside the function
    storage_client = storage.Client()

    image_id = row["ImageID"]
    bucket_name, image_path = image_dir.split("/", 3)[2], image_dir.split("/", 3)[3] + f"/{image_id}.jpg"

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
        return {"ImageID": image_id, "Error": f"Image not found: {image_path}"}

    original_height, original_width = image.shape[:2]

    # Resize the image
    resized_image = cv2.resize(image, target_size)

    # Scale bounding boxes
    x_scale = target_size[0] / original_width
    y_scale = target_size[1] / original_height
    processed_annotations = []

    for annotation in row["annotations"]:
        xmin = max(0, int(annotation["XMin"] * original_width * x_scale))
        ymin = max(0, int(annotation["YMin"] * original_height * y_scale))
        xmax = min(target_size[0], int(annotation["XMax"] * original_width * x_scale))
        ymax = min(target_size[1], int(annotation["YMax"] * original_height * y_scale))
        label = annotation["LabelName"]

        processed_annotations.append({"XMin": xmin, "YMin": ymin, "XMax": xmax, "YMax": ymax, "LabelName": label})

    # Save resized image to GCS
    resized_image_path = f"{resized_image_dir}/{image_id}.jpg"
    _, buffer = cv2.imencode('.jpg', resized_image)
    resized_bucket_name = resized_image_dir.split("/", 3)[2]
    resized_blob_path = resized_image_dir.split("/", 3)[3] + f"/{image_id}.jpg"

    try:
        resized_bucket = storage_client.bucket(resized_bucket_name)
        resized_blob = resized_bucket.blob(resized_blob_path)
        resized_blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    except Exception as e:
        return {"ImageID": image_id, "Error": "Failed to save resized image"}

    return {"ImageID": image_id, "Annotations": processed_annotations, "FilePath": resized_image_path}


# Apply preprocessing using mapPartitions for faster execution
def preprocess_partition(partition):
    return [preprocess_image(row, image_dir, resized_image_dir, target_size) for row in partition]

# Use mapPartitions for efficient preprocessing
preprocessed_rdd = grouped_df.rdd.mapPartitions(preprocess_partition)

# Define schema for the processed data
annotations_schema = ArrayType(
    StructType([
        StructField("XMin", IntegerType(), True),
        StructField("YMin", IntegerType(), True),
        StructField("XMax", IntegerType(), True),
        StructField("YMax", IntegerType(), True),
        StructField("LabelName", StringType(), True),
    ])
)

schema = StructType([
    StructField("ImageID", StringType(), True),
    StructField("Annotations", annotations_schema, True),
    StructField("FilePath", StringType(), True),
    StructField("Error", StringType(), True)
])

# Convert RDD to DataFrame
processed_df = spark.createDataFrame(preprocessed_rdd, schema)

# Filter out images with errors
processed_df = processed_df.filter(col("Error").isNull()).drop("Error")

# Save the processed data as a Parquet file
processed_df.write.parquet(output_parquet_path, mode="overwrite")

# Stop Spark session
spark.stop()

