# Import necessary libraries
import sys
import os
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from google.cloud import storage

# Check if the correct number of command-line arguments are provided.
# This script requires paths for the input CSV, image directory, resized image directory, and output Parquet file.
if len(sys.argv) != 5:
    print("Usage: python data_preprocessing.py <csv_path> <image_dir> <resized_image_dir> <output_parquet_path>")
    sys.exit(1)

# Parse command-line arguments
csv_path = sys.argv[1]
image_dir = sys.argv[2]
resized_image_dir = sys.argv[3]
output_parquet_path = sys.argv[4]
target_size = (299, 299)  # Define the target size for resizing images.

# Initialize Spark session with optimized configurations for better performance.
spark = SparkSession.builder \
    .appName("OptimizedDataPreprocessingForML") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "21") \
    .getOrCreate()

# Load the bounding box data from the CSV file.
# The header is used to name columns, and schema inference is enabled.
data_df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Cache the DataFrame in memory to speed up subsequent transformations.
data_df.cache()

# Group bounding box annotations by ImageID.
# This aggregates all annotations for a single image into a list of structs.
grouped_df = data_df.groupBy("ImageID").agg(
    collect_list(struct("XMin", "YMin", "XMax", "YMax", "LabelName")).alias("annotations")
)

# Unpersist the original data_df to free up memory as it's no longer needed.
data_df.unpersist()

# Function to preprocess a single image and its bounding boxes.
def preprocess_image(row, image_dir, resized_image_dir, target_size):
    """
    Resizes an image, scales its bounding boxes, and saves the resized image to GCS.
    Args:
        row (Row): A Spark DataFrame row containing ImageID and annotations.
        image_dir (str): GCS path to the original images.
        resized_image_dir (str): GCS path to save resized images.
        target_size (tuple): The target (width, height) for resizing.
    Returns:
        dict: A dictionary with processed data or an error message.
    """
    from google.cloud import storage

    # Initialize GCS client within the function for use in Spark executors.
    storage_client = storage.Client()

    image_id = row["ImageID"]
    bucket_name = image_dir.split("/", 3)[2]
    image_path = f"{image_dir.split('/', 3)[3]}{image_id}.jpg"

    try:
        # Fetch the image from GCS.
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(image_path)
        image_data = blob.download_as_bytes()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return {"ImageID": image_id, "Error": f"Failed to decode image: {image_path}"}

        original_height, original_width = image.shape[:2]

        # Resize the image to the target size.
        resized_image = cv2.resize(image, target_size)

        # Calculate scaling factors for bounding boxes.
        x_scale = target_size[0] / original_width
        y_scale = target_size[1] / original_height
        processed_annotations = []

        # Scale each bounding box to match the resized image.
        for annotation in row["annotations"]:
            xmin = max(0, int(annotation["XMin"] * original_width * x_scale))
            ymin = max(0, int(annotation["YMin"] * original_height * y_scale))
            xmax = min(target_size[0], int(annotation["XMax"] * original_width * x_scale))
            ymax = min(target_size[1], int(annotation["YMax"] * original_height * y_scale))
            label = annotation["LabelName"]

            processed_annotations.append({"XMin": xmin, "YMin": ymin, "XMax": xmax, "YMax": ymax, "LabelName": label})

        # Save the resized image back to GCS.
        resized_image_path = f"{resized_image_dir}{image_id}.jpg"
        _, buffer = cv2.imencode('.jpg', resized_image)
        resized_bucket_name = resized_image_dir.split("/", 3)[2]
        resized_blob_path = f"{resized_image_dir.split('/', 3)[3]}{image_id}.jpg"

        resized_bucket = storage_client.bucket(resized_bucket_name)
        resized_blob = resized_bucket.blob(resized_blob_path)
        resized_blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

        return {"ImageID": image_id, "Annotations": processed_annotations, "FilePath": resized_image_path}

    except Exception as e:
        return {"ImageID": image_id, "Error": f"Error reading/saving image: {str(e)}"}

# Function to apply preprocessing to a partition of the RDD for better performance.
def preprocess_partition(partition):
    """Applies the preprocess_image function to a partition of the RDD."""
    return [preprocess_image(row, image_dir, resized_image_dir, target_size) for row in partition]

# Use mapPartitions for efficient parallel preprocessing.
preprocessed_rdd = grouped_df.rdd.mapPartitions(preprocess_partition)

# Define the schema for the resulting DataFrame.
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

# Convert the processed RDD back to a DataFrame using the defined schema.
processed_df = spark.createDataFrame(preprocessed_rdd, schema)

# Filter out any rows where an error occurred during preprocessing.
processed_df = processed_df.filter(col("Error").isNull()).drop("Error")

# Save the final processed data as a Parquet file for efficient storage and retrieval.
# Parquet is a columnar format that is highly optimized for Spark.
processed_df.write.parquet(output_parquet_path, mode="overwrite")

# Stop the Spark session to release cluster resources.
spark.stop()


