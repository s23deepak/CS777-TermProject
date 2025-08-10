# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import boto3
import botocore
import re
import sys
from google.cloud import storage

# Initialize Spark session
# Creates a SparkSession, which is the entry point to any Spark functionality.
# The application is named "OpenImagesDownloader".
spark = SparkSession.builder.appName("OpenImagesDownloader").getOrCreate()

# Constants
# BUCKET_NAME: The name of the S3 bucket where the Open Images dataset is stored.
BUCKET_NAME = 'open-images-dataset'
# REGEX: A regular expression to parse the image path and extract the split (train/test/validation) and image ID.
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

# Get command-line arguments
# image_list_path: The path to the text file containing the list of images to download.
image_list_path = sys.argv[1] # 'image_list.txt'
# OUTPUT_BUCKET: The name of the Google Cloud Storage (GCS) bucket where the images will be uploaded.
OUTPUT_BUCKET = sys.argv[2] # Your GCS bucket name
# OUTPUT_PREFIX: The destination folder within the GCS bucket.
OUTPUT_PREFIX = sys.argv[3]  # Destination folder 

def check_and_homogenize_one_image(image):
    """
    Parses an image string to extract the split and image ID using regex.
    Args:
        image (str): The image string, e.g., 'train/000002b66c9c498e'.
    Returns:
        tuple: A tuple containing the split and image ID, or None if the format is invalid.
    """
    match = re.match(REGEX, image)
    if match:
        split, image_id = match.groups()
        return (split, image_id)
    else:
        return None

def download_and_upload_one_image(row):
    """
    Downloads an image from the S3 bucket and uploads it to the GCS bucket.
    Args:
        row (tuple): A tuple containing the split and image ID.
    Returns:
        bool: True if the download and upload were successful, False otherwise.
    """
    split, image_id = row
    
    # Create an S3 client with unsigned configuration for public access.
    s3_bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)
    
    # Create a GCS client.
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(OUTPUT_BUCKET)
    
    try:
        # Define the destination blob in GCS.
        blob = gcs_bucket.blob(f'{OUTPUT_PREFIX}{split}/{image_id}.jpg')
        # Download the image from S3 directly to the GCS blob.
        s3_bucket.download_fileobj(f'{split}/{image_id}.jpg', blob.open('wb'))
        print(f'Uploaded {OUTPUT_PREFIX}{split}/{image_id}.jpg')
        return True
    except Exception as e:
        print(f'ERROR when downloading/uploading image `{split}/{image_id}`: {str(e)}')
        return False

def download_all_images(image_list_path):
    """
    Orchestrates the download and upload process for all images in the list.
    Args:
        image_list_path (str): The path to the text file with the list of images.
    """
    # Read the image list file into a Spark RDD.
    image_list_rdd = spark.sparkContext.textFile(image_list_path)
    
    # Process the image list:
    # 1. Strip whitespace and remove the '.jpg' extension.
    # 2. Homogenize the image path to get the split and image ID.
    # 3. Filter out any invalid image formats.
    valid_images = image_list_rdd.map(lambda x: x.strip().replace('.jpg', '')) \
                                 .map(check_and_homogenize_one_image) \
                                 .filter(lambda x: x is not None)
    
    # Map the download_and_upload_one_image function to each valid image.
    # The .collect() action triggers the computation.
    results = valid_images.map(download_and_upload_one_image).collect()
    
    # Count the number of successful operations.
    successful_operations = sum(results)
    print(f"Successfully downloaded and uploaded {successful_operations} images out of {len(results)}")


# Execute the main function to start the download process.
download_all_images(image_list_path)

# Stop the Spark session to release resources.
spark.stop()
