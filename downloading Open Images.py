from pyspark.sql import SparkSession
from pyspark import SparkFiles
import boto3
import botocore
import re
import sys
from google.cloud import storage

# Initialize Spark session
spark = SparkSession.builder.appName("OpenImagesDownloader").getOrCreate()

# Constants
BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

# Get argument from job submission
image_list_path = sys.argv[1] # 'image_list.txt'
OUTPUT_BUCKET = sys.argv[2] # Your GCS bucket name
OUTPUT_PREFIX = sys.argv[3]  # Destination folder 

def check_and_homogenize_one_image(image):
    match = re.match(REGEX, image)
    if match:
        split, image_id = match.groups()
        return (split, image_id)
    else:
        return None

def download_and_upload_one_image(row):
    split, image_id = row
    
    # Create S3 client
    s3_bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)
    
    # Create GCS client
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(OUTPUT_BUCKET)
    
    try:
        # Download from S3 and upload to GCS
        blob = gcs_bucket.blob(f'{OUTPUT_PREFIX}{split}/{image_id}.jpg')
        s3_bucket.download_fileobj(f'{split}/{image_id}.jpg', blob.open('wb'))
        print(f'Uploaded {OUTPUT_PREFIX}{split}/{image_id}.jpg')
        return True
    except Exception as e:
        print(f'ERROR when downloading/uploading image `{split}/{image_id}`: {str(e)}')
        return False

def download_all_images(image_list_path):
    # Read image list file
    image_list_rdd = spark.sparkContext.textFile(image_list_path)
    
    # Process and filter the image list
    valid_images = image_list_rdd.map(lambda x: x.strip().replace('.jpg', '')) \
                                 .map(check_and_homogenize_one_image) \
                                 .filter(lambda x: x is not None)
    
    # Download images and upload to GCS
    results = valid_images.map(download_and_upload_one_image).collect()
    
    # Count successful downloads/uploads
    successful_operations = sum(results)
    print(f"Successfully downloaded and uploaded {successful_operations} images out of {len(results)}")


# Execute the main function
download_all_images(image_list_path)

# Stop the Spark session
spark.stop()
