# CS777-TermProject
Object detection pipeline using a subset of the Google Open Images Dataset
The Data is stored at https://storage.googleapis.com/openimages/web/challenge2019_downloads.html .

To download the specific images into Google Storage Bucket run the "downloading Open Images.py" script.
1st argument is path to image_list.txt that contains image_ids in the format of (test|train|validation|challenge2018)/image_id .
2nd argument is name of the Google Bucket to store.
3rd argument is path of the folder where the images are saved.

Then run the preprocessing.py script to resize the images, bounding boxes and store this infomation as parquet.
1st argument is the path of csv containg information of image_id and bounding boxes
2nd argument is the path of where images are stored
3rd argument is the path where the resized images should be stored
4th argument is the path where the parquet folder should be stored.

After this you can proceed to train and evaluate the models

SSD -
1st argument path of parquet folder
2nd argument path of resized images folder

Faster R-CNN
1st argument path of parquet folder
2nd argument path to save the model
