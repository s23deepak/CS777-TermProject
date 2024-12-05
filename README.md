# CS777-TermProject
Object detection pipeline using a subset of the Google Open Images Dataset<br/>
The Data is stored at https://storage.googleapis.com/openimages/web/challenge2019_downloads.html .<br/>

To download the specific images into Google Storage Bucket run the "downloading Open Images.py" script.<br/>
1st argument is path to image_list.txt that contains image_ids in the format of (test|train|validation|challenge2018)/image_id .<br/>
2nd argument is name of the Google Bucket to store.<br/>
3rd argument is path of the folder where the images are saved.<br/>

Then run the preprocessing.py script to resize the images, bounding boxes and store this infomation as parquet.<br/>
1st argument is the path of csv containg information of image_id and bounding boxes<br/>
2nd argument is the path of where images are stored<br/>
3rd argument is the path where the resized images should be stored<br/>
4th argument is the path where the parquet folder should be stored.<br/>

After this you can proceed to train and evaluate the models<br/>

SSD -<br/>
1st argument path of parquet folder<br/>
2nd argument path of resized images folder<br/>

Faster R-CNN<br/>
1st argument path of parquet folder<br/>
2nd argument path to save the model<br/>
