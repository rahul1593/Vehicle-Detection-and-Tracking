# Vehicle Detection and Tracking

In this project, I have implemented a pipeline to find the cars in a video of moving cars on road by using HOG features and color histograms for training a Support Vector Classifier with "rbf" kernel, which is later used for classification.

## Important Files and Directories

__Files__

* Project.ipynb     : Jupyter notebook containing porject code
* writeup_report.md : Writeup report for this project
* mylib.py          : Python module containing code for feature extraction
* vehicleScan.py    : Python module containing code for processing video frames
* svm_model.7z.001/2/3 : Model saved as 3 parts(can be extracted using 7zip)
* svm_scaler.pkl    : Saved scaler session

__Directories__

* output_images : Output images for the project
* output_videos : The video outputs
* test_images   : Images used for testing
