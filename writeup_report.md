
# Vehicle Detection and Tracking

In this project, I'm going to draw bounding boxes around the cars in the video of moving cars on road.

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
- Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
- Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
- Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.


## Histogram of Oriented Gradients (HOG)

[//]: # (Image References)

[image0]: ./output_images/car_n_car.JPG "Car-Not-Car Image"
[image1]: ./output_images/car_hog.JPG "Car Hog features"
[image2]: ./output_images/notcar_hog.JPG "Not Car hog features"
[image3]: ./output_images/roi_bb.JPG "Region of Interset for sliding window search"
[image4]: ./output_images/hmap_bb.JPG "Output of pipeline"
[image5]: ./output_images/multiple_bb.JPG "Multiple Detections"
[image6]: ./output_images/labels.JPG "Label"
[image7]: ./output_images/combined_bb.JPG "Single Bounding Box"
[op_vid]: ./output_videos/project_video.mp4 "Output Video"

#### 1. Extracting HOG features from the training images

The code for this step is contained in cell number 4 of the python notebook. The function used for feature extraction (`extract_features`) is present in `mylib.py` at line number 61.

I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:
![alt text][image0]

I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.

Following example uses the LUV color space and HOG parameters of orientations=11, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):

   __Car Image Features__
![alt text][image1]

   __Not Car Image__
![alt text][image2]

#### 2. Choice of HOG parameters and other features.

I tried various combinations of parameters and color spaces. RGB colorspace could not give high accuracy on test dataset. HSV and HLS colorspaces were marginally better than RGB, but YUV and LUV colorspace were able to give highest accuracies with LUV giving the highest accuracy with less orientiations. *cells_per_block* and *pixels_per_cell* for HOG were set to __2__ and __8__ which gave highest accuracy with __11__ orientations.

Along with HOG, I also used spatial features with size of _16x16_ and color histogram features with bin size _32_ to get more accurate output.

#### 3. Training the classifier

Code for training the classifier can be found in cell number 5 of the python notebook. I have used _rbf_ kernel with parameter _C_ set to __10__ for the SVM classifier (SVC).

Initially I used linear classifier, since its training time and prediction time was very low. But I was getting a lot of false positives while classifying the images. So, finally I used _rbf_ kernel which heavily reduced the number of false positives and gave accurate results when used with parameter _C=10_.

## Sliding Window Search

#### 1. Sliding window implementation

I implemented sliding windows for window sizes of __96x96__ and __72x72__ pixels. Code for the same can be found in cell number 6 of python notebook. The function called for creating windows(`get_windows`) is defined at line number 20 in file `vehicleScan.py`.

The size of images in training dataset is 64x64 pixels. So the image to be classified is scaled according to the window size before classification. I choose two different window sizes, i.e 96 and 72 to clearly identify the vehicles appearing close and far away respectively in the scanned image.

Windows are created at specific offset in x and y axis so that they do not do blind search in the entire image. Following image shows the area for search:

![alt text][image3]

#### 2. Pipeline for Image

Ultimately I searched on two scales using LUV single channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. For each output heatmap is generated for multiple detections and thresholded after normalisation to get rid of false positives and get tighter bounding boxes. Here are some example images:

__Multiple Detections__
![alt text][image5]

__Combined Result__
![alt text][image7]

To improve performance, I create the windows only once and use them repeatedly in each frame after scaling. I extract HOG features for the target image only once and subsample them for each window to be scanned.


## Video Implementation

#### 1.  Video Pipeline Output

The entire pipeline for video is same as for single image, except for the function called to get the output for each frame. This function (`get_processed_frame`) is defined in file `vehicleScan.py` at line number 189.

Here's a link to my video result.
![alt text][op_vid]

#### 2. Pipeline Implementation.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

To get a smooth output and remove false positives from the video, summing of past 20 frames' heatmaps is done. Sum is maintained as global variable and is updated by adding new frame's heatmap and subtracting oldest one's heatmap. Along with thresholding the summed heatmap after normalising it to have values in range 0-255, helps in removing transient false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:

Here are 3 different frames and their corresponding heatmaps:
![alt text][image4]

Here is the output of scipy.ndimage.measurements.label() on the integrated heatmap from a single frames:
![alt text][image6]


## Discussion

This pipeline searches for cars on one side of the input frame which is hardcode at the moment. Also the entire process is single threaded which doesn't give instant output in case if it is used in real time. Also, the thresholds does not account for varying light in the image.

So, following improvements can be made to this pipeline:
- Use the angle of turn(by detecting the lane lines) to determine the region of interest for sliding window search
- Use median color in image frame to decide the threshold for the heatmap to improve detection in varying light
- Use multiprocessing for simultaneously processing the input frames.
