
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Get hog features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    features = hog(img, orientations=orient, 
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    block_norm= 'L2-Hys',
                    transform_sqrt=True, 
                    visualise=vis, feature_vector=feature_vec)
    return features


# get hog features for all image channels
def get_hog_feature_all_channels(img, orient, pix_per_cell, cell_per_block):
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    hf_ch1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    #hf_ch2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    #hf_ch3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    return hf_ch1#, hf_ch2, hf_ch3

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', include_spatial=True, spatial_size=(32, 32),
                     include_hist=True, hist_bins=32, hist_range=(0, 256),
                     include_hog=True, orient=11, pix_per_cell=8, cell_per_block=3, feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img = cv2.imread(file)
        #img_i = np.invert(img)
        for image in [img]:
            img_features = []
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            else:
                feature_image = np.copy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Apply bin_spatial() to get spatial color features
            if include_spatial:
                spatial_features = bin_spatial(feature_image, size=spatial_size)
                img_features.append(spatial_features)
            # Apply color_hist() also with a color space option now
            if include_hist:
                hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
                img_features.append(hist_features)
            # include hog features
            if include_hog:
                #hf1, hf2, hf3 = get_hog_feature_all_channels(feature_image, orient, pix_per_cell, cell_per_block)
                #hog_features = np.hstack((hf1.ravel(), hf2.ravel(), hf3.ravel()))
                hf1 = get_hog_feature_all_channels(feature_image, orient, pix_per_cell, cell_per_block)
                hog_features = hf1.ravel()
                img_features.append(hog_features)
            # Append the new feature vector to the features list
            features.append(np.concatenate(img_features))
    # Return list of feature vectors
    return features
