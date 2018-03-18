import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from mylib import *

# Class to store few common variables related to windows to be scanned in each image
class Window:
    def __init__(self):
        self.windows = None
        self.x_range_img = []
        self.y_range_img = []

# get prediction for each window
def get_predictions(classifier, Xscaler, img, all_windows, hog_features, y_start_pos, x_start_pos, cell_per_block, pix_per_cell,
                    inc_spatial, inc_hist, spatial_size = (32, 32), hist_bins = 32, hist_range = (0, 256)):
    predictions = []
    hog_f1 = hog_features[0]
    hog_f2 = hog_features[1]
    hog_f3 = hog_features[2]
    # loop over windows to get features for
    for i in range(len(all_windows)):
        win_features = []
        # image patch for which to get features
        win = all_windows[i].astype(int)
        tlt, brt = win
        img_win = np.copy(img[tlt[1]:brt[1], tlt[0]:brt[0]])
        img_win = cv2.resize(img_win, (64, 64))
        # get spatial features
        if inc_spatial:
            spf = bin_spatial(img_win, size=spatial_size)
            win_features.append(spf)
        # get color histogram features
        if inc_hist:
            hsf = color_hist(img_win, nbins=hist_bins, bins_range=hist_range)
            win_features.append(hsf)
        # get hog features
        hxs = int((tlt[0] - x_start_pos)//pix_per_cell)    #cell no. equals block no.
        hys = int((tlt[1] - y_start_pos)//pix_per_cell)
        hxe = int((brt[0] - x_start_pos)//pix_per_cell) - cell_per_block + 1   #total blocks in window range
        hye = int((brt[1] - y_start_pos)//pix_per_cell) - cell_per_block + 1
        
        # check for proper shape, modify 6 to calculation with parameters
        shp = hog_f1[hys:hye, hxs:hxe].shape
        if shp[0] != shp[1] or shp[0] < 6:
            predictions.append(0)
            continue
        if hxe > hog_f1.shape[1] or hye > hog_f1.shape[0]:
            predictions.append(0)
            continue
        hog_ch1 = hog_f1[hys:hye, hxs:hxe].ravel()
        hog_ch2 = hog_f2[hys:hye, hxs:hxe].ravel()
        hog_ch3 = hog_f3[hys:hye, hxs:hxe].ravel()
        #print(hys, hye, hxs, hxe, hog_f1[hys:hye, hxs:hxe].shape)
        hgf = np.hstack((hog_ch1, hog_ch2, hog_ch3))
        win_features.append(hgf)
        #features.append(win_features)
        #print(win_features[0].shape, win_features[1].shape, win_features[2].shape, x_start_pos)
        test_features = Xscaler.transform(np.hstack(win_features).reshape(1, -1))
        predictions.append(classifier.predict(test_features))
    return np.array(predictions)

# takes a side x side image and applies threshold to it
def cool_heat(hot_img, threshold):
    hot_img[hot_img <= threshold] = 0
    return hot_img
    
# find cars in given image by using the given trained classifier and all other parameters
def find_cars(img, classifier, Xscaler, orient, pix_per_cell, cell_per_block, inc_spatial, spatial,
                inc_hist, hbins, overlap, trained_win_size, all_windows):
    # output blank heatmap
    hot_img = np.zeros_like(img[:,:,0])
    hot_img.astype(np.int)
    # convert to LUV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # find boxes and hog features for entire area to be scanned for each window size
    for win_size in all_windows:
        x_range_img = all_windows[win_size].x_range_img
        y_range_img = all_windows[win_size].y_range_img
        # resize the image so that current window size scales to 64x64 px, i.e, the size of images in training data
        scale = trained_win_size/win_size
        # image patch for which to get hog features
        window_clip = np.copy(img[y_range_img[0]:y_range_img[1]+1, x_range_img[0]:x_range_img[1]+1])
        # resize to scale to window size of 64x64 px
        window_clip = cv2.resize(window_clip, (int(window_clip.shape[1]*scale), int(window_clip.shape[0]*scale)))
        res_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        scaled_ystart = y_range_img[0]*scale
        scaled_xstart = x_range_img[0]*scale
        # get hog features for the entire patch
        hog_features = get_hog_feature_all_channels(window_clip, orient, pix_per_cell, cell_per_block)
        # scale the windows
        curr_window = np.copy(all_windows[win_size].windows)
        curr_window = np.asarray(curr_window)*scale
        # get predictions for the searched windows
        predictions = get_predictions(classifier, Xscaler, res_img, curr_window, hog_features,
                                        scaled_ystart, scaled_xstart, cell_per_block,
                                        pix_per_cell, inc_spatial=True, inc_hist=True,
                                        spatial_size = (spatial, spatial),
                                        hist_bins = hbins, hist_range = (0, 256))
        
        # get indexes on non-zero(car) predictions
        hot_indexes = np.array((predictions.nonzero())[0])
        # add heat to hot_img
        #print(all_windows[win_sizes[i]].hot_indexes)
        for ind in hot_indexes:
            # get corners x-y from hot windows
            tlt, brt = all_windows[win_size].windows[ind]
            #weights proportional to size of the window
            hot_img[tlt[1]:brt[1]+1, tlt[0]:brt[0]+1] = hot_img[tlt[1]:brt[1]+1, tlt[0]:brt[0]+1] + 1
    return hot_img

def draw_labeled_bboxes(img, labels):
    h, w = img.shape[:2]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if 2*w//8 >= abs(bbox[0][0]-bbox[1][0]) >= w//24 and abs(bbox[0][1]-bbox[1][1]) >= h//24:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
	
def process_image(img, svc, X_scaler, pdata):
	# get the heat ap from the processed image
    heat_map = find_cars(img, svc, X_scaler, pdata["orient"],
                        pdata["pix_per_cell"], pdata["cell_per_block"], pdata["inc_spatial"], pdata["spatial"],
                        pdata["inc_hist"], pdata["hbins"], pdata["overlap"], pdata["trained_win_size"], pdata["windows"])
    return heat_map
