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


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def get_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5), draw=False):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = img.shape[0]//2
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            if draw:
                cv2.rectangle(img, (startx, starty), (endx, endy), (0,0,255), 4)
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# get prediction for each window
def get_predictions(classifier, Xscaler, img, all_windows, hog_features, y_start_pos, x_start_pos, cell_per_block, pix_per_cell,
                    inc_spatial, inc_hist, spatial_size = (32, 32), hist_bins = 32, hist_range = (0, 256)):
    predictions = []
    hog_f1 = hog_features#[0]
    #hog_f2 = hog_features[1]
    #hog_f3 = hog_features[2]
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
        #hog_ch2 = hog_f2[hys:hye, hxs:hxe].ravel()
        #hog_ch3 = hog_f3[hys:hye, hxs:hxe].ravel()
        hgf = hog_ch1#np.hstack((hog_ch1, hog_ch2, hog_ch3))
        win_features.append(hgf)
        test_features = Xscaler.transform(np.hstack(win_features).reshape(1, -1))
        predictions.append(classifier.predict(test_features))
    return np.array(predictions)

# takes a side x side image and applies threshold to it
def cool_heat(hot_img, threshold):
    hot_img[hot_img <= threshold] = 0
    return hot_img
    
# find cars in given image by using the given trained classifier and all other parameters
def find_cars(img, classifier, Xscaler, orient, pix_per_cell, cell_per_block, inc_spatial, spatial,
                inc_hist, hbins, overlap, trained_win_size, all_windows, draw_detections=False):
    # output blank heatmap of floating point numbers
    hot_img = np.zeros_like(img[:,:,0])
    hot_img.astype(np.float64)
    # convert to LUV colorspace
    dimg = None
    if draw_detections:
        dimg = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
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
        for ind in hot_indexes:
            # get corners x-y from hot windows
            tlt, brt = all_windows[win_size].windows[ind]
            #weights proportional to size of the window
            hot_img[tlt[1]:brt[1]+1, tlt[0]:brt[0]+1] = hot_img[tlt[1]:brt[1]+1, tlt[0]:brt[0]+1] + 1
            if draw_detections:
                cv2.rectangle(dimg, (tlt[0], tlt[1]), (brt[0], brt[1]), (0,0,255), 4)
    if draw_detections:
        return hot_img, dimg
    return hot_img

#draw the boxes
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
        if w//2 >= abs(bbox[0][0]-bbox[1][0]) >= 40 and abs(bbox[0][1]-bbox[1][1]) >= 40:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# process image and return the heat map for the deteced objects.
# To be utilized in multithreaded environment
def process_image(img, svc, X_scaler, pdata):
	# get the heat ap from the processed image
    heat_map = find_cars(img, svc, X_scaler, pdata["orient"],
                        pdata["pix_per_cell"], pdata["cell_per_block"], pdata["inc_spatial"], pdata["spatial"],
                        pdata["inc_hist"], pdata["hbins"], pdata["overlap"], pdata["trained_win_size"], pdata["windows"])
    return heat_map

# variable to store sum of frames
frames_sum = None
# get the frame with bounding box drawn on the input image
def get_processed_frame(img):
    global hot_frames, frames_sum, svc, X_scaler, proc_data
    # float64 type heat map
    heat_map = find_cars(img, svc, X_scaler, proc_data["orient"],
                        proc_data["pix_per_cell"], proc_data["cell_per_block"], proc_data["inc_spatial"], proc_data["spatial"],
                        proc_data["inc_hist"], proc_data["hbins"], proc_data["overlap"], proc_data["trained_win_size"], proc_data["windows"])
    
    fln = len(hot_frames)
	# if first frame, initialize sum of frames to be zero, then just add current frame to it, to get total heat
    if fln == 0:
	    frames_sum = np.copy(heat_map)
    else:
        frames_sum = frames_sum + heat_map
    # normalize the heatmap to be in range 0-255 and change data type to be uint8
    nheat_map = (frames_sum/np.max(frames_sum))*254
    nheat_map.astype(np.uint8)
    # remove unwanted heat by thresholding and smoothen the output using gaussian blur
    nheat_map = cool_heat(nheat_map, 50)
    nheat_map = cv2.GaussianBlur(nheat_map, (5,5), 0)
    # add frame to buffer and update sum of frames, addition of current frame is already done
    hot_frames.append(heat_map)
    if len(hot_frames) > 20:
        frames_sum = frames_sum - hot_frames[0]
        hot_frames.pop(0)
    # generate labels for hot boxes and draw them onto output image
    op = draw_labeled_bboxes(img, label(nheat_map))
	# superimpose heatmap on one corner fo output video
    mini_hm = cv2.resize(nheat_map, (360, 240))
    mini_hm = np.dstack((mini_hm, np.zeros_like(mini_hm), np.zeros_like(mini_hm)))
    op[0:240, op.shape[1]-360:op.shape[1]] = mini_hm
    return op
