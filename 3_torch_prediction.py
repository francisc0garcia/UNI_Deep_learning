# -*- coding: utf-8 -*-
"""
Test predictions using online video
"""

from __future__ import division, print_function, absolute_import

'''
Load dependencies
'''
import PyTorch
import PyTorchHelpers
import numpy as np
import numpy as np
import imutils
import cv2
import time
import math

'''
define initial variables
'''

# backend [cpu|cuda|cl]
backend = 'cpu'

#name_trained_model='torch_model_v1_canny.ascii'
#lua_model = 'torch_model_V1_canny.lua'

name_trained_model='torch_model_dataset_numbers_head.ascii'
lua_model = 'torch_model_dataset_numbers.lua'

#name_trained_model='torch_model'
#lua_model = 'torch_model.lua'

#name_trained_model='torch_model_dataset_head_hand'
#lua_model = 'torch_model_dataset_head_hand.lua'

path_file_trained_model='/home/pach0/Documents/DFKI/code/DeepLearning/Hysociatea/models/'
num_classes = 12 # Number of possible classes
image_size = 28  # Pixel width and height.

#path_video = '/home/pacho-i7/Documents/DFKI/recorded/set_video_6/recorded_emb_2501.avi'
path_video = '/home/pach0/Documents/DFKI/recorded/combined_videos.mp4'

skipped_frames_tracking = 1

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
 
'''Auxiliar methods'''
def pyramid(image, scale=1.5, minSize=(28, 28)):
    # yield the original image
    yield image
 
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
 
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
            # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

cap = cv2.VideoCapture(path_video)

# load model
TorchModel = PyTorchHelpers.load_lua_class(lua_model, 'TorchModel')
torchModel = TorchModel(backend, image_size, num_classes)

torchModel.loadModel(path_file_trained_model + name_trained_model, backend)

#torchModel.saveModel(path_file_trained_model + name_trained_model + '.ascii')

# 65 65 - 2 - 0.3
font = cv2.FONT_HERSHEY_SIMPLEX


#resize_x = 512
#resize_y = 288

resize_x = 300
resize_y = 200

#resize_x = 200
#resize_y = 150

(winW, winH) = (int(resize_x/3), int(resize_y/2) )
step_size=2

#max_images = int( step_size * (resize_x / winW) * (resize_y / winH) )


#tracker:
#tracker = cv2.MultiTracker("KCF")

frame_number = 0
init_prev_frame = False

#def calback():
#    return 0
    
#cv2.namedWindow('image_edges')
#cv2.createTrackbar('low_threshold', 'image_edges', low_threshold, 255, calback)
#cv2.createTrackbar('high_threshold', 'image_edges', low_threshold*ratio, 255, calback)

while(cap.isOpened()):
    ret, frame = cap.read()

    if(ret is False):
        break

    frame = cv2.resize(frame, (resize_x, resize_y), interpolation = cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #equ = cv2.equalizeHist(image_gray)
    #image_gray = equ.copy()
    
    
    ''' Canny models '''
    image_blur = cv2.blur(image_gray, (3,3))
    image_edges = auto_canny(image_blur, sigma=0.53)
    
    image_edges = cv2.blur(image_edges, (3, 3))
    image_edges[image_edges > 0] = 255
    image_edges = 255 - image_edges

    ''''''
    '''Color models 
    image_edges = image_gray
    '''
    color_image = frame.copy()
    haar_image = frame.copy()
    tracker_image = frame.copy()
    
    index_x_y = []
    i = 0 
    for (x_w, y_w, window) in sliding_window(image_edges, stepSize=int(winW/step_size), windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        index_x_y.append([x_w, y_w])
        i += 1
        
    max_images = len(index_x_y)
    segmented_image = np.empty(shape=(max_images, image_size * image_size), dtype=float)
    i = 0 
    for (x_w, y_w) in index_x_y:
        window = cv2.resize(window, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
        batch_data = window.reshape((1, image_size * image_size)).astype(np.float16)
        segmented_image[i] = batch_data
        
    if (  frame_number % skipped_frames_tracking) == 0:
        #print(i)
        res = torchModel.predict(segmented_image)
        Y_segmented_estimated = res['prediction'].asNumpyTensor().reshape( len(segmented_image) ) 
        Y_segmented_log = res['log'].asNumpyTensor().reshape( len(segmented_image) ) 
    
        i = 0 
        
        #print(Y_segmented_log)
        for Y_estimated in Y_segmented_estimated:
            #print(Y_estimated)
            log_local = Y_segmented_log[i]
            # and log_local == 0
            x_w = int(index_x_y[i][0])
            y_w = int(index_x_y[i][1]) 
                
            #and frame.shape[1] > np.abs(index_x_y[i][0]) 
            #and frame.shape[0] > np.abs(index_x_y[i][1])
            #(Y_estimated > 10)   and
            # 
            if (  (Y_estimated > 10 ) and log_local == 0  ) :
                
                #if(np.max(Y_estimated) > threslhold_detection):
                #cv2.putText(tracker_image, str(Y_estimated), (int(x_w+winH/(2*step_size)),int(y_w+winW/(2*step_size))), font, 0.5, (0,250,0) )
                #cv2.putText(tracker_image, 'X', (int(x_w+winH/(4*step_size)),int(y_w+winW/(4*step_size))), font, 2, (0,250,0) )
                #cv2.rectangle(tracker_image, (x_w-5, y_w-5),(x_w+winW+10,y_w+winH+10), (0, 255, 0), 2)
                #print(index_x_y)
                '''tracker'''
                #if(x_w > 0 or y_w > 0):
                cv2.rectangle(tracker_image, (x_w, y_w),(x_w+winW,y_w+winH), (255, 0, 0), 2)
                cv2.putText(tracker_image, str(Y_estimated), (int(x_w+winH/(2*step_size)),int(y_w+winW/(2*step_size))), font, 0.5, (0,250,0) )
                    
                    #print (Y_segmented_estimated)
                '''
                bbox1 = (x_w, y_w, winW, winH)
                tracker = cv2.MultiTracker("KCF")
                tracker.add(tracker_image, (bbox1))
                '''
            else:
                cv2.rectangle(tracker_image, (x_w, y_w),(x_w+winW,y_w+winH), (0, 255, 0), 2)
                cv2.putText(tracker_image, str(Y_estimated), (int(x_w+winH/(2*step_size)),int(y_w+winW/(2*step_size))), font, 0.5, (0,255,0) )
                    
                ''' '''
            i += 1
    '''
    ok, boxes = tracker.update(tracker_image)
    
    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(tracker_image, p1, p2, (255,0,0), 2)
    '''
    edge_color = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2BGR)
    res = np.hstack((tracker_image, edge_color)) #stacking images side-by-side
    #cv2.imshow("tracker_image", tracker_image)  
    #cv2.imshow("image_gray", image_edges)  
    cv2.imshow("res", res)  
       
    #cv2.imshow("Window", color_image)
    cv2.waitKey(1)
    
    frame_number += 1
    prev_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()