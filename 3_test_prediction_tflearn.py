# -*- coding: utf-8 -*-
"""
3_test_prediction using tflearn model

"""
from __future__ import division, print_function, absolute_import

'''
Load dependencies
'''
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import numpy as np
import imutils
import cv2

'''
define initial variables
'''
#name_trained_model='color_mixed_classes.tfl'
name_trained_model='canny_model_full_V2.tfl'
path_file_trained_model='/home/pacho-i7/Documents/DFKI/code/DeepLearning/Hysociatea/models/'
num_classes = 12 # Number of possible classes
image_size = 28  # Pixel width and height.

low_threshold = 14
ratio = 3
kernel_size = 3
aperture_size = 3

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

cap = cv2.VideoCapture('/home/pacho-i7/Documents/recorded/recorded_08_04_2016/recorded_emb_2501.avi')
#cap = cv2.VideoCapture(0)


'''
TensorFlow - tflearn
'''



''' Building convolutional network - model MNIST
network = input_data(shape=[None, image_size, image_size, 1], name='input')
network = conv_2d(network, 32, 15, activation='tanh', regularizer="L2")
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 64, 5, activation='tanh', regularizer="L2")
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.4)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.4)
network = fully_connected(network, num_classes, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.05,
                     loss='categorical_crossentropy', name='target')
'''


''' Building convolutional network - model CIFAR 
network = input_data(shape=[None, image_size, image_size, 1], name='input')
network = conv_2d(network, 32, 3, activation='tanh', name='conv_2d_1')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='tanh',  name='conv_2d_2')
network = conv_2d(network, 128, 3, activation='tanh',  name='conv_2d_2')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='tanh', name='fully_connected_1')
network = dropout(network, 0.5)
network = fully_connected(network, num_classes, activation='softmax', name='fully_connected_2')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
'''

''' Personalized convolutional network'''
network = input_data(shape=[None, image_size, image_size, 1], name='input')
network = conv_2d(network, 32, 3, activation='tanh',  name='conv_2d_2')
network = max_pool_2d(network, 2)
network = fully_connected(network, 64, activation='tanh', name='fully_connected_1')
network = dropout(network, 0.6)
network = fully_connected(network, num_classes, activation='softmax', name='fully_connected_2')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
''''''


model = tflearn.DNN(network, tensorboard_verbose=0)

''' Load precomputed model '''
model.load(path_file_trained_model + name_trained_model)
''''''

'''
# print weigths and bias
fully_connected_1_vars = tflearn.variables.get_layer_variables_by_name('fully_connected_1')

print("fully_connected_1 layer weights:")
print(model.get_weights(fully_connected_1_vars[0]))

print("fully_connected_1 layer bias:")
print(model.get_weights(fully_connected_1_vars[1]))

fully_connected_2_vars = tflearn.variables.get_layer_variables_by_name('fully_connected_2')

print("fully_connected_2 layer weights:")
print(model.get_weights(fully_connected_2_vars[0]))

print("fully_connected_2 layer bias:")
print(model.get_weights(fully_connected_2_vars[1]))
'''
# 65 65 - 2 - 0.3
font = cv2.FONT_HERSHEY_SIMPLEX
(winW, winH) = (65, 65)
threslhold_detection=0.60
scaling_image = 0.3
step_size=2

# haar cascade
face_cascade = cv2.CascadeClassifier('/home/pacho-i7/Documents/software/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('/home/pacho-i7/Documents/software/opencv/opencv/data/haarcascades/haarcascade_upperbody.xml')

#tracker:
#tracker = cv2.MultiTracker("MIL")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1]*scaling_image), int(frame.shape[0]*scaling_image)), interpolation = cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ''' Canny models '''
    image_blur = cv2.blur(image_gray, (kernel_size,kernel_size))
    image_edges = cv2.Canny(image_blur, low_threshold, low_threshold*ratio, aperture_size)
    ''''''
    '''Color models 
    image_edges = image_gray
    '''
    color_image = frame.copy()
    haar_image = frame.copy()
    tracker_image = frame.copy()
    
    
    '''Haar detector'''
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(haar_image, (x,y), (x+w,y+h), (255,0,0),2)
    
    cv2.imshow('haar_image', haar_image)
    
    
    for (x_w, y_w, window) in sliding_window(image_edges, stepSize=int(winW/step_size), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        #if window.shape[0] != winH or window.shape[1] != winW:
        #    continue
        
        window = cv2.resize(window, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
        
        batch_data = window.reshape([-1, image_size, image_size, 1])
        Y_estimated = model.predict(batch_data)

        if( np.argmax(Y_estimated) > 9 and np.max(Y_estimated) > threslhold_detection):
        #if(np.max(Y_estimated) > threslhold_detection):
            cv2.rectangle(color_image, (x_w, y_w),(x_w+winW,y_w+winH), (255,0,0), 2)
            if( np.argmax(Y_estimated) > 9 ):
                cv2.putText(color_image, str(np.argmax(Y_estimated)), (int(x_w+winH/(2*step_size)),int(y_w+winW/(2*step_size))), font, np.max(Y_estimated), (0,0,250) )
            else:
                cv2.putText(color_image, str(np.argmax(Y_estimated)), (int(x_w+winH/(2*step_size)),int(y_w+winW/(2*step_size))), font, np.max(Y_estimated), (0,250,0) )
            
        '''tracker
        bbox1 = (x_w, y_w, winW, winH)
        tracker.add(tracker_image, (bbox1))
            
        ok, boxes = tracker.update(tracker_image)
    
        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(tracker_image, p1, p2, (200,0,0))
        cv2.imshow("tracker_image", tracker_image)        
        '''
        
        #cv2.imshow("partial_window", tmp_image)        
        #print("New estimated: ", str(np.argmax(Y_estimated)))
        #cv2.waitKey(0)
        
        #
        #   print(np.argmax(Y_estimated))
        
    cv2.imshow("Window", color_image)
    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()