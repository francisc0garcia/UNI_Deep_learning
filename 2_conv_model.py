# -*- coding: utf-8 -*-
"""
2. create a convolutional neural network for training model
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
from six.moves import cPickle as pickle

'''
define initial variables
'''
#np.random.seed(133) # for reproducible research
pickle_file = 'color_mixed_classes.pickle'
name_trained_model='color_mixed_classes.tfl'
path_file_trained_model='/home/pacho-i7/Documents/DFKI/code/DeepLearning/Hysociatea/models/'
num_classes = 12 # Number of possible classes
image_size = 28  # Pixel width and height.


''' Open dataset file '''
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

'''
Reformat into a shape that's more adapted to the models we're going to train:
- data as a flat matrix
- labels as float 1-hot encodings

'''
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

X = train_dataset
Y = train_labels

testX = test_dataset
testY = test_labels

X = X.reshape([-1, image_size, image_size, 1])
testX = testX.reshape([-1, image_size, image_size, 1])

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

''' Personalized convolutional network '''
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

''' Load precomputed model 
model.load(path_file_trained_model + name_trained_model)
'''

''' Train using classifier model MNIST 
model.fit({'input': X}, {'target': Y}, n_epoch=2,
          validation_set=({'input': testX}, {'target': testY}),
          snapshot_step=500,  show_metric=True, run_id='convnet_test')
'''

''' Train using classifier model CIFAR '''
model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(testX, testY),  
         # snapshot_step=200, snapshot_epoch=False,  
          show_metric=True, batch_size=10, run_id='cifar10_cnn')
''' '''         

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


# Manually save model
model.save(path_file_trained_model + name_trained_model)

def test_accuracy(total_samples):
    total_incorrect = 0
    
    for i in range(total_samples):
        index=np.random.randint( len(train_dataset), size=1)
        
        batch_data = train_dataset[index,:].reshape([-1, image_size, image_size, 1])
        batch_labels = train_labels[index,:] 
                
        result = model.predict(batch_data)
        if(np.argmax(result) != np.argmax(batch_labels)):
            #print("predicted: ", np.argmax(result), "Original: ", np.argmax(batch_labels))
            total_incorrect += 1
    
    print("Total incorrect", total_incorrect, " percentage: ", float(total_incorrect) / total_samples) 
#batch_labels

test_accuracy(3000)