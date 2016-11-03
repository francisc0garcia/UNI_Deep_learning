# -*- coding: utf-8 -*-
"""
Learning using torch
"""
from __future__ import print_function, division
#import sys
#import os
#from docopt import docopt
import PyTorch
import PyTorchHelpers
import numpy as np
#from mnist import MNIST
import time

from six.moves import cPickle as pickle

'''
define initial variables
'''

# backend [cpu|cuda|cl]
backend = 'cpu'
batchSize = 300
numEpochs = 10
batchesPerEpoch = 500
learningRate = 0.025

#np.random.seed(133) # for reproducible research
pickle_file = 'dataset_numbers_head.pickle'
name_trained_model='torch_model_dataset_numbers_head'
lua_model = 'torch_model_dataset_numbers.lua'
path_file_trained_model='/home/pach0/Documents/DFKI/code/DeepLearning/Hysociatea/models/'
num_classes = 11 # Number of possible classes
image_size = 28  # Pixel width and height.

force_train_model = 1

class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

def reformat(dataset, labels):
      dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
      labels = labels.astype(np.float32)
      return dataset, labels
      
def load_data():
    ''' Open dataset file '''
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      #valid_dataset = save['valid_dataset']
      #valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      
    X, Y = reformat(train_dataset, train_labels)
    # valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    testX, testY = reformat(test_dataset, test_labels)    
    
    return X, Y, testX, testY
  
# Start timer
timer_local = Timer()

X, Y, testX, testY = load_data()

'''
Normalization
'''
mean = X.mean()
std = X.std()

X -= mean
X /= std

testX -= mean
testX /= std
''''''

images = np.array(X, dtype=np.float32) 
labels = np.array(Y, dtype=np.uint8)

del X
del Y

time_hhmmss = timer_local.get_time_hhmmss()
print("Time Load data: %s" % time_hhmmss )

''' load LUA model '''
TorchModel = PyTorchHelpers.load_lua_class(lua_model, 'TorchModel')
torchModel = TorchModel(backend, image_size, num_classes)

labels += 1  # since torch/lua labels are 1-based

# if labels are not correct, just ignored
labels[labels > 10] = 9

N = labels.shape[0]

''' Train model '''
if force_train_model == 1:
    for epoch in range(numEpochs):
        epochLoss = 0
        epochNumRight = 0
        
        for b in range(batchesPerEpoch):
            indexes = np.random.randint(N, size=(batchSize))
            res = torchModel.trainBatch(learningRate, 
                                        images[indexes], 
                                        labels[indexes])

            epochNumRight += res['numRight']
            epochLoss +=  res['loss']
            if b % 100 == 0:
                print('epoch ' + str(epoch) + ' batch ' + str(b) + ' accuracy: ' + str(res['numRight'] * 100.0 / batchSize) + '%')
        
        ''' Save model '''        
        #torchModel.saveModel(path_file_trained_model + name_trained_model)
        torchModel.saveModel(path_file_trained_model + name_trained_model + '.ascii')
        
        learningRate /= 1.1
        print('dropping learning rate to %s' % learningRate)
        
        
            
    time_hhmmss = timer_local.get_time_hhmmss()
    print("Time training: %s" % time_hhmmss )


''' Load model '''
if force_train_model == 0:
    torchModel.loadModel(path_file_trained_model + name_trained_model)

''' Use Test dataset: verify model '''
images = np.array(testX, dtype=np.float32) 
labels = np.array(testY, dtype=np.uint8)

labels += 1  # since torch/lua labels are 1-based
# if labels are not correct, just ignored
labels[labels > 10] = 9

N = labels.shape[0]

numBatches = N // batchSize
epochLoss = 0
epochNumRight = 0
for b in range(numBatches):
    res= torchModel.predict(images[b * batchSize:(b+1) * batchSize])
    predictions = res['prediction'].asNumpyTensor().reshape(batchSize)    
    labelsBatch = labels[b * batchSize:(b+1) * batchSize]
    numRight = (predictions == labelsBatch).sum()
    epochNumRight += numRight

print('test results: accuracy: ' + str(epochNumRight * 100.0 / N) + '%')

time_hhmmss = timer_local.get_time_hhmmss()
print("Time prediction: %s" % time_hhmmss )

del torchModel  # hint to help gc free up memory
