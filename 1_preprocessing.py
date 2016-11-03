# -*- coding: utf-8 -*-
"""
1. Preprocessing images

load images and save preprocessed file
"""
from __future__ import print_function

'''
Load dependencies
'''
import numpy as np
import os
from six.moves import cPickle as pickle
import cv2

'''
define initial variables
'''
#np.random.seed(133) # for reproducible research
force_load_images = True # update images from folders
num_classes = 12 # Number of possible classes
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

''' storage color images '''
dataset_train_folder = '/home/pach0/Documents/DFKI/code/DeepLearning/Hysociatea/data/dataset_numbers_head/train'
dataset_test_folder = '/home/pach0/Documents/DFKI/code/DeepLearning/Hysociatea/data/dataset_numbers_head/eval'
name_mixed_classes_file = 'dataset_numbers_head.pickle'
''''''

'''Auxiliar methods'''
def extract_classes(dataset_folder):
    '''Load name of folders'''
    root = dataset_folder
    
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
        'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
        
    # print(data_folders)
    return data_folders

train_folders = extract_classes(dataset_train_folder)
test_folders = extract_classes(dataset_test_folder)

def load_class(folder):
  '''Load the data for a single body part.'''
  image_files = os.listdir(folder)
  min_num_images = len(image_files) * 0.9 # minimum 90% of total images
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  # print(folder)
  image_index = 0
  for im_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
        color = cv2.imread(image_file)
        if(color != None):      
            image_data = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            '''
            cv2.imshow("image_data", color)
            cv2.waitKey(40);
            '''
            if image_data.shape != (image_size, image_size):
                #resize
                image_data = cv2.resize(image_data, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
                
            dataset[image_index, :, :] = image_data
            image_index += 1
        else:
            print('Could not read:', image_file)
    except e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index + 1
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
  print('tensor shape:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset, num_images

def pickle_classes(data_folders, force=False):
  dataset_names = []
  num_total_images = 0
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
      num_total_images += len(os.listdir( folder.replace(".pickle", "") ))
    else:
      print('Pickling %s.' % set_filename)
      dataset, num_images = load_class(folder)
      num_total_images += num_images
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
      
  return dataset_names, num_total_images

train_datasets, num_train_samples = pickle_classes(train_folders, force_load_images)
test_datasets, num_test_samples = pickle_classes(test_folders, force_load_images)

'''
Next, we'll randomize the data. It's important to have the labels 
well shuffled for the training and test distributions to match.
'''

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

'''
Merge and prune the training data as needed.
The labels will be stored into a separate array of integers 0 through n.
Also create a validation dataset for hyperparameter tuning.
'''
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, total_images, perc_validation_set = 0):
  valid_dataset, valid_labels = make_arrays(int(total_images * perc_validation_set), image_size)
  train_dataset, train_labels = make_arrays(int(total_images * (1-perc_validation_set)), image_size)

  val_index_initial_dataset=0
  val_index_final_dataset=0
  train_index_initial_dataset=0
  train_index_final_dataset=0
  
  for label, pickle_file in enumerate(pickle_files):       
      try:
          with open(pickle_file, 'rb') as f:
            class_set = pickle.load(f)
            # let's shuffle the class_set to have random validation and training set
            np.random.shuffle(class_set)            
            
            total_images_per_class = len(class_set)
            num_images_val = int(total_images_per_class*perc_validation_set)
            num_images_train = int(total_images_per_class*(1-perc_validation_set))
            
            ''' add validation images to dataset'''
            if valid_dataset is not None:
                val_index_final_dataset += num_images_val
                
                valid_letter = class_set[:num_images_val, :, :]
                valid_dataset[val_index_initial_dataset:val_index_final_dataset, :, :] = valid_letter
                valid_labels[val_index_initial_dataset:val_index_final_dataset] = label
                
                val_index_initial_dataset += num_images_val
    
            ''' add train images to dataset'''
            train_index_final_dataset += num_images_train
            
            train_letter = class_set[num_images_val:num_images_val+num_images_train, :, :]
            train_dataset[train_index_initial_dataset:train_index_final_dataset, :, :] = train_letter
            train_labels[train_index_initial_dataset:train_index_final_dataset] = label
            
            train_index_initial_dataset += num_images_train
        
            del class_set
            del train_letter
            print("finished ", pickle_file)
            f.close()
        
      except Exception as e:
          print('Unable to process data from', pickle_file, ':', e)
          raise

  return valid_dataset, valid_labels, train_dataset, train_labels

porc_validation_size = 0.4

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, num_train_samples, porc_validation_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


_, _, test_dataset, test_labels = merge_datasets(test_datasets, num_test_samples, 0)
print('Testing:', test_dataset.shape, test_labels.shape)
test_dataset, test_labels = randomize(test_dataset, test_labels)


'''
Finally, let's save the data for later reuse:
'''
try:
  f = open(name_mixed_classes_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', name_mixed_classes_file, ':', e)
  raise

statinfo = os.stat(name_mixed_classes_file)
print('Compressed pickle size:', statinfo.st_size)


