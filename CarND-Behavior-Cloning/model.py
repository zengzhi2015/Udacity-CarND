# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:21:50 2017

@author: Zhi Zeng
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.layers.core import Flatten
from keras.layers.core import Dropout

if __name__ == '__main__':
    #%% Read the driving_log
    base_folder = 'F:\\CarND\\CarND-Behavior-Cloning\\'
    driving_log_path = base_folder+'driving_log.csv'
    table = pd.read_csv(driving_log_path)
    #%% Reformat the name of the image file
    image_name_list = table.center.copy()
    for i in range(table.shape[0]):
        image_name_list[i] = table.center[i].split('/')[-1]
        image_name_list[i] = image_name_list[i].split('\\')[-1]
    angle_list = table.steering.values
    #%% Get the shape of the image
    image_path = base_folder+'IMG\\'+image_name_list[0]
    image = Image.open(image_path)
    image_array = np.asarray(image)
    image_shape = image_array.shape
    print('The size of the dataset is: {}'.format(len(image_name_list)))
    print('The size of the image is: {}'.format(image_shape))
    print('The type of the image is: {}'.format(image_array.dtype))
    #%% Define the generator
    def generator(image_name_list,angle_list,batch_size=32):
        #batch_size = int(batch_size/2)
        num_samples = len(image_name_list)
        while 1: # Used as a reference pointer so code always loops back around
            image_name_list,angle_list = shuffle(image_name_list,angle_list)
            for offset in range(0, num_samples, batch_size):
                batch_image_names = image_name_list[offset:offset+batch_size]
                batch_images = []
                for image_name in batch_image_names:
                    image_path = base_folder+'IMG\\'+image_name
                    image = Image.open(image_path)
                    image_array = np.asarray(image)
                    batch_images.extend([image_array])
                batch_angles = angle_list[offset:offset+batch_size]
                # preprocess the data
                X_train = (np.float32(batch_images)/255.0-0.5)*2.0
                y_train = np.array(batch_angles)
                yield shuffle(X_train, y_train)
    
    image_name_list,angle_list = shuffle(image_name_list,angle_list)
    X_train_files, X_valid_files, y_train, y_valid = train_test_split(image_name_list, angle_list,
                                                        test_size=0.05)
    train_generator = generator(X_train_files, y_train, batch_size=64)
    validation_generator = generator(X_valid_files, y_valid, batch_size=64)
    #%% Build VGG16 based feature extractor over a custom input tensor
    # Fix error with TF and Keras
    tf.python.control_flow_ops = tf
    
    extractor = VGG16(input_shape=(160,320,3),weights='imagenet', include_top=False)
    extractor.summary()
    #%% Build the model
    features = extractor.output
    flat_feature = Flatten()(features)
    #drop_feature = Dropout(0.5)(flat_feature)
    dense1 = Dense(2048,activation='relu')(flat_feature)
    dense2 = Dense(512,activation='linear')(dense1)
    predictions = Dense(1,activation='linear')(dense2)
    
    model = Model(input=extractor.input, output=predictions)
    model.summary()
    #%% Prepering for training
    # Train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers
    for layer in extractor.layers:
        layer.trainable = False
    
    # Compile and train the model here.
    model.compile(optimizer='adam', loss='mean_absolute_error')
    #%% train the model
    model.fit_generator(train_generator,
                        samples_per_epoch = len(X_train_files), 
                        validation_data=validation_generator, 
                        nb_val_samples=len(X_valid_files), 
                        nb_epoch=3)
    #%% Save the model
    model.save('model.h5')