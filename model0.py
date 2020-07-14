import numpy as np 
import pandas as pd

import os

import random
import shutil
import pathlib

path = "Dataset" 
path = pathlib.Path(path)
diseases = list(path.glob('*/*/*.jpg'))
print(len(diseases))
#using shutil.copy, os.mkdir, you need to create separate train_dir and test_dir folders
test_dir = "Dataset/Testing"
train_dir = "Dataset/Training"
#in case somebody needs the code for this, please raise an issue or start a new pull request.

import os
import random
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.mkdir("Dataset/Working")
os.mkdir("Dataset/Working/IV3")

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
xception_weights = 'Dataset/xception_weights_tf_dim_ordering_tf_kernels.h5'
from tensorflow.keras.applications.inception_v3 import InceptionV3
#import inception_v4 as INCV4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_gen = ImageDataGenerator(
        rescale = 1./255,
        featurewise_center = False,  
        samplewise_center = False, 
        featurewise_std_normalization = False,  
        samplewise_std_normalization = False,  
        zca_whitening = False,  
        rotation_range = 10,  
        zoom_range = 0.1, 
        width_shift_range = 0.2,  
        height_shift_range = 0.2, 
        horizontal_flip = True,  
        vertical_flip = False) 

test_gen = ImageDataGenerator(
        rescale = 1./255)

test_generator = train_gen.flow_from_directory(
        test_dir,
        target_size = (224,224),
        batch_size = 32,
        )
train_generator = test_gen.flow_from_directory(
        train_dir,
        target_size = (224,224),
        batch_size = 32)

train_steps = int(4359/32)#change karna haiiiiii
test_steps = int(684/32)

#base_model = InceptionV3(weights = 'imagenet', include_top = False) #pretained CNN
base_model = Xception(weights=xception_weights, include_top = True,input_shape = [224,224,3])
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,(3,3), input_shape = [224,224,2], activation = 'relu')(x)
x = Dense(512,(3,3), input_shape = [224,224,2], activation = 'relu')(x)
predictions = Dense(2, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

os.mkdir("Dataset/Working/IV3/new_run.h5")
filepath = "Dataset/Working/IV3/new_run.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)

def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()

print(train_steps)
print(test_steps)

history = model.fit_generator(train_generator, 
                    steps_per_epoch = train_steps,
                    validation_data = test_generator,
                    validation_steps = test_steps,
                    epochs = 32,
                    verbose = 1,
                    callbacks = [checkpoint, reduce_lr, early_stop])

#print(max(history.history[val_accuracy]))

show_final_history(history)
