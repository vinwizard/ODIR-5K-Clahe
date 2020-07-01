import os
import random
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
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
training_set = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical') #eith binary or categorial

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'Dataset/Testing',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

cnn = tf.keras.models.Sequential()
#adding first layer
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size = 3,activation = 'relu', input_shape =[64,64,3]))

#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#new layer and pooling of it
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size = 3,activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))



#Flattening
cnn.add(tf.keras.layers.Flatten())
#now this is similar to any dataset as worked on beofre, so apply normal neural network to it
#fully connected layer
cnn.add(tf.keras.layers.Dense(units = 128, activation= 'relu'))
#o/p layer
cnn.add(tf.keras.layers.Dense(units = 17, activation= 'sigmoid'))

#compile
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

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
    
    
history = cnn.fit_generator(training_set,
                  steps_per_epoch = int(6588/32),
                  epochs = 32,
                  validation_data = test_set,
                  validation_steps = int(1377/32))

show_final_history(history)
print(max(history.history["val_accuracy"]))
print(len(history.history["val_accuracy"]))
print(history.history["confusion matrix"])


from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size =(64,64) ) #converts to PIL img
#now to convert it to array 
test_image = image.img_to_array(test_image)
#taking care of batches(maybe bitches too lol)
test_image = np.expand_dims(test_image, axis = 0)#axis =0 implies dimension to hich it is added is first
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    print('Dog')
else:
    print('Cat')    #first parenthesis shows batch dimension, and seocnd shows the result in that