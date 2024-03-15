#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import numpy as np
import PIL

model = tf.keras.models.load_model('./.settings/DerekModel.keras')

data_dir = '/Users/derkio6/Code/ML-project/data/archive/test_set/test_set/dogs/dog.4001.jpg'

img_height = img_width = 255

normalized_img = tf.keras.utils.load_img(
    data_dir,
    color_mode = 'grayscale',
    target_size = (img_height, img_width),
    interpolation='bilinear'
    )

normalized_img.shape()


'''
img_height = img_width = 255

img = tf.keras.utils.image_dataset_from_directory(
   data_dir,
   seed = 636,
   image_size = (img_height, img_width),
   batch_size = 32
   )

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = img.class_names

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

'''
