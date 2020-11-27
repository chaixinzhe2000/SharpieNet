import datetime
import os
import random

import numpy as np
import tensorflow as tf
from IPython.core.display import display
from keras_preprocessing.image import array_to_img

import preprocess


# hyperparameters
batch_size = 8
original_size = 500
upscale_factor = 5
input_size = original_size // upscale_factor

# joining relative path to form a full path
dirname = os.path.dirname(__file__)
image_path = os.path.join(dirname, "BSDS500/data/images")

train_data, test_data = \
    preprocess.get_normalized_data(image_path, batch_size, original_size)

# Scale from (0, 255) to (0, 1)
train_ds = train_data.map(preprocess.scaling)
valid_ds = test_data.map(preprocess.scaling)

for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))


### TODO: TOOOOOO CHANGGEGEEGEE

# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

"""
Let's take a look at the input and target data.
"""

for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))