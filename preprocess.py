import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras import layers
import random
from datetime import datetime

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory


def get_normalized_data(image_path, batch_size, dimension):
    train_ds = image_dataset_from_directory(
        image_path,
        batch_size=batch_size,
        image_size=(dimension, dimension),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
    )

    valid_ds = image_dataset_from_directory(
        image_path,
        batch_size=batch_size,
        image_size=(dimension, dimension),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
    )

    return train_ds, valid_ds


def normalize(input_image):
    input_image = input_image / 255.0
    return input_image


# Use TF Ops to process.
def process_input(img, img_sz):
    img = tf.image.rgb_to_yuv(img)
    last_dimension_axis = len(img.shape) - 1
    y, u, v = tf.split(img, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [img_sz, img_sz], method="area")


def process_target(img):
    img = tf.image.rgb_to_yuv(img)
    y, u, v = tf.split(img, 3, axis=len(img.shape) - 1)
    return y