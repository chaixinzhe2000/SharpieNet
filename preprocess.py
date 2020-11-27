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

def scaling(input_image):
    input_image = input_image / 255.0
    return input_image