import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

def get_datasets(image_path, batch_size, dimension):
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
def shrink_input(img, new_img_size):
    # TODO: check method="area"? maybe bilinear or bicubic instead? (I see bicubic online a lot)
    return tf.image.resize(img, [new_img_size, new_img_size], method="area")


def resize_image(image, resulting_size):
    return tf.image.resize(image, [resulting_size, resulting_size], PIL.Image.BICUBIC)