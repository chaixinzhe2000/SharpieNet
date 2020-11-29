import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory


def get_normalized_x_and_y(full_training_data_path, HR_size, LR_size):
    x = []
    y = []
    for file_name in os.listdir(full_training_data_path):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(full_training_data_path, file_name)
            y_image_PIL = tf.keras.preprocessing.image.load_img(image_path)
            y_image_array = tf.keras.preprocessing.image.img_to_array(y_image_PIL)
            y_image_array_cropped = y_image_array[0:HR_size, 0:HR_size, :]
            y.append(y_image_array_cropped)
            # y_image_PIL_cropped = tf.keras.preprocessing.image.array_to_img(y_image_array_cropped)
            x_image_PIL = tf.image.resize(y_image_array_cropped, [LR_size,LR_size])
            x_image_array = tf.keras.preprocessing.image.img_to_array(x_image_PIL)
            x.append(x_image_array)
    x = np.array(x)/255
    y = np.array(y)/255
    return x, y


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
def shrink_input(img, new_img_size):
    # TODO: check method="area"? maybe bilinear or bicubic instead? (I see bicubic online a lot)
    return tf.image.resize(img, [new_img_size, new_img_size])


def resize_image(image, resulting_size):
    return tf.image.resize(image, [resulting_size, resulting_size], PIL.Image.BICUBIC)