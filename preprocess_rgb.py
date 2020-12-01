import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory


def get_normalized_x_and_y(full_training_data_path, HR_size, LR_size):
    x = []
    y = []
    number_of_images = 0
    for file_name in os.listdir(full_training_data_path):
        if file_name.endswith(".jpg"):
            number_of_images+=1
            image_path = os.path.join(full_training_data_path, file_name)
            y_image_PIL = tf.keras.preprocessing.image.load_img(image_path)
            y_image_array = tf.keras.preprocessing.image.img_to_array(y_image_PIL)
            y_image_array_cropped = y_image_array[0:HR_size, 0:HR_size, :]
            y.append(y_image_array_cropped)
            # y_image_PIL_cropped = tf.keras.preprocessing.image.array_to_img(y_image_array_cropped)
            x_image_PIL = tf.image.resize(y_image_array_cropped, [LR_size, LR_size], method="area")
            x_image_array = tf.keras.preprocessing.image.img_to_array(x_image_PIL)
            x.append(x_image_array)

    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="float32")
    global_rgb_mean = np.mean(y, axis=tuple(range(y.ndim-1)), dtype="float64")
    global_rgb_std = np.std(y, axis=tuple(range(y.ndim-1)), dtype="float64")
    x /= 255.0
    y /= 255.0
    return x, y, global_rgb_mean, global_rgb_std




def further_normalization(train_x, train_y, global_rgb_mean, global_rgb_std):
    train_x = (train_x-global_rgb_mean)/global_rgb_std
    train_y = (train_y-global_rgb_mean)/global_rgb_std
    train_x /= 255.0
    train_y /= 255.0
    return train_x, train_y

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