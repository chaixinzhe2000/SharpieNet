import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt


class EDSR_class(tf.keras.Model):
    def __init__(self, upscale_factor, channels, input_shape):
        super(EDSR_class, self).__init__()
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.input_shape = input_shape

        # define inputs and outputs for keras model using functional API
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding="same", activation="relu", kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
        x = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
        x = tf.keras.layers.Conv2D(self.channels*(self.upscale_factor**2), 3, padding="same", activation="relu", kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
        outputs = tf.nn.depth_to_space(x, upscale_factor)

        # initialize model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def train(self, training_images, training_labels, loss_function, optimizer, number_of_batches, number_of_epochs):
        # compile model
        self.model.compile(optimizer=optimizer, loss=loss_function)
        # fit model
        batch_size = tf.floor(tf.shape(training_images)[0], number_of_batches)
        self.model.fit(training_images, training_labels, batch_size, number_of_epochs)

    def test_evaluate_and_output_images(self, test_images, test_labels):
        predictions = self.model.predict(test_images)

        # spit out images comparing test_images with test_labels


        # evaluate model against psnr or smth like that and return tuple with those measurements