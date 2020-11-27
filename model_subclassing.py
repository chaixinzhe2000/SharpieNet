import tensorflow as tf
import math
import numpy as np

from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt


# Common
# def my_conv(model, in_channels, out_channels, kernel_size, bias=True):
#     return tf.keras.layers.Conv2D(model.number_of_features, model.kernel_size, strides=(1, 1),
#                                   padding=model.kernel_size // 2, activation="relu",
#                                   kernel_initializer="Orthogonal")

# ResBlock for our model:
# no batch norm
# no activation functions after each conv layer
# no res scaling factor
class ResBlock(tf.keras.layers):
    def __init__(self, model, kernel_size, filters):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="same",
                                            kernel_initializer="Orthogonal")
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="same",
                                            kernel_initializer="Orthogonal")
        self.activation_fxn = tf.keras.activations.relu()
        # self.res_factor = res_factor

    def call(self, model, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.activation_fxn(x)
        x += input_tensor
        return x


# Upsampler for our model
# no batch norm
# no activation function
class Upsampler(tf.keras.layers):
    def __init__(self, model, number_of_features):
        super(Upsampler, self).__init__()
        # ONLY WORKS FOR model.scaling_factor == 3 and MAYBE model.kernel_size == 3
        self.model_scaling_factor = model.scaling_factor
        if model.scaling_factor == 3:
            self.conv1 = tf.keras.layers.Conv2D(number_of_features * (model.scaling_factor ** 2),
                                                model.kernel_size, strides=(1, 1),
                                                padding="same",
                                                kernel_initializer="Orthogonal")
            # CAN I DEFINE A FUNCTION LIKE THIS?
            self.pixel_shuffle = tf.nn.depth_to_space
        else:
            raise NotImplementedError

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pixel_shuffle(x, self.model_scaling_factor)
        return x


# EDSR super class
class EDSR_super:
    def __init__(self, upscale_factor, channels, input_shape):
        # from EDSR (torch)
        self.number_of_resblocks = 32
        self.number_of_features = 256
        self.kernel_size = 3
        # self.res_factor = 0.1
        self.scaling_factor = 3
        self.final_output_channels = 3

        # from keras tutorial
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.input_shape = input_shape

        # model
        inputs = tf.keras.Input(shape=self.input_shape)
        # head
        x = tf.keras.layers.Conv2D(self.number_of_features, self.kernel_size, strides=(1, 1),
                                   padding=self.kernel_size // 2,
                                   kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
        # body
        for i in range(self.number_of_resblocks):
            x = ResBlock(self, self.kernel_size, self.number_of_features).call(model=self, input_tensor=x)
        # tail
        x = Upsampler(model=self, number_of_features=self.number_of_features).call(x)
        x = tf.keras.layers.Conv2D(self.final_output_channels, self.kernel_size, strides=(1, 1),
                                   padding="same", kernel_initializer="Orthogonal")(x)
        self.EDSR_model = tf.keras.Model(inputs, x)
        self.EDSR_model.summary()

    def train(self):
        pass

    def test(self):
        pass

# class EDSR_class(tf.keras.Model):
#     def __init__(self, upscale_factor, channels, input_shape):
#         super(EDSR_class, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.channels = channels
#         self.input_shape = input_shape
#
#         # define inputs and outputs for keras model using functional API
#         inputs = tf.keras.Input(shape=self.input_shape)
#         x = tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding="same", activation="relu",
#                                    kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
#         x = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu",
#                                    kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
#         x = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="same", activation="relu",
#                                    kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
#         x = tf.keras.layers.Conv2D(self.channels * (self.upscale_factor ** 2), 3, padding="same", activation="relu",
#                                    kernel_initializer="Orthogonal", input_shape=self.input_shape)(x)
#         outputs = tf.nn.depth_to_space(x, upscale_factor)
#
#         # initialize model
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
#     def train(self, training_images, training_labels, loss_function, optimizer, number_of_batches, number_of_epochs):
#         # compile model
#         self.model.compile(optimizer=optimizer, loss=loss_function)
#         # fit model
#         batch_size = tf.floor(tf.shape(training_images)[0], number_of_batches)
#         self.model.fit(training_images, training_labels, batch_size, number_of_epochs)
#
#     def test_evaluate_and_output_images(self, test_images, test_labels):
#         predictions = self.model.predict(test_images)
#
#         # spit out images comparing test_images with test_labels
#
#         # evaluate model against psnr or smth like that and return tuple with those measur
