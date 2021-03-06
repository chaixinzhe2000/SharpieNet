import tensorflow as tf
import numpy as np

# # MeanShift layer for our model
# class MeanShift(tf.keras.layers):
#     def __init__(self):
#         pass

# ResBlock for our model:
# no batch norm
# no activation functions after each conv layer
# no res scaling factor
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay
import os


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, model, kernel_size, filters, initializer):
        super(ResBlock, self).__init__()
        self.initializer = initializer
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="SAME",
                                            kernel_initializer=self.initializer)
        self.activation_fxn = tf.keras.activations.relu
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="SAME",
                                            kernel_initializer=self.initializer)

        # self.res_factor = res_factor

    def call(self, model, input_tensor):
        x = self.conv1(input_tensor)
        x = self.activation_fxn(x)
        x = self.conv2(x)
        x = tf.multiply(x, model.res_scaling)
        x += input_tensor
        return x


# Upsampler for our model
# no batch norm
# no activation function
class Upsampler(tf.keras.layers.Layer):
    def __init__(self, model, number_of_features, initializer):
        super(Upsampler, self).__init__()
        # ONLY WORKS FOR model.scaling_factor == 3 and MAYBE model.kernel_size == 3
        self.model_scaling_factor = model.scaling_factor
        self.initializer = initializer
        if model.scaling_factor == 3:
            self.conv1 = tf.keras.layers.Conv2D(number_of_features * (model.scaling_factor ** 2),
                                                model.kernel_size, strides=(1, 1),
                                                padding="SAME",
                                                kernel_initializer=self.initializer)
            # TODO: CAN I DEFINE A FUNCTION LIKE THIS?
            self.pixel_shuffle = tf.nn.depth_to_space
        else:
            raise NotImplementedError

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        # TODO: check if this is the right "block_size" does it want 3 or 9
        x = self.pixel_shuffle(x, self.model_scaling_factor)
        return x


class vgg16_preprocessing_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(vgg16_preprocessing_layer, self).__init__()
        # ONLY WORKS FOR model.scaling_factor == 3 and MAYBE model.kernel_size == 3

    def call(self, input_tensor):
        input_tensor *= 255.0
        return tf.keras.applications.vgg16.preprocess_input(input_tensor)


# EDSR super class
class EDSR_super:
    def __init__(self, image_size):
        self.number_of_resblocks = 2
        self.number_of_features = 2
        self.vgg_out_layer = 2

        self.kernel_size = 3
        self.res_scaling = 0.2
        self.scaling_factor = 3
        self.final_output_channels = 3

        self.input_shape = (100, 100, 3)

        # model
        inputs = tf.keras.Input(shape=self.input_shape)
        # head
        x = tf.keras.layers.Conv2D(self.number_of_features, self.kernel_size, strides=(1, 1),
                                   padding="SAME",
                                   kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
        # body
        for i in range(self.number_of_resblocks):
            x = ResBlock(self, self.kernel_size, self.number_of_features, "Orthogonal").call(model=self, input_tensor=x)
        # tail
        x = Upsampler(model=self, number_of_features=self.number_of_features,
                      initializer=tf.keras.initializers.GlorotUniform()).call(x)
        x = tf.keras.layers.Conv2D(self.final_output_channels, self.kernel_size, strides=(1, 1),
                                   padding="SAME", kernel_initializer="Orthogonal")(x)

        self.EDSR_model_l1 = tf.keras.Model(inputs, x, name="MAIN_MODEL")
        self.EDSR_model_l1.summary()

        # set up ESDR model using vgg16 perceptual loss
        self.perceptual_loss_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet",
                                                                       input_shape=(300, 300, 3))
        self.perceptual_loss_model.trainable = False
        for layer in self.perceptual_loss_model.layers:
            layer.trainable = False

        selected_outputs = self.perceptual_loss_model.layers[self.vgg_out_layer].output

        self.perceptual_loss_model = tf.keras.Model(self.perceptual_loss_model.input, selected_outputs,
                                                    name="VGG16_partial_for_loss")
        self.perceptual_loss_model.summary()

        loss_model_outputs = self.perceptual_loss_model(vgg16_preprocessing_layer().call(self.EDSR_model_l1.output))
        # initialize fully connected model
        self.EDSR_full_model = tf.keras.Model(self.EDSR_model_l1.input, loss_model_outputs, name="FULL_MODEL")

        '''
        # # if the line above doesn't work due to a type problem, make a list with lossModelOutputs:
        # lossModelOutputs = [lossModelOutputs[i] for i in range(len(selectedLayers))]
        '''

    def train_l1(self, train_x, train_y, epochs, batch_size, run_trial_id, verbose=2):
        self.optimizer_l1 = tf.keras.optimizers.Adam(
            learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[5e-4, 5e-5]))
        self.loss_fxn_l1 = tf.keras.losses.MeanSquaredError()
        self.EDSR_model_l1.compile(optimizer=self.optimizer_l1, loss=self.loss_fxn_l1)
        history = self.EDSR_model_l1.fit(x=train_x, y=train_y, epochs=epochs, verbose=verbose, shuffle=True)
        print('FINISHED TRAINING USING L1 LOSS')
        filepath = "saved_models/TRIAL" + str(run_trial_id) + "-RB_" + str(self.number_of_resblocks) + "-FEATS_" + str(
            self.number_of_features) + "-VGGOUT_" + str(self.vgg_out_layer) + "-BSZ_" + str(
            batch_size) + "FINAL_MODEL_ONLY_L1_TRAINING.hdf5"
        self.EDSR_full_model.save(filepath=filepath)

    def train_perceptual(self, train_x, train_y, epochs, batch_size, run_trial_id, verbose=2):
        self.learning_rate_perceptual = PiecewiseConstantDecay(boundaries=[100000], values=[5e-4, 5e-5])
        self.optimizer_full = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_perceptual)

        train_y *= 255.0
        train_y = tf.keras.applications.vgg16.preprocess_input(train_y)
        Y_train_feature_sets = self.perceptual_loss_model.predict(train_y)

        self.EDSR_full_model.summary()
        self.EDSR_full_model.compile(optimizer=self.optimizer_full, loss='mse', metrics=['mse'])
        print('FINISHED COMPILING FULL MODEL \n STARTING TO TRAIN NOW')

        filepath = "saved_models/TRIAL" + str(run_trial_id) + "-RB_" + str(self.number_of_resblocks) + "-FEATS_" + str(
            self.number_of_features) + "-VGGOUT_" + str(self.vgg_out_layer) + "-BSZ_" + str(
            batch_size) + "-EPOCH_{epoch:02d}-LOSS_{loss:.1f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto',
                                     save_freq=int(np.shape(train_x)[0] / batch_size) * 10)

        self.EDSR_full_model.fit(x=train_x, y=Y_train_feature_sets, batch_size=batch_size, epochs=epochs,
                                 verbose=verbose, callbacks=[checkpoint], shuffle=True)

        print('FINISHED TRAINING USING PERCEPTUAL LOSS')

    def test_perceptual_loss(self, image_x_np_array, image_y_np_array):
        image_x = self.perceptual_loss_model.predict(image_x_np_array)
        image_y = self.perceptual_loss_model.predict(image_y_np_array)
        print("MEAN SQUARED ERROR PERCEPTUAL LOSS: ", np.mean((image_y - image_x) ** 2))

    def predict_l1(self, test_image):
        return self.EDSR_model_l1.predict(test_image)

    def predict_perceptual(self, test_image):
        return self.EDSR_full_model.predict(test_image)
