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
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, model, kernel_size, filters):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="SAME",
                                            kernel_initializer="Orthogonal")
        self.activation_fxn = tf.keras.activations.relu
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="SAME",
                                            kernel_initializer="Orthogonal")

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
    def __init__(self, model, number_of_features):
        super(Upsampler, self).__init__()
        # ONLY WORKS FOR model.scaling_factor == 3 and MAYBE model.kernel_size == 3
        self.model_scaling_factor = model.scaling_factor
        if model.scaling_factor == 3:
            self.conv1 = tf.keras.layers.Conv2D(number_of_features * (model.scaling_factor ** 2),
                                                model.kernel_size, strides=(1, 1),
                                                padding="SAME",
                                                kernel_initializer="Orthogonal")
            # TODO: CAN I DEFINE A FUNCTION LIKE THIS?
            self.pixel_shuffle = tf.nn.depth_to_space
        else:
            raise NotImplementedError

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        # TODO: check if this is the right "block_size" does it want 3 or 9
        x = self.pixel_shuffle(x, self.model_scaling_factor)
        return x


# EDSR super class
class EDSR_super:
    def __init__(self, image_size):
        # from EDSR (torch)
        # self.number_of_resblocks = 32
        # self.number_of_features = 256
        self.number_of_resblocks = 16
        self.number_of_features = 128
        self.kernel_size = 3
        self.res_scaling = 0.1
        self.scaling_factor = 3
        self.final_output_channels = 1

        # from keras tutorial
        # self.upscale_factor = upscale_factor
        # self.channels = channels
        self.input_shape = (None, None, 1)

        # model
        inputs = tf.keras.Input(shape=self.input_shape)
        # subtract mean (mean shift)
        # TODO: not implemented meanshift yet
        # head
        x = tf.keras.layers.Conv2D(self.number_of_features, self.kernel_size, strides=(1, 1),
                                   padding="SAME",
                                   kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
        # body
        # TODO: Not sure if we can for loop like this
        for i in range(self.number_of_resblocks):
            x = ResBlock(self, self.kernel_size, self.number_of_features).call(model=self, input_tensor=x)
        # tail
        x = Upsampler(model=self, number_of_features=self.number_of_features).call(x)
        x = tf.keras.layers.Conv2D(self.final_output_channels, self.kernel_size, strides=(1, 1),
                                   padding="SAME", kernel_initializer="Orthogonal")(x)
        # add mean (mean shift)
        # TODO: not implemented meanshift yet
        # initialize model
        self.EDSR_model_l1 = tf.keras.Model(inputs, x)
        # self.EDSR_model_l1.summary()


        # TODO: MAYBE TRY CAFFE FOR PERCEPTUAL LOSS BECAUSE THAT IS APPARENTLY BETTER
        # make 3 copies of EDSR_model_l1's output
        triple_output = tf.keras.layers.Concatenate()([self.EDSR_model_l1.output, self.EDSR_model_l1.output,
                                                       self.EDSR_model_l1.output])
        # set up ESDR model using vgg16 perceptual loss
        self.perceptual_loss_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                                                 input_tensor=None)
                                                                 # input_shape=(tf.shape(self.EDSR_model_l1.output)[0],
                                                                 #              tf.shape(self.EDSR_model_l1.output)[1],
                                                                 #              3))
        selected_layers = [1, 3, 6, 11, 13, 17]
        selected_outputs = [self.perceptual_loss_model.layers[i].output for i in selected_layers]
        # TODO: change line above into the for loop below
        # for layer_index in selected_layers:
        #     selected_outputs.append(self.perceptual_loss_model[layer_index].output)
        self.perceptual_loss_model = tf.keras.Model(self.perceptual_loss_model.inputs, selected_outputs)
        self.perceptual_loss_model.trainable = False
        loss_model_outputs = self.perceptual_loss_model(triple_output)
        # initialize fully connected model
        self.EDSR_full_model = tf.keras.Model(self.EDSR_model_l1.input, loss_model_outputs)
        self.EDSR_full_model.summary()
        '''
        # # if the line above doesn't work due to a type problem, make a list with lossModelOutputs:
        # lossModelOutputs = [lossModelOutputs[i] for i in range(len(selectedLayers))]
        '''



    def train_l1(self, training_data, epochs, validation_data, verbose=2):
        self.optimizer_l1 = tf.keras.optimizers.Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
        self.loss_fxn_l1 = tf.keras.losses.MeanSquaredError()
        self.EDSR_model_l1.compile(optimizer=self.optimizer_l1, loss=self.loss_fxn_l1)
        history = self.EDSR_model_l1.fit(training_data, epochs=epochs, validation_data=validation_data, verbose=verbose)
        print('FINISHED TRAINING USING L1 LOSS')

    def train_perceptual(self, training_data, epochs, validation_data, verbose=2):
        training_data_list = list(training_data)
        print(np.shape(training_data_list))
        triple_training_data_list = np.concatenate((training_data_list, training_data_list, training_data_list), axis=-1)
        triple_training_data_list = np.asarray(triple_training_data_list).astype('float32')
        triple_training_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(triple_training_data_list))
        loss_model_labels = self.perceptual_loss_model.predict(triple_training_data)
        for layer in self.perceptual_loss_model.layers[:]:
            layer.trainable = False
        self.learning_rate_perceptual = PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
        self.optimizer_perceptual = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_perceptual)
        self.EDSR_full_model.compile(optimizer=self.optimizer_perceptual, loss='mse')
        print('FINISHED COMPILING FULL MODEL \n STARTING TO TRAIN NOW')
        self.EDSR_full_model.fit(training_data, loss_model_labels, epochs=epochs, validation_data=validation_data, verbose=verbose)
        print('FINISHED TRAINING USING PERCEPTUAL LOSS')

    def test(self):
        pass

    def predict_l1(self, test_image):
        return self.EDSR_model_l1.predict(test_image)

    def predict_perceptual(self, test_image):
        return self.EDSR_full_model.predict(test_image)
