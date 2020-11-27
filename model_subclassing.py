import tensorflow as tf

# MeanShift layer for our model
class MeanShift(tf.keras.layers):


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
        self.activation_fxn = tf.keras.activations.relu()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                                            padding="same",
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
    def __init__(self, upscale_factor, channels, input_shape):
        # from EDSR (torch)
        self.number_of_resblocks = 32
        self.number_of_features = 256
        self.kernel_size = 3
        self.res_scaling = 0.1
        self.scaling_factor = 3
        self.final_output_channels = 3

        # from keras tutorial
        # self.upscale_factor = upscale_factor
        # self.channels = channels
        self.input_shape = input_shape

        # model
        inputs = tf.keras.Input(shape=self.input_shape)
        # subtract mean (mean shift)

        # head
        x = tf.keras.layers.Conv2D(self.number_of_features, self.kernel_size, strides=(1, 1),
                                   padding=self.kernel_size // 2,
                                   kernel_initializer="Orthogonal", input_shape=self.input_shape)(inputs)
        # body
        # TODO: Not sure if we can for loop like this
        for i in range(self.number_of_resblocks):
            x = ResBlock(self, self.kernel_size, self.number_of_features).call(model=self, input_tensor=x)
        # tail
        x = Upsampler(model=self, number_of_features=self.number_of_features).call(x)
        x = tf.keras.layers.Conv2D(self.final_output_channels, self.kernel_size, strides=(1, 1),
                                   padding="same", kernel_initializer="Orthogonal")(x)
        # add mean (mean shift)

        # initialize model
        self.EDSR_model = tf.keras.Model(inputs, x)
        self.EDSR_model.summary()

    def train(self, training_data, epochs, loss_fxn, optimizer, validation_data=None, verbose=2):
        self.EDSR_model.compile(optimizer=optimizer, loss=loss_fxn)
        history = self.EDSR_model.fit(training_data, epochs=epochs, validation_data=validation_data, verbose=verbose)
        return history

    def test(self):
        pass