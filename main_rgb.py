import os

import model_subclassing_rgb
import preprocess_rgb
import postprocess_rgb
import tensorflow as tf
import numpy as np


def main():
    # defining hyperparameters
    batch_size = 8
    original_size = 300
    upscale_factor = 3
    epochs_for_l1 = 1
    epochs_for_perceptual = 1
    input_size = original_size // upscale_factor
    LR_size = input_size
    HR_size = original_size

    # joining relative path to form a full path
    dirname = os.path.dirname(__file__)
    training_full_path = os.path.join(dirname, "BSDS500/data/small_training_set/small_training_set_child")
    train_x, train_y = preprocess_rgb.get_normalized_x_and_y(training_full_path, HR_size, LR_size)
    print("PREPROCESSING IS DONE")

    # initialize the model
    model = model_subclassing_rgb.EDSR_super(input_size)
    '''
    # train the model using l1 loss
    weights_file = os.path.join(dirname, 'l1_training_weights.h5')
    model.train_l1(train_data, epochs_for_l1, validation_data=validation_data, verbose=2)
    # model.save_weights(weights_file)
    '''
    # train the model using perceptual loss
    # TODO: implement perceptual loss
    model.train_perceptual(train_x=train_x, train_y=train_y, epochs=epochs_for_perceptual, verbose=2)

    # test the model and output results
    # set up the directory from where we get test images
    test_path = os.path.join(dirname, "BSDS500/data/test")
    # make a list of test image paths
    list_of_test_img_paths = []
    for file_name in os.listdir(test_path):
        if file_name.endswith(".jpg"):
            list_of_test_img_paths.append(file_name)

    for file_name in list_of_test_img_paths:
        # load and preprocess test image
        test_image_path = os.path.join(test_path, file_name)
        HR_test_image = tf.keras.preprocessing.image.load_img(test_image_path)
        # TODO: check why we don't use shrink_input here??? (used to be resize_image)
        LR_test_image = preprocess_rgb.shrink_input(HR_test_image, LR_size)
        # TODO: upscale_image below
        # input_y, input_Cb, input_Cr, = postprocess_rgb.rgb_to_yuv_normalized(HR_test_image)
        # input = np.expand_dims(input_y, axis=0)
        # call model.predict and process yuv image to rgb
        '''
        yuv_predicted_image = model.predict_l1(input)
        '''
        predicted_image = model.predict_perceptual(LR_test_image)

        # rgb_predicted_image = postprocess_rgb.yuv_to_rgb(yuv_predicted_image, input_Cb, input_Cr)
        # TODO: instead of using matplotlib, just output a png or jpeg from rgb_predicted_image
        # TODO: for now, we use matplotlib to see if ours works.
        file_name = os.path.splitext(file_name)[0]
        postprocess_rgb.save_result(predicted_image, "predicted", str(file_name))
        postprocess_rgb.save_result(HR_test_image, "HR", str(file_name))
        LR_test_image_for_plot = preprocess_rgb.resize_image(HR_test_image, HR_size)
        postprocess_rgb.save_result(LR_test_image_for_plot, "LR", str(file_name))



if __name__ == "__main__":
    main()
