import os
import preprocess
import postprocess
import model_subclassing
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
    image_path = os.path.join(dirname, "BSDS500/data/small_training_set")

    train_data, validation_data = \
        preprocess.get_normalized_data(image_path, batch_size, original_size)

    # Scale from (0, 255) to (0, 1)
    train_data = train_data.map(preprocess.normalize)
    validation_data = validation_data.map(preprocess.normalize)

    train_data = train_data.map(
        lambda x: (preprocess.process_input(x, input_size), preprocess.process_target(x))
    )

    # prefetch is for computation optimization
    train_data = train_data.prefetch(buffer_size=32)

    validation_data = validation_data.map(
        lambda x: (preprocess.process_input(x, input_size), preprocess.process_target(x))
    )

    # prefetch is for computation optimization
    validation_data = validation_data.prefetch(buffer_size=32)

    print("PREPROCESSING IS DONE")

    # initialize the model
    model = model_subclassing.EDSR_super(input_size)
    '''
    # train the model using l1 loss
    weights_file = os.path.join(dirname, 'l1_training_weights.h5')
    model.train_l1(train_data, epochs_for_l1, validation_data=validation_data, verbose=2)
    # model.save_weights(weights_file)
    '''
    # train the model using perceptual loss
    # TODO: implement perceptual loss
    model.train_perceptual(training_data=train_data, epochs=epochs_for_perceptual, validation_data=validation_data,verbose=2)

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
        LR_test_image = preprocess.resize_image(HR_test_image, LR_size)
        # TODO: upscale_image below
        input_y, input_Cb, input_Cr, = postprocess.rgb_to_yuv_normalized(HR_test_image)
        input = np.expand_dims(input_y, axis=0)
        # call model.predict and process yuv image to rgb
        '''
        yuv_predicted_image = model.predict_l1(input)
        '''
        yuv_predicted_image = model.predict_perceptual(input)

        rgb_predicted_image = postprocess.yuv_to_rgb(yuv_predicted_image, input_Cb, input_Cr)
        # TODO: instead of using matplotlib, just output a png or jpeg from rgb_predicted_image
        # TODO: for now, we use matplotlib to see if ours works.
        file_name = os.path.splitext(file_name)[0]
        postprocess.save_result(rgb_predicted_image, "predicted", str(file_name))
        postprocess.save_result(HR_test_image, "HR", str(file_name))
        LR_test_image_for_plot = preprocess.resize_image(HR_test_image, HR_size)
        postprocess.save_result(LR_test_image_for_plot, "LR", str(file_name))



if __name__ == "__main__":
    main()
