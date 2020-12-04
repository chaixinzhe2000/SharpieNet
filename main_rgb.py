import os

import model_subclassing_rgb
import preprocess_rgb
import postprocess_rgb
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt



def main():
    # defining hyperparameters
    batch_size = 25
    original_size = 300
    upscale_factor = 3
    epochs_for_l1 = 50
    epochs_for_perceptual = 200
    input_size = original_size // upscale_factor
    LR_size = input_size
    HR_size = original_size
    run_trial_id = random.randint(0, 10000)

    # joining relative path to form a full path
    dirname = os.path.dirname(__file__)
    training_full_path = os.path.join(dirname, "div2k_and_BSDS500_dataset/train")

    train_x, train_y, global_rgb_mean, global_rgb_std = preprocess_rgb.get_normalized_x_and_y(training_full_path, HR_size, LR_size)
    # train_x, train_y = preprocess_rgb.further_normalization(train_x, train_y, global_rgb_mean, global_rgb_std)

    print("PREPROCESSING IS DONE")

    # initialize the model
    model = model_subclassing_rgb.EDSR_super(input_size)
    '''
    # train the model using l1 loss
    weights_file = os.path.join(dirname, 'l1_training_weights.h5')
    model.train_l1(train_data, epochs_for_l1, validation_data=validation_data, verbose=2)
    # model.save_weights(weights_file)
    '''
    # TODO: train with l1 loss
    # model.train_l1(train_x=train_x, train_y=train_y, epochs=epochs_for_l1, batch_size=batch_size, run_trial_id=run_trial_id, verbose=2)
    # TODO: train with perceptual loss
    model.train_perceptual(train_x=train_x, train_y=train_y, epochs=epochs_for_perceptual, batch_size=batch_size, run_trial_id=run_trial_id, verbose=2)

    # test the model and output results
    # set up the directory from where we get test images
    test_path = os.path.join(dirname, "div2k_and_BSDS500_dataset/test")

    LR_test_images, HR_test_images, global_rgb_mean, global_rgb_std = preprocess_rgb.get_normalized_x_and_y(test_path, HR_size, LR_size)

    for i in range(len(LR_test_images)):
        # load and preprocess test image
        print(np.shape(LR_test_images[i]))
        input = np.expand_dims(LR_test_images[i], axis=0)
        print(np.shape(input))

        predicted_image = model.predict_l1(input)
        print(np.shape(predicted_image))
        # print(predicted_image)
        postprocess_rgb.save_result(predicted_image[0] * 255, "predicted", str(i), run_trial_id)
        postprocess_rgb.save_result(HR_test_images[i] * 255, "HR", str(i), run_trial_id)
        postprocess_rgb.save_result(LR_test_images[i] * 255, "LR", str(i), run_trial_id)


if __name__ == "__main__":
    main()
