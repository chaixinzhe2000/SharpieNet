import tensorflow as tf
import os
import preprocess_rgb
import numpy as np
import postprocess_rgb

def main():
    LR_size = 100
    HR_size = 300
    run_trial_id = 100

    model = tf.keras.models.load_model('my_model.h5')
    dirname = os.path.dirname(__file__)
    test_path = os.path.join(dirname, "BSDS500/data/test")

    LR_test_images, HR_test_images, global_rgb_mean, global_rgb_std = preprocess_rgb.get_normalized_x_and_y(test_path,
                                                                                                            HR_size,
                                                                                                            LR_size)
    for i in range(len(LR_test_images)):
        # load and preprocess test image
        print(np.shape(LR_test_images[i]))
        input = np.expand_dims(LR_test_images[i], axis=0)
        print(np.shape(input))
        predicted_image = model.predict_l1(input)

        # print(predicted_image)
        postprocess_rgb.save_result(predicted_image[0] * 255, "predicted", str(i), run_trial_id)
        postprocess_rgb.save_result(HR_test_images[i] * 255, "HR", str(i), run_trial_id)
        postprocess_rgb.save_result(LR_test_images[i] * 255, "LR", str(i), run_trial_id)

if __name__ == "__main__":
    main()