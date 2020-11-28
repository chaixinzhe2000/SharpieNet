import PIL
import tensorflow as tf
import numpy as np

# takes images of y (ranging from 0-1 values)
from tensorflow.python.keras.preprocessing.image import img_to_array


def y_to_rgb_normalized(images):
    y, Cb, Cr = images.convert("YCbCr").split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0  # maybe no need for astype
    return y, Cb, Cr




def restore_image(predicted_img, cb, cr):
    img_y = predicted_img[0] * 255.0
    img_y = img_y.clip(0, 255)
    img_y = img_y.reshape((np.shape(img_y)[0], np.shape(img_y)[1]))
    img_y = PIL.Image.fromarray(np.uint8(img_y), mode="L")
    img_cb = cb.resize(img_y.size, PIL.Image.BICUBIC)
    img_cr = cr.resize(img_y.size, PIL.Image.BICUBIC)
    return PIL.Image.merge("YCbCr", (img_y, img_cb, img_cr)).convert("RGB")
