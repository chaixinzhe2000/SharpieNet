import PIL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import mpl_toolkits

# takes images of y (ranging from 0-1 values)
from tensorflow.python.keras.preprocessing.image import img_to_array


def rgb_to_yuv_normalized(image_data):
    y, Cb, Cr = image_data.convert("YCbCr").split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0  # maybe no need for astype
    return y, Cb, Cr


def yuv_to_rgb(predicted_img, cb, cr):
    img_y = predicted_img[0] * 255.0
    img_y = img_y.clip(0, 255)
    img_y = img_y.reshape((np.shape(img_y)[0], np.shape(img_y)[1]))
    img_y = PIL.Image.fromarray(np.uint8(img_y), mode="L")
    img_cb = cb.resize(img_y.size, PIL.Image.BICUBIC)
    img_cr = cr.resize(img_y.size, PIL.Image.BICUBIC)
    rgb_predicted_image = PIL.Image.merge("YCbCr", (img_y, img_cb, img_cr)).convert("RGB")
    return rgb_predicted_image

def save_result(img, resolution, file_name):
    """Plot the result with zoom-in area."""
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(resolution+"-"+file_name)
    # # zoom-factor: 2.0, location: upper-left
    # axins = zoomed_inset_axes(ax, 2, loc=2)
    # axins.imshow(img_array[::-1], origin="lower")

    # # Specify the limits.
    # x1, x2, y1, y2 = 200, 300, 100, 200
    # # Apply the x-limits.
    # axins.set_xlim(x1, x2)
    # # Apply the y-limits.
    # axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(file_name+"-"+resolution+".png")