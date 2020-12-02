import PIL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import mpl_toolkits

# takes images of y (ranging from 0-1 values)
from tensorflow.python.keras.preprocessing.image import img_to_array


def save_result(img, resolution, file_name, run_trial_id):
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
    plt.savefig("results/"+str(run_trial_id)+file_name+"-"+resolution+".png")



