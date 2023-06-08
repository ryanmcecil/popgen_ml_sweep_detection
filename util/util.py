import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def save_grey_image(image, filename: str, colorbar: bool = True, xticks=None, yticks=None, cmap='Greys_r', symmetric_range: bool = False, logscale: bool = False,
                    scale_vs: float = None):
    plt.clf()
    if symmetric_range:
        abs_max = np.abs(np.max(image))
        abs_min = np.abs(np.min(image))
        max_val = max([abs_max, abs_min])
        vmin = -max_val
        vmax = max_val
    else:
        vmin = None
        vmax = None
    if scale_vs is not None:
        vmin = scale_vs*vmin
        vmax = scale_vs*vmax
    if logscale:
        plt.imshow(image, cmap=cmap, norm=LogNorm())
    else:
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    plt.savefig(filename)
