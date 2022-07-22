import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from matplotlib import pyplot as plt


def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def save_grey_image(image, filename: str, colorbar: bool = True, xticks = None, yticks = None):
    plt.clf()
    plt.imshow(image, cmap='Greys_r')
    if colorbar:
        plt.colorbar()
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.savefig(filename)