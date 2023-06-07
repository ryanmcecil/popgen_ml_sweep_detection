import os
from typing import Dict

from reproduce.imagene_results.msms.imagene_results import (
    get_training_settings, imagene_conversion_config, imagene_sim_config,
    train_test_sortings)
from util.util import getGPU


def tiny_imagene_model_config() -> Dict:
    """
    Returns
    -------
    Dict: Configuration with overall architecture similar to Imagene model but only one convolution kernel and no
        dense layers.
    """
    model_config = {
        'type': 'ml',
        'name': 'imagene',
        'convolution': True,
        'max_pooling': True,
        'relu': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 3,
        'kernel_width': 3,
        'num_dense_layers': 0
    }
    return model_config


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Testing Tiny Imagene Sortings')
    train_test_sortings(imagene_sim_config('0.01'),
                        imagene_conversion_config(),
                        get_training_settings(),
                        tiny_imagene_model_config(),
                        os.path.join(os.getcwd(), 'reproduce/imagene/msms/results/tiny_imagene_msms_cnm_results.png'))
