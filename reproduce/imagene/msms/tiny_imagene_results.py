from typing import Dict
from util.util import getGPU
import os
from reproduce.imagene.msms.imagene_results import imagene_sim_config, imagene_conversion_config, train_test_sortings, \
    get_training_settings


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
