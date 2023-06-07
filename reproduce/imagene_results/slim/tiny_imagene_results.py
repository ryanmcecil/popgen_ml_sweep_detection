import os

from reproduce.imagene_results.msms.tiny_imagene_results import \
    tiny_imagene_model_config
from reproduce.imagene_results.slim.imagene_results import (
    get_training_settings, imagene_conversion_config, imagene_sim_config,
    train_test_sortings)
from util.util import getGPU

if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Testing Tiny Imagene Sortings')
    train_test_sortings(imagene_sim_config('0.01'),
                        imagene_conversion_config(),
                        get_training_settings(),
                        tiny_imagene_model_config(),
                        os.path.join(os.getcwd(), 'reproduce/imagene/slim/results/tiny_imagene_msms_cnm_results.png'))
