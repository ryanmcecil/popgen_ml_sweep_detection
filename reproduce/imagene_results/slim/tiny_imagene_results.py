from util.util import getGPU
import os
from reproduce.imagene.slim.imagene_results import imagene_sim_config, imagene_conversion_config, train_test_sortings, \
    get_training_settings
from reproduce.imagene.msms.tiny_imagene_results import tiny_imagene_model_config

if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Testing Tiny Imagene Sortings')
    train_test_sortings(imagene_sim_config('0.01'),
                        imagene_conversion_config(),
                        get_training_settings(),
                        tiny_imagene_model_config(),
                        os.path.join(os.getcwd(), 'reproduce/imagene/slim/results/tiny_imagene_msms_cnm_results.png'))
