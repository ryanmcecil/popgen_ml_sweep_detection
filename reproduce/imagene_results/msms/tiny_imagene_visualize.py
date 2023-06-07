import os

from models.retrieve_model import retrieve_model
from reproduce.imagene_results.msms.tiny_imagene_results import (
    get_training_settings, imagene_conversion_config, imagene_sim_config,
    tiny_imagene_model_config)
from util.util import getGPU

if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Visualizing Tiny Imagene')

    config = {
        'train': {'simulations': imagene_sim_config('0.01'),
                  'conversions': imagene_conversion_config('Rows'),
                  'training': get_training_settings()},
        'model': tiny_imagene_model_config()
    }

    imagene = retrieve_model(config)(config)
    imagene.visualize_layer_outputs(
        'reproduce/imagene/msms/results/tiny_imagene_row_sorting', 1)
    imagene.visualize_parameters(os.path.join(
        os.getcwd(), 'reproduce/imagene/msms/results/tiny_imagene_row_sorting'))
