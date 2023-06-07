
from reproduce.imagene_results.msms.tiny_imagene_stat_comparison import \
    stat_comparison_plot_accs_and_corr_matrix
from reproduce.imagene_results.slim.imagene_results import (
    get_training_settings, imagene_conversion_config, imagene_sim_config)
from reproduce.imagene_results.slim.tiny_imagene_results import \
    tiny_imagene_model_config
from util.util import getGPU

if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Comparing Tiny Imagenene to Statistics')
    conv_config = imagene_conversion_config()
    conv_config[0]['sorting'] = 'Rows'
    model_config = {
        'train': {'simulations': imagene_sim_config('0.01'),
                  'conversions': conv_config,
                  'training': get_training_settings()},
        'model': tiny_imagene_model_config()
    }
    stat_comparison_plot_accs_and_corr_matrix(model_config, 'reproduce/imagene/slim/results/tiny_imagene',
                                              imagene_sim_config, imagene_conversion_config)
