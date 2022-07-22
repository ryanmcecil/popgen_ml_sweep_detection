
from reproduce.imagene.slim.imagene_results import imagene_sim_config, imagene_conversion_config, get_training_settings
from reproduce.imagene.slim.tiny_imagene_results import tiny_imagene_model_config
from util.util import getGPU
from reproduce.imagene.msms.tiny_imagene_stat_comparison import stat_comparison_plot_accs_and_corr_matrix

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
    stat_comparison_plot_accs_and_corr_matrix(model_config,'reproduce/imagene/slim/results/tiny_imagene',
                                              imagene_sim_config, imagene_conversion_config)
