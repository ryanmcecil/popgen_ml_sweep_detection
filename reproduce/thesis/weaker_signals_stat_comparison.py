"""For comparing R-Imagene results across weaker and stronger signals"""
from arch_analysis import imagene_model_config, get_training_settings, imagene_sim_config, imagene_conversion_config
from visualize_r_imagene import r_imagene_model_config
from r_imagene_stat_comparison import stat_comparison_plot_accs_and_corr_matrix
import tensorflow as tf
import os
from util.util import getGPU

if __name__ == '__main__':
    getGPU()
    sel_coeffs = ['0.0075', '0.005']
    for sel_coeff in sel_coeffs:

        r_imagene_config = {
            'train': {'simulations': imagene_sim_config(sel_coeff),
                      'conversions': imagene_conversion_config('Rows'),
                      'training': get_training_settings()},
            'model': r_imagene_model_config()
        }
        imagene_config = {
            'train': {'simulations': imagene_sim_config(sel_coeff),
                      'conversions': imagene_conversion_config('Rows'),
                      'training': get_training_settings()},
            'model': imagene_model_config()
        }

        stat_comparison_plot_accs_and_corr_matrix([imagene_config, r_imagene_config],
                                                  ['Imagene', 'R-Imagene'],
                                                  base_dir=os.path.join(os.getcwd(),
                                                                        f'reproduce/thesis/results/selcoeff_{sel_coeff}_imagene_stat_comparison'),
                                                  model_sim_config=imagene_sim_config(sel_coeff),
                                                  model_conversions_config=imagene_conversion_config('Rows'))