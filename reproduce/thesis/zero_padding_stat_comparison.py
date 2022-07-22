from typing import List, Dict
from arch_analysis import imagene_model_config, imagene_sim_config
from visualize_r_imagene import r_imagene_model_config
from simulate.popgen_simulators import retrieve_simulator
import numpy as np
from math import ceil
import os
from r_imagene_stat_comparison import stat_comparison_plot_accs_and_corr_matrix
from util.util import getGPU
import tensorflow as tf


def imagene_zero_padding_conversion_config(resize_dimensions: int) -> List:
    """
    Parameters
    ----------
    sorting: (str) - Type of sorting to be used during processing

    Returns
    -------
    List: Configuration with settings equivalent to processing in
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
    conversion_config = [{'conversion_type': 'zero_padding_imagene',
                          'sorting': 'Rows',
                          'min_minor_allele_freq': 0.01,
                          'resize_dimensions': resize_dimensions
                          }]
    return conversion_config


def write_to_txt_file(filename: str, max_width: int, rounded_max_width: int):
    with open(filename, 'w') as txt_file:
        txt_file.write(f'Max Width: {max_width}\n')
        txt_file.write(f'Max Rounded Width: {rounded_max_width}\n')


def load_max_width_values(filename: str):
    if os.path.isfile(filename):
        with open(filename, 'r') as txt_file:
            lines = txt_file.readlines()
            max_width = int([str(i) for i in lines[0].split() if i.isdigit()][0])
            max_rounded_with = int([str(i) for i in lines[1].split() if i.isdigit()][0])
            return max_width, max_rounded_with
    else:
        return None, None


def retrieve_max_width(sim_configs):
    widths = []
    for label in 'sweep', 'neutral':
        config = sim_configs[label][0]
        simulator = retrieve_simulator(config['software'])(config)
        for i in range(config['N']):
            id = i + 1
            widths.append(simulator.load_data(id_num=id, datatype='popgen_image').shape[1])
            if id % 100 == 0:
                print(id)
    max_width = np.max(widths)
    print(f'Maximum Width across 100,000 Images: {max_width}')
    return max_width


def roundup_to_next_hundredth(x):
    return int(ceil(x / 100.0)) * 100


def get_training_settings() -> Dict:
    """
    Returns
    -------
    Dict: Settings for training the Imagene model. This does not follow the original Imagene performance but
        is still able to achieve similar performance depending on the model initialization.
    """
    return {'epochs': 5,
            'batch_size': 32,
            'train_proportion': 0.8,
            'validate_proportion': 0.1,
            'test_proportion': 0.1
            }

if __name__ == '__main__':
    getGPU()

    txt_filename = os.path.join(os.getcwd(), 'reproduce/thesis/results/zero_padding_results_max_width.txt')

    max_width, max_rounded_width = load_max_width_values(txt_filename)
    if max_width is None:
        # Get max width of simulations
        max_width = retrieve_max_width(imagene_sim_config('0.01'))

        # Round up to next hundredth
        max_rounded_width = roundup_to_next_hundredth(max_width)

        # Store values
        write_to_txt_file(txt_filename,
                          max_width=max_width,
                          rounded_max_width=max_rounded_width)

   # sel_coeffs = ['0.01', '0.0075', '0.005']
    sel_coeffs = ['0.005']
    for sel_coeff in sel_coeffs:
        config = r_imagene_model_config()
        config['image_width'] = max_width
        r_imagene_config = {
            'train': {'simulations': imagene_sim_config(sel_coeff),
                      'conversions': imagene_zero_padding_conversion_config(max_width),
                      'training': get_training_settings()},
            'model': config
        }
        config = imagene_model_config()
        config['image_width'] = max_width
        imagene_config = {
            'train': {'simulations': imagene_sim_config(sel_coeff),
                      'conversions': imagene_zero_padding_conversion_config(max_width),
                      'training': get_training_settings()},
            'model': config
        }

        stat_comparison_plot_accs_and_corr_matrix([imagene_config, r_imagene_config],
                                                  ['Imagene', 'R-Imagene'],
                                                  base_dir = os.path.join(os.getcwd(), f'reproduce/thesis/results/zero_padding_sel_coeff_{sel_coeff}_imagene_stat_comparison'),
                                                  model_sim_config=imagene_sim_config(sel_coeff),
                                                  model_conversions_config=imagene_zero_padding_conversion_config(max_width))