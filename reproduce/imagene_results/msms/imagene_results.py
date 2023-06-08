import copy
import os
from typing import Dict, List

import tensorflow as tf
from matplotlib import pyplot as plt

from models.retrieve_model import retrieve_model
from util.util import getGPU


def imagene_sim_config(selection_coeff: str) -> Dict:
    """
    Returns
    -------
    Dict: Configuration with settings equivalent to simulations produced in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
    sim_config = {
        'neutral': [
            {'software': 'msms',
             'NREF': '10000',
             'N': 50000,
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ': '`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': '0',
             }
        ],
        'sweep': [
            {'software': 'msms',
             'N': 50000,
             'NREF': '10000',
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ': '`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': selection_coeff,
             }
        ]
    }
    return sim_config


def imagene_conversion_config(sorting: str = 'None') -> List:
    """
    Parameters
    ----------
    sorting: (str) - Type of sorting to be used during processing

    Returns
    -------
    List: Configuration with settings equivalent to processing in
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
    conversion_config = [{'conversion_type': 'imagene',
                          'sorting': sorting,
                          'min_minor_allele_freq': 0.01,
                          'resize_dimensions': 128
                          }]
    return conversion_config


def imagene_model_config() -> Dict:
    """
    Returns
    -------
    Dict: Configuration with model equivalent to Imagene model from
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x

    """
    model_config = {
        'type': 'ml',
        'name': 'imagene',
        'convolution': True,
        'max_pooling': True,
        'relu': True,
        'filters': 32,
        'image_height': 128,
        'image_width': 128,
        'depth': 3,
        'kernel_height': 3,
        'kernel_width': 3,
        'num_dense_layers': 1
    }
    return model_config


def train_test_sortings(sim_config: Dict,
                        conversion_config: List,
                        train_config: Dict,
                        model_config: Dict,
                        output_file: str):
    """Trains and tests each sorting. Stores results in confusion matrices and saves png.

    Parameters
    ----------
    sim_config: (Dict) - Simulation configuration
    conversion_config: (List) - Conversion configuration
    train_config: (Dict) - Training configuration
    model_config: (Dict) - Model configuration
    output_file: (str) - Png output file to save results
    """
    sortings = ('None', 'Rows', 'Cols', 'RowsCols')
    fix, axes = plt.subplots(1, 4, figsize=(15, 5))
    for sorting, ax in zip(sortings, axes.flatten()):
        conversion_config[0]['sorting'] = sorting
        config = {
            'train': {'simulations': copy.deepcopy(sim_config),
                      'conversions': copy.deepcopy(conversion_config),
                      'training': copy.deepcopy(train_config)},
            'model': copy.deepcopy(model_config)
        }
        tf.keras.backend.clear_session()
        imagene = retrieve_model(config)(config)
        imagene.test(plot_cm=True, ax=ax, test_in_batches=True)
        ax.title.set_text(sorting)
        print(f'{sorting} has been tested')
    plt.tight_layout()
    plt.savefig(output_file)


def get_training_settings() -> Dict:
    """
    Returns
    -------
    Dict: Settings for training the Imagene model. This does not follow the original Imagene performance but
        is still able to achieve similar performance depending on the model initialization.
    """
    return {'epochs': 5,
            'batch_size': 64,
            'train_proportion': 0.8,
            'validate_proportion': 0.1,
            'test_proportion': 0.1
            }


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Testing Imagene Sortings')
    if not os.path.exists(os.path.join(os.getcwd(), 'reproduce/imagene_results/msms/results')):
        os.mkdir(os.path.join(
            os.getcwd(), 'reproduce/imagene_results/msms/results'))
    train_test_sortings(imagene_sim_config('0.01'),
                        imagene_conversion_config(),
                        get_training_settings(),
                        imagene_model_config(),
                        os.path.join(os.getcwd(), 'reproduce/imagene_results/msms/results/imagene_msms_cnn_results.png'))
