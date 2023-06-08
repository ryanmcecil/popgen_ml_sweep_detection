import os
from typing import Dict

from reproduce.imagene_results.msms.imagene_results import (
    get_training_settings, imagene_conversion_config, imagene_model_config,
    train_test_sortings)
from util.util import getGPU


def imagene_sim_config(selection_coeff: str) -> Dict:
    """
    Returns
    -------
    Dict: Configuration with settings equivalent to simulations produced in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x but using SLiM software
    """
    sim_config = {
        'neutral': [
            {'software': 'slim',
             'template': 'msms_match.slim',
             'N': 50000,
             'NINDIV': '64'
             }
        ],
        'sweep': [
            {'software': 'slim',
             'template': 'msms_match_selection.slim',
             'N': 50000,
             'NINDIV': '64',
             'SELCOEFF': selection_coeff,
             }
        ]
    }
    return sim_config


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Testing Imagene Sortings')
    if not os.path.exists(os.path.join(os.getcwd(), 'reproduce/imagene_results/slim/results')):
        os.mkdir(os.path.join(
            os.getcwd(), 'reproduce/imagene_results/slim/results'))
    train_test_sortings(imagene_sim_config('0.01'),
                        imagene_conversion_config(),
                        get_training_settings(),
                        imagene_model_config(),
                        os.path.join(os.getcwd(), 'reproduce/imagene_results/slim/results/imagene_msms_cnm_results.png'))
