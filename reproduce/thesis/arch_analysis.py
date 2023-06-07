<<<<<<< HEAD
import csv
import os
from typing import Dict, List

import tensorflow as tf
from sklearn.metrics import roc_auc_score

from models.retrieve_model import retrieve_model
from reproduce.thesis.architectures_to_test import architectures_to_test
from util.util import getGPU
=======
from typing import Dict, List
from reproduce.thesis.architectures_to_test import architectures_to_test
import os, csv
import tensorflow as tf
from util.util import getGPU
from models.retrieve_model import retrieve_model
from sklearn.metrics import roc_auc_score
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83


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
        'image_height': 128,
        'image_width': 128,
        'relu': True,
        'max_pooling': True,
        'filters': 32,
        'depth': 3,
        'convolution': True,
        'kernel_height': 3,
        'kernel_width': 3,
        'num_dense_layers': 1
    }
    return model_config


def imagene_sim_config(selection_coeff: str) -> Dict:
    """
    Returns
    -------
    Dict: Configuration with settings equivalent to simulations produced in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
    sim_config = {
        'neutral': [
<<<<<<< HEAD
            {'software': 'slim',
             'template': 'schaffner_model_neutral.slim',
             'N': 10000,
             'NINDIV': '64'
             }
        ],
        'sweep': [
            {'software': 'slim',
             'template': 'schaffner_model_sweep.slim',
             'N': 10000,
             'NINDIV': '64',
             'SELCOEFF': '0.01',
             'SWEEPPOP': 1,
=======
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
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
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
<<<<<<< HEAD
                          'resize_dimensions': 128,
                          'pop': 1
=======
                          'resize_dimensions': 128
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
                          }]
    return conversion_config


def get_training_settings() -> Dict:
    """
    Returns
    -------
    Dict: Settings for training the Imagene model. This does not follow the original Imagene performance but
        is still able to achieve similar performance depending on the model initialization.
    """
<<<<<<< HEAD
    return {'epochs': 20,
=======
    return {'epochs': 5,
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            'batch_size': 64,
            'train_proportion': 0.8,
            'validate_proportion': 0.1,
            'test_proportion': 0.1
            }


if __name__ == '__main__':
    getGPU()
<<<<<<< HEAD
    results_file = os.path.join(
        os.getcwd(), 'reproduce/thesis2/results/arch_analysis.csv')
=======
    results_file = os.path.join(os.getcwd(), 'reproduce/thesis/results/arch_analysis.csv')
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

    tested_archs = {}
    # if os.path.isfile(results_file):
    #     with open(results_file, 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         for row in reader:
    #             tested_archs[row[0]] = row[1]

    with open(results_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Architecture', 'Accuracy', 'AUC', 'Parameters'])

        for arch in architectures_to_test():
            if arch[0] not in tested_archs:
                config = {
                    'train': {'simulations': imagene_sim_config('0.01'),
                              'conversions': imagene_conversion_config('Rows'),
                              'training': get_training_settings()},
                    'model': arch[1]
                }
                tf.keras.backend.clear_session()
                imagene = retrieve_model(config)(config)
                print(imagene.model.summary())
<<<<<<< HEAD
                prediction_vals, label_vals, acc_val = imagene.test(
                    prediction_values=True, label_values=True, accuracy_value=True, test_in_batches=True)
=======
                prediction_vals, label_vals, acc_val = imagene.test(prediction_values=True, label_values=True, accuracy_value=True, test_in_batches=True)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
                num_parameters = imagene.number_of_parameters()
                roc_auc = roc_auc_score(label_vals, prediction_vals)
                print(acc_val)
                print(roc_auc)
<<<<<<< HEAD
                writer.writerow(
                    [arch[0], f'{acc_val:.3f}', f'{roc_auc:.3f}', num_parameters])
            else:
                pass
                # writer.writerow([arch[0], tested_archs[arch[0]]])
=======
                writer.writerow([arch[0], f'{acc_val:.3f}', f'{roc_auc:.3f}', num_parameters])
            else:
                pass
                #writer.writerow([arch[0], tested_archs[arch[0]]])

>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
