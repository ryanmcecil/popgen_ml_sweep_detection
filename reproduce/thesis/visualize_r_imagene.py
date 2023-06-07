<<<<<<< HEAD
import csv
import os
from typing import Dict

from sklearn.metrics import roc_auc_score

from models.retrieve_model import retrieve_model
from reproduce.thesis.arch_analysis import (get_training_settings,
                                            imagene_conversion_config,
                                            imagene_sim_config)
from util.util import getGPU
=======
from typing import Dict
from reproduce.thesis.arch_analysis import get_training_settings, imagene_conversion_config, imagene_sim_config
from util.util import getGPU
from models.retrieve_model import retrieve_model
import os
import csv
from sklearn.metrics import roc_auc_score
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83


def r_imagene_model_config() -> Dict:
    """
    Returns
    -------
    Dict: Configuration with overall architecture similar to Imagene model but only one convolution kernel and no
        dense layers.
    """
    model_config = {
<<<<<<< HEAD
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': 128,
        'relu': True,
        'max_pooling': False,
        'convolution': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 2,
        'kernel_width': 1,
        'num_dense_layers': 0
    }
=======
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': False,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    return model_config


def r_imagene_regularized_model_config() -> Dict:
    """
    Returns
    -------
    Dict: Configuration with overall architecture similar to Imagene model but only one convolution kernel and no
        dense layers.
    """
    model_config = {
<<<<<<< HEAD
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': 128,
        'relu': True,
        'max_pooling': False,
        'convolution': True,
        'regularizer': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 2,
        'kernel_width': 1,
        'num_dense_layers': 0
    }
=======
             'type': 'ml',
             'name': 'imagene',
            'image_height': 128,
            'image_width': 128,
             'relu': True,
             'max_pooling': False,
             'convolution': True,
             'regularizer': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    return model_config


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Visualizing R-Imagene')

<<<<<<< HEAD
    out_file = os.path.join(
        os.getcwd(), 'reproduce/thesis2/results/r-imagene_visualization_acc.csv')
=======
    out_file = os.path.join(os.getcwd(), 'reproduce/thesis/results/r-imagene_visualization_acc.csv')
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model', 'Accuracy', 'AUC'])

        for model_name, model_config in [('r-imagene', r_imagene_model_config())]:
            config = {
                'train': {'simulations': imagene_sim_config('0.01'),
                          'conversions': imagene_conversion_config('Rows'),
                          'training': get_training_settings()},
                'model': model_config
            }

            imagene = retrieve_model(config)(config)
<<<<<<< HEAD
            imagene.visualize_layer_outputs(os.path.join(
                os.getcwd(), f'reproduce/thesis2/results/{model_name}'), 1)
            imagene.visualize_parameters(os.path.join(
                os.getcwd(), f'reproduce/thesis2/results/{model_name}'))
=======
            imagene.visualize_layer_outputs(os.path.join(os.getcwd(), f'reproduce/thesis/results/{model_name}'), 1)
            imagene.visualize_parameters(os.path.join(os.getcwd(), f'reproduce/thesis/results/{model_name}'))
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

            prediction_vals, label_vals, acc_val = imagene.test(prediction_values=True, label_values=True,
                                                                accuracy_value=True, test_in_batches=True)
            roc_auc = roc_auc_score(label_vals, prediction_vals)
<<<<<<< HEAD
            writer.writerow([model_name, f'{acc_val:.3f}', f'{roc_auc:.3f}'])
=======
            writer.writerow([model_name, f'{acc_val:.3f}', f'{roc_auc:.3f}'])
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83