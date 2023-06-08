
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from models.retrieve_model import retrieve_model
from util.util import getGPU

getGPU()

# Bar plot to visualize main differences in S1 table
#########################################################


def get_interval(accuracy, simulation_type):
    if 'Three Pop' in simulation_type:
        return 1.96*np.sqrt(accuracy*(1-accuracy)/2000)
    else:
        return 1.96*np.sqrt(accuracy*(1-accuracy)/10000)


save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')


# Get simulation, conversion, and training strategies
settings = {'simulations': {
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
                     'SELCOEFF': '0.01',
         }
    ]
},
    'conversions': [{'conversion_type': 'imagene',
                     'sorting': 'Rows',
                     'min_minor_allele_freq': 0.01,
                     'resize_dimensions': 128
                     }],
    'training': {'epochs': 2,
                 'batch_size': 64,
                 'train_proportion': 0.8,
                 'validate_proportion': 0.1,
                 'test_proportion': 0.1,
                 'best_of': 10
                 }}


# Get model configs to plot
models = {
    'Imagene': {
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': 128,
        'relu': True,
        'max_pooling': True,
        'convolution': True,
        'filters': 32,
        'depth': 3,
        'kernel_height': 3,
        'kernel_width': 3,
        'num_dense_layers': 1
    },

    '\n\nMini-CNN': {
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
    },

    'Mini-CNN with 1x2 Kernel': {
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': 128,
        'relu': True,
        'max_pooling': False,
        'convolution': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 1,
        'kernel_width': 2,
        'num_dense_layers': 0
    },

    '\n\nMini-CNN without Relu': {
        'type': 'ml',
        'name': 'imagene',
        'image_height': 128,
        'image_width': 128,
        'relu': False,
        'max_pooling': False,
        'convolution': True,
        'filters': 1,
        'depth': 1,
        'kernel_height': 2,
        'kernel_width': 1,
        'num_dense_layers': 0
    },
}

fig, ax = plt.subplots()
x_pos = range(len(models))

# Generate accuracy values, then plot in bar plot
accuracy_values = []
for model_name, model_config in models.items():
    sim_conversion_training_config = deepcopy(
        settings)  # get simulation config
    config = {
        'train': sim_conversion_training_config,
        'model': model_config
    }
    model = retrieve_model(config)(config.copy())
    data_acc = model.test(accuracy_value=True,
                          test_in_batches=True)
    accuracy_values.append(data_acc[0])

bars = ax.bar(x_pos, np.asarray(accuracy_values), align='center')
errors = [get_interval(accuracy, 'Single Pop') for accuracy in accuracy_values]
ax.errorbar(x_pos, accuracy_values, yerr=errors,
            fmt='none', capsize=5, color='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(models)

ax.set_ylabel('Accuracy')

ax.set_title('Single Pop Model Performance Comparison')

plt.subplots_adjust(bottom=0.2)

plt.savefig(os.path.join(save_folder, 'msms_performance_comparison.png'))
plt.savefig(os.path.join(
    save_folder, 'msms_performance_comparison.pdf'), format='pdf')
