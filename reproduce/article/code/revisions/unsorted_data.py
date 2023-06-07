# Create results directory
import csv
import os
from copy import deepcopy
from multiprocessing import Process

from models.retrieve_model import retrieve_model
from reproduce.article.code.arch_analysis import test_architectures
from reproduce.article.code.run import run_process
from reproduce.article.code.stat_comparison import stat_comparison
from reproduce.article.code.visualize import visualize_models
from util.util import getGPU

# This file is for creating new results based on first round of revision process for PLoS Comp Bio
##################################################################################################

# Producing results to analyze if early stopping or 'simulation on the fly' changes performances
##################################################################################################

# Filepaths
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_dir = os.path.join(save_folder, 'msms_unsorted_results')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


sel_coeffs = ['0.01', '0.005']

conversion_config = {'conversion_type': 'imagene',
                     'sorting': 'Rows',
                     'min_minor_allele_freq': 0.01,
                     'resize_dimensions': 128
                     }

# Function to train models in different processes


def _evaluate(config, save_file, settings):
    getGPU()
    with open(save_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        model = retrieve_model(config)(config.copy())
        data_acc = model.test(accuracy_value=True,
                              test_in_batches=True)
        output = settings + data_acc
        writer.writerow(output)


def evaluate(config, save_file, settings):
    p = Process(target=_evaluate, args=(config, save_file, settings))
    p.start()
    p.join()
    p.terminate()


experiments = [test_architectures, stat_comparison,
               visualize_models]

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
                     'sorting': 'None',
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

# Run experiments
for exp in experiments:
    run_process(exp, settings, save_dir, True)


# Visualize multiple kernels
#####################################################

model_config = {
    'type': 'ml',
    'name': 'imagene',
    'image_height': 128,
    'image_width': 128,
    'relu': True,
    'max_pooling': True,
    'convolution': True,
    'filters': 4,
    'depth': 1,
    'kernel_height': 3,
    'kernel_width': 3,
    'num_dense_layers': 0
}

getGPU()
out_dir = os.path.join(save_dir, f'Mini-CNN-4_3x3_Kernels_visualization/')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
config = {
    'train': deepcopy(settings),
    'model': deepcopy(model_config)
}
imagene = retrieve_model(config)(config)
imagene.visualize_layer_outputs(out_dir, 1)
imagene.visualize_parameters(out_dir)
