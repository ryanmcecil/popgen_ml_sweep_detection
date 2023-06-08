import csv
import os
from copy import deepcopy
from multiprocessing import Process

from models.retrieve_model import retrieve_model
from reproduce.article.code.arch_analysis import test_architectures
from reproduce.article.code.configs import (ml_stat_comparison_model_configs,
                                            stat_comparison_configs,
                                            stat_comparison_conversion_config,
                                            stat_comparison_training_settings)
from reproduce.article.code.run import run_process
from reproduce.article.code.sample_complexity import test_sample_complexity
from reproduce.article.code.stat_comparison import stat_comparison
from reproduce.article.code.visualize import visualize_models
from util.util import getGPU

# This file is for creating new results based on first round of revision process for PLoS Comp Bio
##################################################################################################

# Producing results to analayze effect of increasing number of haplotypes
##################################################################################################

# Filepaths
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')

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

    'Mini-CNN': {
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

    'DeepSet': {
        'type': 'ml',
        'name': 'deepset',
        'filters': 64,
        'image_height': 128,
        'image_width': 128,
        'kernel_size': 5,
        'depth': 2,
        'num_dense_layers': 2,
        'num_units': 64
    }
}


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


# Filepaths
save_file = os.path.join(save_folder, 'large_num_haplotypes_analysis.csv')

# Configs for simulations and training settings
simulation_configs = {
    'imagene_sim_128_row_sorted':
    {'simulations':
     {
         'neutral': [
             {'software': 'msms',
              'NREF': '10000',
              'N': 50000,
              'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
              'LEN': '80000',
              'THETA': '48',
              'RHO': '32',
              'NCHROMS': '1000',
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
              'NCHROMS': '1000',
              'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
              'FREQ': '`bc <<< \'scale=6; 1/100\'`',
              'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
              'SELCOEFF': '0.01',
              }
         ]
     }}}

sim_config = deepcopy(simulation_configs['imagene_sim_128_row_sorted'])
sim_config['simulations']['neutral'][0]['NCHROMS'] = '1000'
sim_config['simulations']['sweep'][0]['NCHROMS'] = '1000'
simulation_configs['imagene_sim_1000_row_sorted'] = deepcopy(sim_config)
del simulation_configs['imagene_sim_128_row_sorted']

training_config = {'epochs': 2,
                   'batch_size': 64,
                   'train_proportion': 0.8,
                   'validate_proportion': 0.1,
                   'test_proportion': 0.1,
                   'best_of': 10
                   }

# Write header to csv file
with open(save_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Simulation Type', 'SELCOEFF',
                    'Model', 'Accuracy'])

# Train and test models on different number of haplotypes
for sel_coeff in sel_coeffs:
    for sim_name, sim_config in simulation_configs.items():
        for model_name, model_config in models.items():
            sim_conversion_training_config = deepcopy(
                sim_config)  # get simulation config
            sim_conversion_training_config['training'] = deepcopy(
                training_config)  # get training config
            # set selection coefficient
            sim_conversion_training_config['simulations']['sweep'][0]['SELCOEFF'] = sel_coeff
            conversion_config = deepcopy(conversion_config)
            model_config = deepcopy(model_config)
            if sim_config['simulations']['neutral'][0]['NCHROMS'] == '1000':
                model_config['image_width'] = 200
                model_config['image_height'] = 200
                conversion_config['resize_dimensions'] = 200
            sim_conversion_training_config['conversions'] = [
                deepcopy(conversion_config)]  # get conversion config
            config = {
                'train': sim_conversion_training_config,
                'model': model_config
            }
            print('Training Model with Config')
            print(config)
            print('------------------------------------')
            evaluate(config, save_file, [sim_name,
                                         sel_coeff, model_name])
        # Also test Garud's H and report result
        stat_config = {
            'train': {'simulations': deepcopy(sim_conversion_training_config['simulations']),
                      'conversions': stat_comparison_conversion_config(stat_name='garud_h1', pop=None),
                      'training': stat_comparison_training_settings()},
            'model': {'type': 'statistic',
                      'name': 'garud_h1'}
        }
        print("Testing Garud's H on Data")
        evaluate(stat_config, save_file, [sim_name,
                                          sel_coeff, "Garud's H"])
