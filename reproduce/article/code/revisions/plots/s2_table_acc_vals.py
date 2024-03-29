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
from reproduce.article.code.widths import retrieve_max_width_from_settings
from util.util import getGPU

# Re-generates accuracy values for S2 table
#############################################

# Filepaths
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_file = os.path.join(save_folder, 'model_performances.csv')

# Configs for simulations and training settings
simulation_configs = {
    'Single Pop Model':
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
     }},
    'Three Pop Model: Sweep in YRI': {
        'simulations': {
            'neutral': [
                {'software': 'slim',
                 'template': 'schaffner_model_neutral.slim',
                 'N': 10000,
                 'NINDIV': '64'
                 }
            ],
            'sweep': [{
                'software': 'slim',
                'template': 'schaffner_model_sweep.slim',
                'N': 10000,
                'NINDIV': '64',
                'SELCOEFF': '0.01',
                'SWEEPPOP': 1,
            }
            ]
        },
    }}

sim_config = deepcopy(simulation_configs['Three Pop Model: Sweep in YRI'])
sim_config['simulations']['sweep'][0]['SWEEPPOP'] = 2
simulation_configs['Three Pop Model: Sweep in CEU'] = deepcopy(sim_config)
sim_config['simulations']['sweep'][0]['SWEEPPOP'] = 3
simulation_configs['Three Pop Model: Sweep in 3'] = deepcopy(sim_config)

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

stat_models = {
    "Garud's H": {'type': 'statistic',
                  'name': 'garud_h1'},
    "Tajima's D": {'type': 'statistic',
                   'name': 'tajima_d'},
    "IHS": {'type': 'statistic',
            'name': 'ihs_maxabs',
            'standardized': True},
}


sel_coeffs = ['0.01', '0.005']

conversion_config = {'conversion_type': 'imagene',
                     'sorting': 'Rows',
                     'min_minor_allele_freq': 0.01,
                     'resize_dimensions': 128
                     }

training_config = {'epochs': 2,
                   'batch_size': 64,
                   'train_proportion': 0.8,
                   'validate_proportion': 0.1,
                   'test_proportion': 0.1,
                   'best_of': 10
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
            # get conversion config based on min width
            sim_conversion_training_config['conversions'] = [
                deepcopy(conversion_config)]
            if 'SWEEPPOP' in sim_config['simulations']['sweep'][0].keys():
                sim_conversion_training_config['conversions'][0]['pop'] = sim_config['simulations']['sweep'][0]['SWEEPPOP']
            model_config = deepcopy(model_config)
            config = {
                'train': sim_conversion_training_config,
                'model': model_config
            }
            print(config)
            evaluate(config, save_file, [sim_name,
                                         sel_coeff, model_name])
        for stat_name, stat_config in stat_models.items():
            if 'SWEEPPOP' in sim_config['simulations']['sweep'][0].keys():
                conv_config = stat_comparison_conversion_config(
                    stat_name=stat_config['name'], pop=sim_config['simulations']['sweep'][0]['SWEEPPOP'])
            else:
                conv_config = stat_comparison_conversion_config(
                    stat_name=stat_config['name'], pop=None)
            stat_config = {
                'train': {'simulations': deepcopy(sim_conversion_training_config['simulations']),
                          'conversions': conv_config,
                          'training': stat_comparison_training_settings()},
                'model': deepcopy(stat_config)
            }
            print("Testing Garud's H on Data")
            evaluate(stat_config, save_file, [sim_name,
                                              sel_coeff, stat_name])
