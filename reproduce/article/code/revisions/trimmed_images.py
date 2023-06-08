import csv
import os
from copy import deepcopy
from multiprocessing import Process

import tensorflow as tf

from models.retrieve_model import retrieve_model
from reproduce.article.code.arch_analysis import test_architectures
from reproduce.article.code.configs import stat_comparison_training_settings
from reproduce.article.code.visualize import visualize
from reproduce.article.code.widths import retrieve_max_width_from_settings
from util.util import getGPU

# This file is for creating new results based on first round of revision process for PLoS Comp Bio
##################################################################################################

# Producing results to analyze how trimming images changes the model performances
##################################################################################################

# Filepaths
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_file = os.path.join(save_folder, 'trimmed_images.csv')

# Configs for simulations and training settings
simulation_configs = {
    'imagene_sim_row_sorted':
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
    'schaffner_sweep_pop1_row_sorted': {
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
    }
}

sim_config = deepcopy(simulation_configs['schaffner_sweep_pop1_row_sorted'])
sim_config['simulations']['sweep'][0]['SWEEPPOP'] = 2
simulation_configs['schaffner_sweep_pop2_row_sorted'] = deepcopy(sim_config)
sim_config['simulations']['sweep'][0]['SWEEPPOP'] = 3
simulation_configs['schaffner_sweep_pop3_row_sorted'] = deepcopy(sim_config)

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

conversion_config = {'conversion_type': 'paring_imagene',
                     'sorting': 'Rows',
                     'min_minor_allele_freq': 0.01,
                     'resize_dimensions': None,
                     }

stat_conversion_config = {'conversion_typ0e': 'raw_pared_data'

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
    writer.writerow(['Simulation Type', 'SELCOEFF', 'Image Width',
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
            if 'pop' in sim_name:
                sim_conversion_training_config['conversions'][0]['pop'] = sim_config['simulations']['sweep'][0]['SWEEPPOP']
            min_width = retrieve_max_width_from_settings(
                sim_conversion_training_config, use_min=True)
            print(
                f'Training Models with Pared Down Data: Min Width is {min_width}')
            sim_conversion_training_config['conversions'][0]['resize_dimensions'] = min_width
            model_config = deepcopy(model_config)
            model_config['image_width'] = min_width
            config = {
                'train': sim_conversion_training_config,
                'model': model_config
            }
            print('Training Model with Config')
            print(config)
            print('------------------------------------')
            evaluate(config, save_file, [sim_name,
                                         sel_coeff, min_width, model_name])
        # Also test Garud's H and report result
        conv_config = deepcopy(conversion_config)
        conv_config['resize_dimensions'] = min_width
        if 'pop' in sim_name:
            conv_config['pop'] = sim_config['simulations']['sweep'][0]['SWEEPPOP']
        stat_config = {
            'train': {'simulations': deepcopy(sim_conversion_training_config['simulations']),
                      'conversions': [conv_config],
                      'training': stat_comparison_training_settings()},
            'model': {'type': 'statistic',
                      'name': 'garud_h1'}
        }
        print("Testing Garud's H on Data")
        evaluate(stat_config, save_file, [sim_name,
                                          sel_coeff, min_width, "Garud's H"])


# Visualize the shap explanations for the different models
##################################################################
# Filepaths
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
main_dir = os.path.join(save_folder, 'trimmed_sweeps_visualizations')
if not os.path.isdir(main_dir):
    os.mkdir(main_dir)


del simulation_configs['imagene_sim_row_sorted']
# del simulation_configs['schaffner_sweep_pop1_row_sorted']
del simulation_configs['schaffner_sweep_pop2_row_sorted']
del simulation_configs['schaffner_sweep_pop3_row_sorted']
for sim_name, sim_config in simulation_configs.items():
    save_dir = os.path.join(main_dir, f'{sim_name}_s=0.01')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for model_name, model_config in models.items():
        sim_conversion_training_config = deepcopy(
            sim_config)
        sim_conversion_training_config['training'] = deepcopy(
            training_config)
        sim_conversion_training_config['conversions'] = [
            deepcopy(conversion_config)]
        if 'pop' in sim_name:
            sim_conversion_training_config['conversions'][0]['pop'] = sim_config['simulations']['sweep'][0]['SWEEPPOP']
        min_width = retrieve_max_width_from_settings(
            sim_conversion_training_config, use_min=True)
        sim_conversion_training_config['conversions'][0]['resize_dimensions'] = min_width
        model_config = deepcopy(model_config)
        model_config['image_width'] = min_width
        config = {
            'train': sim_conversion_training_config,
            'model': model_config
        }
        if model_name != 'DeepSet':
            # Run Shap visualization
            p = Process(target=visualize, args=(model_name, model_config, save_dir,
                                                sim_conversion_training_config, True,))
            p.start()
            p.join()
            p.terminate()

            if model_name == 'R-Imagene' or 'Mini-CNN' in model_name:
                getGPU()
                out_dir = os.path.join(
                    save_dir, f'{model_name}_visualization/')
                config = {
                    'train': sim_conversion_training_config,
                    'model': model_config
                }
                imagene = retrieve_model(config)(config)
                imagene.visualize_layer_outputs(out_dir, 1)
                imagene.visualize_parameters(out_dir)
                tf.keras.backend.clear_session()
