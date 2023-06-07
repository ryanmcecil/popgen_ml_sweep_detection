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

# Producing results to analyze how the models act on soft sweeps
##################################################################################################

# Filepaths
# save_folder = os.path.join(
#     os.getcwd(), 'reproduce/article/results/revisions')
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# save_dir = os.path.join(save_folder, 'soft_sweeps_results')
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)


# # Function to train models in different processes
# def _evaluate(config, save_file, settings):
#     getGPU()
#     with open(save_file, 'a') as csv_file:
#         writer = csv.writer(csv_file)
#         model = retrieve_model(config)(config.copy())
#         data_acc = model.test(accuracy_value=True,
#                               test_in_batches=True)
#         output = settings + data_acc
#         writer.writerow(output)


# def evaluate(config, save_file, settings):
#     p = Process(target=_evaluate, args=(config, save_file, settings))
#     p.start()
#     p.join()
#     p.terminate()


# experiments = [test_architectures, stat_comparison,
#                visualize_models]

# settings = {'simulations': {
#     'neutral': [
#             {'software': 'slim',
#              'template': 'msms_match.slim',
#              'N': 20000,
#              'NINDIV': '64'
#              }
#             ],
#     'sweep': [
#         {'software': 'slim',
#          'template': 'soft_sweep.slim',
#          'N': 20000,
#          'NINDIV': '64',
#          'SELCOEFF': '0.01',
#          'NMutatedIndivs': 600,
#          }
#     ]
# },
#     'conversions': [{'conversion_type': 'imagene',
#                      'sorting': 'Rows',
#                      'min_minor_allele_freq': 0.01,
#                      'resize_dimensions': 128
#                      }],
#     'training': {'epochs': 2,
#                  'batch_size': 64,
#                  'train_proportion': 0.8,
#                  'validate_proportion': 0.1,
#                  'test_proportion': 0.1,
#                  'best_of': 10
#                  }}

# # Run experiments
# for exp in experiments:
#     run_process(exp, settings, save_dir, True)

# Comparison to hard sweep shap images
################################################
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_dir = os.path.join(save_folder, 'msms_hard_sweep_results')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


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


experiments = [visualize_models]

settings = {'simulations': {
    'neutral': [
            {'software': 'slim',
             'template': 'msms_match.slim',
             'N': 20000,
             'NINDIV': '64'
             }
            ],
    'sweep': [
        {'software': 'slim',
         'template': 'msms_match_selection.slim',
         'N': 20000,
         'NINDIV': '64',
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

# Run experiments
for exp in experiments:
    run_process(exp, settings, save_dir, True)
