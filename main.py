# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from util.util import getGPU
from models.retrieve_model import retrieve_model

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sim_settings = {
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
             'SELCOEFF': '0.005',
             }
        ]
    }

    conversion_settings = [{'conversion_type': 'imagene',
                            'sorting': 'None',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            }]

    training_settings = {'epochs': 10,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1
                         }

    model_settings = {
        'type': 'ml',
        'name': 'imasortgene',
        'conv_depth': 3,
        'dense_depth': 3,
        'units': 128,
        'filters': 32,
        'kernel_size': 3,
        'image_height': 128,
        'image_width': 128,
        'depth': 3,
        'num_dense_layers': 3,
        'num_units': 128
    }

    settings = {
        'train': {'simulations': sim_settings,
                  'conversions': conversion_settings,
                   'training': training_settings},
        'model': model_settings
    }

    # config = {
    #     'train': {'simulations': sim_settings,
    #               'conversions': conversion_settings,
    #               'training': training_settings},
    #     'model': model_settings
    # }

    getGPU()

    imasortgene = retrieve_model(settings)(settings)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
