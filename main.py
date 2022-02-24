# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


from models.mlmodels import retrive_ml_model
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
             'SELCOEFF': '0.01',
             }
        ]
    }

    conversion_settings = [{'conversion_type': 'imagene',
                            'sorting': 'Rows',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            }]

    training_settings = {'epochs': 5,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1
                         }

    model_settings = {
        'type': 'ml',
        'name': 'imagene',
        'max_pooling': True,
        'filters': 1,
        'depth': 1,
        'kernel_size': 3,
        'num_dense_layers': 0
    }

    settings = {
        'train': {'simulations': sim_settings,
                  'conversions': conversion_settings,
                   'training': training_settings},
        'model': model_settings
    }

    getGPU()

    imagene = retrive_ml_model(settings['model']['type'])(settings)
    imagene.train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
