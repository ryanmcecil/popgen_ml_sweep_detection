from models.retrieve_model import retrieve_model

# Test code to ensure that simulations, pre-processing, and model training are working

# MSMS One Pop demographic model
sim_config = {
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

# Convert the images using Imagene resizing and row sorting
conversion_config = {'conversion_type': 'imagene',
                     'sorting': 'Rows',
                     'min_minor_allele_freq': 0.01,
                     'resize_dimensions': 128
                     }

# Train the model 10 times with 2 epochs, and take the best performing model on the validation set
training_config = {'epochs': 2,
                   'batch_size': 64,
                   'train_proportion': 0.8,
                   'validate_proportion': 0.1,
                   'test_proportion': 0.1,
                   'best_of': 10
                   }

# Use ML model Imagene
model_config = {
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
}


# Retrieve model with settings
full_config = {
    'train': {'simulations': sim_config, 'conversions': [conversion_config], 'training': training_config},
    'model': model_config
}
imagene = retrieve_model(full_config)(full_config.copy())
acc = imagene.test(accuracy_value=True,
                   test_in_batches=True)
print(
    f'The accuracy of the trained Imagene model on the MSMS simulated data with selection coefficient 0.01 was {acc[0]}')
