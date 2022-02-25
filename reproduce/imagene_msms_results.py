from typing import Dict

def imagene_sim_config(selection_coeff: str) -> Dict:
    """
    Returns
    -------
    Dict: Configuration with settings equivalent to simulations produced in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
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
             'SELCOEFF': selection_coeff,
             }
        ]
    }
    return sim_config

def imagene_conversion_config(sorting: str) -> Dict:
    """
    Parameters
    ----------
    sorting: (str) - Type of sorting to be used on genetic images

    Returns
    -------
    Dict: Configuration with settings equivalent to processing in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x
    """
    conversion_config = [{'conversion_type': 'imagene',
                            'sorting': sorting,
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            }]
    return conversion_config


def imagene_model_config() -> Dict:
    """

    Returns
    -------
    Dict: Configuration with model equivalent to Imagene model from
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x

    """
    model_config = {
        'type': 'ml',
        'name': 'imagene',
        'max_pooling': True,
        'filters': 32,
        'depth': 3,
        'kernel_size': 3,
        'num_dense_layers': 1
    }
    return model_config


if __name__ == '__main__':
    getGPU()

    # Set training settings; same proportions as Imagene
    # Somewhat different epochs, training, and batch structure
    training_settings = {'epochs': 5,
                         'batch_size': 64,
                         'train_proportion': 0.8,
                         'validate_proportion': 0.1,
                         'test_proportion': 0.1
                         }

    # Plot confusion matrix for each sorting
    # 0.01 selection coefficient is equivalent to S=200 in paper
    sortings = ('None', 'Rows', 'Cols', 'RowsCols')
    accs = []
    for sorting in sortings:

        config =  {
            'train': {'simulations': imagene_sim_config('0.01'),
                      'conversions': imagene_conversion_config(sorting),
                      'training': training_settings},
            'model': imagene_model_config()
        }

        imagene = retrieve_model(config)(config)


    imagene = retrive_ml_model(settings['model']['type'])(settings)
    imagene.train()