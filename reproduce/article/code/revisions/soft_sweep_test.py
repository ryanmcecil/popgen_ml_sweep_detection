import os

from models.retrieve_model import retrieve_model
from reproduce.article.code.configs import (stat_comparison_conversion_config,
                                            stat_comparison_training_settings)

# Code to check and see if H2/H1 is able to distinguish between our generated soft and hard sweeps
save_folder = os.path.join(
    os.getcwd(), 'reproduce/article/results/revisions')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

stat_configs = {
    "Garud's H": {'type': 'statistic',
                  'name': 'garud_h1'},
    "Garud's H12": {'type': 'statistic',
                    'name': 'garud_h12'},
    "Garud's H2/H1": {'type': 'statistic',
                      'name': 'garud_h2_h1'}

}

# Test Garud's H1 and Garud's H1/H2 on their ability to distinguish hard sweeps from soft sweeps
for stat_name, stat_config in stat_configs.items():
    stat_config = {
        'train': {'simulations': {
            'hard sweep': [
                {'software': 'slim',
                 'template': 'msms_match_selection.slim',
                 'N': 1000,
                 'NINDIV': '64',
                 'SELCOEFF': '0.01',
                 }
            ],
            'soft sweep': [
                {'software': 'slim',
                 'template': 'soft_sweep.slim',
                 'N': 1000,
                 'NINDIV': '64',
                 'SELCOEFF': '0.01',
                 'NMutatedIndivs': 600,
                 }
            ]
        },
            'conversions': stat_comparison_conversion_config(stat_name=stat_config['name']),
            'training': stat_comparison_training_settings()},
        'model': stat_config.copy()
    }
    stat_model = retrieve_model(stat_config.copy())(stat_config.copy())
    accuracy = stat_model.test(accuracy_value=True)
    print(accuracy)

# Test Garud's H1 and Garud's H1/H2 on their ability to distinguish hard sweeps from neutral
for stat_name, stat_config in stat_configs.items():
    stat_config = {
        'train': {'simulations': {
            'sweep': [
                {'software': 'slim',
                 'template': 'msms_match_selection.slim',
                 'N': 1000,
                 'NINDIV': '64',
                 'SELCOEFF': '0.01',
                 }
            ],
            'neutral': [
                {'software': 'slim',
                 'template': 'msms_match.slim',
                 'N': 1000,
                 'NINDIV': '64'
                 }
            ]
        },
            'conversions': stat_comparison_conversion_config(stat_name=stat_config['name']),
            'training': stat_comparison_training_settings()},
        'model': stat_config.copy()
    }
    stat_model = retrieve_model(stat_config.copy())(stat_config.copy())
    predictions, accuracy = stat_model.test(
        prediction_values=True, accuracy_value=True)
    print(max(predictions))
    print(min(predictions))
    print(accuracy)

# Test Garud's H1 and Garud's H1/H2 on their ability to distinguish hard sweeps from neutral
for stat_name, stat_config in stat_configs.items():
    stat_config = {
        'train': {'simulations': {
            'sweep': [
                {'software': 'slim',
                 'template': 'soft_sweep.slim',
                 'N': 1000,
                 'NINDIV': '64',
                 'SELCOEFF': '0.01',
                 'NMutatedIndivs': 600,
                 }
            ],
            'neutral': [
                {'software': 'slim',
                 'template': 'msms_match.slim',
                 'N': 1000,
                 'NINDIV': '64'
                 }
            ]
        },
            'conversions': stat_comparison_conversion_config(stat_name=stat_config['name']),
            'training': stat_comparison_training_settings()},
        'model': stat_config.copy()
    }
    stat_model = retrieve_model(stat_config.copy())(stat_config.copy())
    accuracy = stat_model.test(accuracy_value=True)
    print(accuracy)
