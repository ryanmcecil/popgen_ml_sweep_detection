from models.retrieve_model import retrieve_model
from models.popgen_summary_statistics import all_statistics, all_image_and_position_statistics
from arch_analysis import imagene_conversion_config, imagene_sim_config, get_training_settings, imagene_model_config
from typing import Dict
import csv
from visualize_r_imagene import r_imagene_model_config
from util.util import getGPU
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from copy import deepcopy


def stat_model_config(name: str) -> Dict:
    """Returns the configuration for a summary statistic model

    Parameters
    ----------
    name: (str) - Name of statistic

    Returns
    -------
    Dict: Summary statistic model configuration

    """
    return {
        'type': 'statistic',
        'name': name
    }


def stat_training_settings() -> Dict:
    """
    Returns
    -------
    Dict: Settings for training statistic
    """
    return {
        'train_proportion': 0.8,
        'validate_proportion': 0.1,
        'test_proportion': 0.1
    }


def raw_conversion_config():
    conversion_config = [{'conversion_type': 'raw_data',
                          'datatype': 'popgen_image',
                          }]
    return conversion_config


def stat_comparison_plot_accs_and_corr_matrix(model_configs,
                                              model_names,
                                              base_dir: str,
                                              model_sim_config,
                                              model_conversions_config):
    """Plots model and statistic accuracies, and a correlation matrix between the prediction outputs. Saves
    the result sin base_dir. Will do so for both the statistics trained on the model conversions and the raw
    data.

    Parameters
    ----------
    model_config: (Dict) - Confiuration of ML model
    base_dir: (str) - Filename to append to beginning of saved files
    """

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    assert len(model_configs) == len(model_names)

    data_predictions, data_acc = {}, {}
    # Get model data
    for i, config in enumerate(model_configs):
        print(f'Testing {model_names[i]}')
        model = retrieve_model(config)(config.copy())
        data_predictions[model_names[i]], data_acc[model_names[i]] = model.test(prediction_values=True,
                                                accuracy_value=True,
                                                test_in_batches=True)

    # Test each statistic on same processed data and on raw data
    # for stat_conv_config, conv_config_name in \
    #        [(raw_conversion_config(), 'raw_data'), (model_conversions_config, 'model_conv')]:
    for stat_conv_config, conv_config_name in \
            [(model_conversions_config, 'model_conv')]:

        data_predicts = data_predictions.copy()
        data_accs = data_acc.copy()

        print(data_accs)

        for stat in all_statistics():
            if stat not in all_image_and_position_statistics():
                print(f'Testing {stat}')
                stat_config = {
                    'train': {'simulations': deepcopy(model_sim_config),
                              'conversions': deepcopy(stat_conv_config),
                              'training': stat_training_settings()},
                    'model': stat_model_config(name=stat)
                }
                stat_model = retrieve_model(stat_config)(stat_config)
                if stat == 'garud_h1':
                    stat_name = 'Garud\'s H1'
                elif stat == 'tajima_d':
                    stat_name = 'Tajima\'s D'
                else:
                    print(stat)
                    raise NotImplementedError
                data_predicts[stat_name], data_accs[stat_name] = stat_model.test(prediction_values=True,
                                                accuracy_value=True,
                                                test_in_batches=True)

        base_filename = os.path.join(base_dir, f'stat_conv_{conv_config_name}')
        print('Plotting')
        # Save csv file of accuracies
        with open(f'{base_filename}_accs_comparison.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(list(data_accs.keys()))
            writer.writerow(list(data_accs.values()))

        # Save correlation matrix
        df = pd.DataFrame(data_predicts)
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(method='spearman'), annot=True, fmt='.2f',
                    cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        plt.title('Correlation Matrix')
        plt.savefig(f'{base_filename}_corr_matrix.png', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Comparing Imagene and R-Imagene to Statistics')
    conv_config = imagene_conversion_config()
    conv_config[0]['sorting'] = 'Rows'
    r_imagene_config = {
        'train': {'simulations': imagene_sim_config('0.01'),
                  'conversions': conv_config,
                  'training': get_training_settings()},
        'model': r_imagene_model_config()
    }
    imagene_config = {
        'train': {'simulations': imagene_sim_config('0.01'),
                  'conversions': conv_config,
                  'training': get_training_settings()},
        'model': imagene_model_config()
    }
    stat_comparison_plot_accs_and_corr_matrix([imagene_config, r_imagene_config],
                                              ['Imagene', 'R-Imagene'],
                                              'reproduce/thesis/results/imagene_stat_comparison',
                                              imagene_sim_config('0.01'),
                                              imagene_conversion_config())

