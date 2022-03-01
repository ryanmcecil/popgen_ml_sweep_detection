from models.retrieve_model import retrieve_model
from models.popgen_summary_statistics import all_statistcs, all_image_and_position_statistics
from reproduce.imagene.msms.imagene_results import imagene_sim_config, imagene_conversion_config, get_training_settings
from reproduce.imagene.msms.tiny_imagene_results import tiny_imagene_model_config
from typing import Dict
import csv
from sklearn.metrics import precision_score
from scipy.stats import spearmanr
import numpy as np
import os
from util.util import getGPU
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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


def compare_model_to_msms_imagene_statistics(model_config, output_file):
    raise NotImplementedError
    print('=========================================================')
    print('Comparing model to msms Imagene statistics')
    with open(output_file, 'w') as csv_file:
        rows = []
        writer = csv.writer(csv_file)
        rows.append(['statistic', 'stat_acc', 'stat_tp', 'stat_tn', 'ml_model', 'ml_acc', 'ml_tp', 'ml_tn', 'Corr'])
        # First train and test model
        model = retrieve_model(model_config)(model_config)
        model_predictions, model_classifications, model_labels, model_acc = model.test(prediction_values=True,
                                                                                       classification_values=True,
                                                                                       label_values=True,
                                                                                       accuracy_value=True,
                                                                                       test_in_batches=True)
        model_predictions = np.asarray(model_predictions)
        model_classifications = np.asarray(model_classifications)
        model_labels = np.asarray(model_labels)
        for stat in all_statistcs():
            print(f'Comparing {stat} to model output')
            conversions = imagene_conversion_config()
            if stat in all_image_and_position_statistics():
                conversions.append({
                    'conversion_type': 'raw_data',
                    'datatype': 'popgen_positions'
                })
            stat_config = {
                'train': {'simulations': imagene_sim_config('0.01'),
                          'conversions': conversions,
                          'training': stat_training_settings()},
                'model': stat_model_config(name=stat)
            }
            stat_model = retrieve_model(stat_config)(stat_config)
            stat_predictions, stat_classifications, stat_labels, stat_acc = stat_model.test(prediction_values=True,
                                                                                            classification_values=True,
                                                                                            label_values=True,
                                                                                            accuracy_value=True)
            stat_predictions = np.asarray(stat_predictions)
            stat_classifications = np.asarray(stat_classifications)
            stat_labels = np.asarray(stat_labels)
            new_row = []
            new_row.append(stat)
            new_row.append(stat_acc)
            new_row.append(precision_score(stat_labels, stat_classifications))
            new_row.append(np.sum(stat_labels[stat_labels == 0] == stat_classifications[stat_labels == 0]) / np.sum(
                stat_labels == 0))
            ###################################
            new_row.append(model_config['model']['name'])
            new_row.append(model_acc)
            new_row.append(precision_score(model_labels, model_classifications))
            new_row.append(np.sum(model_labels[model_labels == 0] == model_classifications[model_labels == 0]) / np.sum(
                model_labels == 0))
            #####################################3
            corr, p_value = spearmanr(a=stat_predictions, b=model_predictions)
            ########################################
            new_row.append(corr)
            rows.append(new_row)
        writer.writerows(rows)


def stat_comparison_plot_accs_and_corr_matrix(model_config: Dict, base_filename: str):
    """Plots model and statistic accuracies, and a correlation matrix between the prediction outputs

    Parameters
    ----------
    model_config: (Dict) - Confiuration of ML model
    base_filename: (str) - Filename to append to beginning of saved files
    """

    # Get model data
    print('Testing model')
    model = retrieve_model(model_config)(model_config)
    data_predictions, data_acc = {}, {}
    data_predictions['tiny_imagene'], data_acc['tiny_imagene'] = model.test(prediction_values=True,
                                            accuracy_value=True,
                                            test_in_batches=True)

    # Get statistic data
    for stat in all_statistcs():
        conversions = imagene_conversion_config()
        if stat not in all_image_and_position_statistics():
            print(f'Testing {stat}')
            stat_config = {
                'train': {'simulations': imagene_sim_config('0.01'),
                          'conversions': conversions,
                          'training': stat_training_settings()},
                'model': stat_model_config(name=stat)
            }
            stat_model = retrieve_model(stat_config)(stat_config)
            data_predictions[stat], data_acc[stat] = stat_model.test(prediction_values=True,
                                            accuracy_value=True,
                                            test_in_batches=True)

    print('Plotting')
    # Save csv file of acuracies
    with open(f'{base_filename}_stat_accs.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list(data_acc.keys()))
        writer.writerow(list(data_acc.values()))

    # Save correlation matrix
    df = pd.DataFrame(data_predictions)
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(method='spearman'), annot=True, fmt='.4f',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.savefig(f'{base_filename}_stat_corr.png', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    getGPU()
    print('---------------------------------')
    print('Comparing Tiny Imagenene to Statistics')
    conv_config = imagene_conversion_config()
    conv_config[0]['sorting'] = 'Rows'
    model_config = {
        'train': {'simulations': imagene_sim_config('0.01'),
                  'conversions': conv_config,
                  'training': get_training_settings()},
        'model': tiny_imagene_model_config()
    }
    stat_comparison_plot_accs_and_corr_matrix(model_config,'reproduce/imagene/msms/results/tiny_imagene')
