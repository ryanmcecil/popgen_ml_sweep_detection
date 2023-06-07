"""
Performance comparison of ML models to Statistics
"""

<<<<<<< HEAD
import csv
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from models.retrieve_model import retrieve_model
from reproduce.article.code.configs import (ml_stat_comparison_model_configs,
                                            stat_comparison_configs,
                                            stat_comparison_conversion_config,
                                            stat_comparison_training_settings)
from reproduce.article.code.widths import retrieve_image_width_from_settings
from util.util import getGPU
=======
from models.retrieve_model import retrieve_model
import csv
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from reproduce.article.code.configs import ml_stat_comparison_model_configs, stat_comparison_configs, \
    stat_comparison_conversion_config, stat_comparison_training_settings
from util.util import getGPU
import pickle
from reproduce.article.code.widths import retrieve_image_width_from_settings
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83


def stat_comparison(model_sim_conversion_training_config,
                    base_dir: str,
<<<<<<< HEAD
                    generate_results: bool = True,
                    stat_configs=None):
=======
                    generate_results: bool=True):
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    """Plots model and statistic accuracies, and a correlation matrix between the prediction outputs. Saves
    the results in base_dir. Will do so for both the statistics trained on the model conversions and the raw
    data.
    """

    base_filename = os.path.join(base_dir, f'stat_comparison')
    csv_filename = f'{base_filename}_accs.csv'
    predictions_file = f'{base_filename}_predictions.pkl'

    if generate_results:

        getGPU()

<<<<<<< HEAD
        data_predictions, data_acc, data_labels = {}, {}, {}
=======
        data_predictions, data_acc = {}, {}
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

        # Get model performances
        for model_name, model_config in ml_stat_comparison_model_configs(width=retrieve_image_width_from_settings(model_sim_conversion_training_config)).items():
            print(f'Testing {model_name}')
            config = {
                'train': model_sim_conversion_training_config.copy(),
                'model': model_config.copy()
            }
            model = retrieve_model(config)(config.copy())
<<<<<<< HEAD
            data_predictions[model_name], data_labels[model_name], data_acc[model_name] = model.test(prediction_values=True,
                                                                                                     accuracy_value=True,
                                                                                                     classification_values=True,
                                                                                                     test_in_batches=True)
=======
            data_predictions[model_name], data_acc[model_name] = model.test(prediction_values=True,
                                                                            accuracy_value=True,
                                                                            test_in_batches=True)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

        # Check for pop data
        if 'pop' in model_sim_conversion_training_config['conversions'][0]:
            pop = model_sim_conversion_training_config['conversions'][0]['pop']
        else:
            pop = None

        # Get statistic performances
<<<<<<< HEAD
        if stat_configs is None:
            stat_configs = stat_comparison_configs()
        for stat_name, stat_config in stat_configs.items():
=======
        for stat_name, stat_config in stat_comparison_configs().items():
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            stat_config = {
                'train': {'simulations': model_sim_conversion_training_config['simulations'].copy(),
                          'conversions': stat_comparison_conversion_config(stat_name=stat_config['name'], pop=pop),
                          'training': stat_comparison_training_settings()},
                'model': stat_config.copy()
            }
            stat_model = retrieve_model(stat_config.copy())(stat_config.copy())
<<<<<<< HEAD
            data_predictions[stat_name], data_labels[stat_name], data_acc[stat_name] = stat_model.test(prediction_values=True,
                                                                                                       accuracy_value=True,
                                                                                                       classification_values=True,
                                                                                                       test_in_batches=True)
=======
            data_predictions[stat_name], data_acc[stat_name] = stat_model.test(prediction_values=True,
                                                                               accuracy_value=True,
                                                                               test_in_batches=True)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            print(data_acc)

            # Save csv file of accuracies
            with open(csv_filename, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(list(data_acc.keys()))
                writer.writerow(list(data_acc.values()))

            # Save predictions
            with open(predictions_file, 'wb') as f:
                pickle.dump(data_predictions, f)

    # Load csv file
    # with open(csv_filename, 'r') as csv_file:
    #     reader = csv.reader(csv_file)
    #     keys = next(reader)
    #     values = next(reader)
    #     data_acc = {key: value for key, value in zip(keys, values)}

    # load predictions file
    with open(predictions_file, 'rb') as f:
        data_predictions = pickle.load(f)

    # Save correlation matrix
    df = pd.DataFrame(data_predictions)
    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(method='spearman'), annot=True, fmt='.2f',
<<<<<<< HEAD
                cmap=plt.get_cmap('Blues'), cbar=True, ax=ax, vmin=0, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.title('Correlation Matrix')
    plt.savefig(f'{base_filename}_corr_matrix.png',
                bbox_inches='tight', pad_inches=0.0)

    # Save label comparison matrix
    method_pairs = list(combinations(data_labels, 2))
    write_to = f'{base_filename}_label_comparison.csv'
    with open(write_to, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model 1', 'Model 2', 'Percent Label Equal'])
        for model1, model2 in method_pairs:
            label_sim_acc = np.sum(1-np.abs(np.asarray(data_labels[model1]) - np.asarray(
                data_labels[model2]))) / len(data_labels[model1])
            writer.writerow([model1, model2, label_sim_acc])
=======
                cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.title('Correlation Matrix')
    plt.savefig(f'{base_filename}_corr_matrix.png', bbox_inches='tight', pad_inches=0.0)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
