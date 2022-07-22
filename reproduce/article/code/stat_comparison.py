"""
Performance comparison of ML models to Statistics
"""

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


def stat_comparison(model_sim_conversion_training_config,
                    base_dir: str,
                    generate_results: bool=True):
    """Plots model and statistic accuracies, and a correlation matrix between the prediction outputs. Saves
    the results in base_dir. Will do so for both the statistics trained on the model conversions and the raw
    data.
    """

    base_filename = os.path.join(base_dir, f'stat_comparison')
    csv_filename = f'{base_filename}_accs.csv'
    predictions_file = f'{base_filename}_predictions.pkl'

    if generate_results:

        getGPU()

        data_predictions, data_acc = {}, {}

        # Get model performances
        for model_name, model_config in ml_stat_comparison_model_configs(width=retrieve_image_width_from_settings(model_sim_conversion_training_config)).items():
            print(f'Testing {model_name}')
            config = {
                'train': model_sim_conversion_training_config.copy(),
                'model': model_config.copy()
            }
            model = retrieve_model(config)(config.copy())
            data_predictions[model_name], data_acc[model_name] = model.test(prediction_values=True,
                                                                            accuracy_value=True,
                                                                            test_in_batches=True)

        # Check for pop data
        if 'pop' in model_sim_conversion_training_config['conversions'][0]:
            pop = model_sim_conversion_training_config['conversions'][0]['pop']
        else:
            pop = None

        # Get statistic performances
        for stat_name, stat_config in stat_comparison_configs().items():
            stat_config = {
                'train': {'simulations': model_sim_conversion_training_config['simulations'].copy(),
                          'conversions': stat_comparison_conversion_config(stat_name=stat_config['name'], pop=pop),
                          'training': stat_comparison_training_settings()},
                'model': stat_config.copy()
            }
            stat_model = retrieve_model(stat_config.copy())(stat_config.copy())
            data_predictions[stat_name], data_acc[stat_name] = stat_model.test(prediction_values=True,
                                                                               accuracy_value=True,
                                                                               test_in_batches=True)
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
                cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.title('Correlation Matrix')
    plt.savefig(f'{base_filename}_corr_matrix.png', bbox_inches='tight', pad_inches=0.0)