"""
Sample complexity analysis for models and statistics
"""
import csv
import os
from copy import deepcopy

from matplotlib import pyplot as plt

from models.retrieve_model import retrieve_model
from reproduce.article.code.configs import (ml_stat_comparison_model_configs,
                                            stat_comparison_configs,
                                            stat_comparison_conversion_config,
                                            stat_comparison_training_settings)
from reproduce.article.code.widths import retrieve_image_width_from_settings
from util.util import getGPU


def add_to_accs_dict(data_accs,
                     key,
                     value):
    """Adds accuracy values to dictionary. If key does not exist in dictionary, creates new list of accuracy values

    Parameters
    ----------
    data_accs: (Dict) - Dictionary of accuracy values
    key: (str) - New key to be added to accuracy dictionary
    value: (float) - Accuracy value to be newly added
    """
    if key not in data_accs:
        data_accs[key] = []
    data_accs[key].append(value)


def get_train_sample_sizes(num_training_samples: int):
    """Returns list of dataset sample sizes to test to build sample complexity plots"""
    sample_sizes = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 100000]
    return [sample_size for sample_size in sample_sizes if sample_size <= num_training_samples]


def test_sample_complexity(sim_conversion_training_config, save_dir: str, generate_results: bool = True):

    # Get list of sample complexities
    num_training_samples = sim_conversion_training_config['simulations']['neutral'][0]['N'] + \
        sim_conversion_training_config['simulations']['sweep'][0]['N']
    train_sample_sizes = get_train_sample_sizes(num_training_samples)

    # Generate results and store in csv if specified
    csv_filename = os.path.join(save_dir, 'sample_complexity.csv')
    if generate_results:

        getGPU()

        data_accs = {}
        print('Beginning Sample Complexity Analysis')
        # For each sample complexity test models and statistics
        for train_sample_size in train_sample_sizes:
            print(f'Current dataset size {train_sample_size}')
            print('----------------------------------------------')

            # Set sample complexity
            copy_sim_conversion_training_config = deepcopy(
                sim_conversion_training_config)
            copy_sim_conversion_training_config['simulations']['neutral'][0]['N'] = train_sample_size // 2
            copy_sim_conversion_training_config['simulations']['sweep'][0]['N'] = train_sample_size // 2

            # train and test all models
            for model_name, model_config in ml_stat_comparison_model_configs(width=retrieve_image_width_from_settings(sim_conversion_training_config)).items():
                print(f'Testing {model_name}')
                config = {
                    # ensure train sample size is set correctly
                    'train': copy_sim_conversion_training_config.copy(),
                    # ensure test sample size is set correctly
                    'test': sim_conversion_training_config.copy(),
                    'model': model_config.copy()
                }
                model = retrieve_model(config)(config.copy())
                add_to_accs_dict(data_accs, model_name, model.test(
                    accuracy_value=True, test_in_batches=True)[0])

            # train and test all statistics
            # Check for pop data
            if 'pop' in sim_conversion_training_config['conversions'][0]:
                pop = sim_conversion_training_config['conversions'][0]['pop']
            else:
                pop = None

            # Get statistic performances
            for stat_name, stat_config in stat_comparison_configs().items():
                stat_config = {
                    'train': {'simulations': copy_sim_conversion_training_config['simulations'].copy(),
                              'conversions': stat_comparison_conversion_config(stat_name=stat_config['name'], pop=pop),
                              'training': stat_comparison_training_settings()},
                    'test': {'simulations': sim_conversion_training_config['simulations'].copy(),
                             'conversions': stat_comparison_conversion_config(stat_name=stat_config['name'], pop=pop),
                             'training': stat_comparison_training_settings()},
                    'model': stat_config.copy()
                }
                stat_model = retrieve_model(
                    stat_config.copy())(stat_config.copy())
                acc = stat_model.test(
                    accuracy_value=True, test_in_batches=True)[0]
                add_to_accs_dict(data_accs, stat_name, acc)

        # Convert sample complexity to actual training size
        sample_sizes = [int(sample_size*0.8)
                        for sample_size in train_sample_sizes]

        # store results in csv
        csv_filename = os.path.join(save_dir, 'sample_complexity.csv')
        with open(csv_filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Training Set Size'] +
                            [name for name in data_accs])
            for i, sample_size in enumerate(sample_sizes):
                writer.writerow([sample_size] + [accs[i]
                                for _, accs in data_accs.items()])

    # load csv file
    plt.clf()
    with open(csv_filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data_accs = {key: [] for key in next(reader)}
        for row in reader:
            for i, key in enumerate(data_accs):
                data_accs[key].append(float(row[i]))
        del data_accs['Training Set Size']

    plot_filename = os.path.join(save_dir, 'sample_complexity.png')
    actual_train_sample_sizes = [int(0.8 * sample_size)
                                 for sample_size in train_sample_sizes]
    for name, accs in data_accs.items():
        plt.plot(actual_train_sample_sizes, accs, label=name)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.grid()
    if 'msms' in sim_conversion_training_config['simulations']['neutral'][0]['software']:
        plt.axvline(x=int(0.8*num_training_samples), color='r',
                    label='Size Used', linestyle='dashed')
    else:
        plt.axvline(x=16_000, color='r', label='Size Used', linestyle='dashed')
    plt.legend()
    plt.savefig(plot_filename)
