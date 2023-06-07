from reproduce.thesis.visualize_r_imagene import r_imagene_model_config
from reproduce.thesis.arch_analysis import imagene_conversion_config, imagene_sim_config
from typing import Dict
import os
import tensorflow as tf
from models.retrieve_model import retrieve_model
from sklearn.metrics import roc_auc_score
import csv
from matplotlib import pyplot as plt
from util.util import getGPU


def training_settings(training_prop: float) -> Dict:
    """
    Returns
    -------
    Dict: Settings for training the Imagene model. This does not follow the original Imagene performance but
        is still able to achieve similar performance depending on the model initialization.
    """
    return {'epochs': 20,
            'batch_size': 64,
            'train_proportion': training_prop,
            'validate_proportion': 1 - training_prop - .1,
            'test_proportion': .1
            }


def imagene_model_config():
    return {
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


def test_model(training_prop: float,
               model_config: Dict):
    config = {
                'train': {'simulations': imagene_sim_config('0.01'),
                          'conversions': imagene_conversion_config('Rows'),
                          'training': training_settings(training_prop = training_prop)
                          },
                'model': model_config,
                'validate': False
                         }
    tf.keras.backend.clear_session()
    imagene = retrieve_model(config)(config)
    prediction_vals, label_vals, acc_val = imagene.test(prediction_values=True, label_values=True,
                                                        accuracy_value=True, test_in_batches=True)
    roc_auc = roc_auc_score(label_vals, prediction_vals)
    return acc_val, roc_auc



if __name__ == '__main__':
    getGPU()

    # Set up lists to store results
    r_imagene_acc_vals = []
    imagene_acc_vals = []
    r_imagene_auc_vals = []
    imagene_auc_vals = []

    # CSV file to save results to
    results_file = os.path.join(os.getcwd(), 'reproduce/thesis/results/sample_complexity.csv')

    training_samples_proportions = [.01, .02, .04, .08, .15, .4, .6, .8]
    training_samples_list = [prop*100000 for prop in training_samples_proportions]
    for training_prop in training_samples_proportions:
        print(training_prop)
        acc, auc = test_model(training_prop=training_prop,
                              model_config = imagene_model_config())
        imagene_acc_vals.append(acc)
        imagene_auc_vals.append(auc)
        print(acc)

        acc, auc = test_model(training_prop=training_prop,
                              model_config=r_imagene_model_config())
        r_imagene_acc_vals.append(acc)
        r_imagene_auc_vals.append(auc)
        print(acc)

    # Save results to csv file
    with open(results_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Training Samples', 'Imagene Acc', 'Imagene AUC', 'R-Imagene Acc', 'R-Imagene AUC'])
        for num_training_samples, i_acc, i_auc, i_r_acc, i_r_auc in list(zip(training_samples_list,
                                                                             imagene_acc_vals,
                                                                             imagene_auc_vals,
                                                                             r_imagene_acc_vals,
                                                                             r_imagene_auc_vals)):
            writer.writerow([num_training_samples, i_acc, i_auc, i_r_acc, i_r_auc])

    # Plot results
    plt.clf()
    plt.plot(training_samples_list, imagene_acc_vals, label='Imagene Accuracy')
    plt.plot(training_samples_list, r_imagene_acc_vals, label='R-Imagene Accuracy')
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/results/sample_complexity_acc_plot.png'))

    plt.clf()
    plt.plot(training_samples_list, imagene_auc_vals, label='Imagene AUC')
    plt.plot(training_samples_list, r_imagene_auc_vals, label='R-Imagene AUC')
    plt.xlabel('Training Dataset Size')
    plt.ylabel('AUC')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/results/sample_complexity_auc_plot.png'))



