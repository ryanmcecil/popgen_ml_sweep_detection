"""
Visualizing learned model parameters
"""

from models.retrieve_model import retrieve_model
import os
import csv
from sklearn.metrics import roc_auc_score
from reproduce.article.code.configs import ml_stat_comparison_model_configs
import tensorflow as tf
from util.util import getGPU
from multiprocessing import Process
from reproduce.article.code.widths import retrieve_image_width_from_settings


def visualize(model_name,
              model_config,
              save_dir,
              model_sim_conversion_training_config,
              generate_results):
    """Explains model prediction using shapley values"""

    getGPU()
    tf.compat.v1.disable_eager_execution()

    out_dir = os.path.join(save_dir, f'{model_name}_visualization/')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, f'acc.csv')
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model', 'Accuracy', 'AUC'])
        config = {
            'train': model_sim_conversion_training_config,
            'model': model_config
        }
        imagene = retrieve_model(config)(config)
        prediction_vals, label_vals, acc_val = imagene.test(prediction_values=True, label_values=True,
                                                            accuracy_value=True, test_in_batches=True)
        roc_auc = roc_auc_score(label_vals, prediction_vals)
        writer.writerow([model_name, f'{acc_val:.3f}', f'{roc_auc:.3f}'])
        imagene.apply_shap(save_dir=out_dir, generate_results=generate_results)


def visualize_models(model_sim_conversion_training_config,
                     save_dir: str,
                     generate_results: bool=True):
    """Visualizes the ML models given sim, conversion, and training settings, and a save directory"""
    for model_name, model_config in ml_stat_comparison_model_configs(width=retrieve_image_width_from_settings(model_sim_conversion_training_config)).items():

        if model_name != 'DeepSet':

            # Run Shap visualization
            p = Process(target=visualize, args=(model_name, model_config, save_dir,
                                                model_sim_conversion_training_config, generate_results,))
            p.start()
            p.join()
            p.terminate()

            # Run layer and weights visualization for R-Imagene
            if model_name == 'R-Imagene':
                getGPU()
                out_dir = os.path.join(save_dir, f'{model_name}_visualization/')
                config = {
                    'train': model_sim_conversion_training_config,
                    'model': model_config
                }
                imagene = retrieve_model(config)(config)
                imagene.visualize_layer_outputs(out_dir, 1)
                imagene.visualize_parameters(out_dir)
                tf.keras.backend.clear_session()
