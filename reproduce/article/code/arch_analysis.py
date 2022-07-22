import os, csv
import tensorflow as tf
from models.retrieve_model import retrieve_model
from sklearn.metrics import roc_auc_score
from reproduce.article.code.configs import imagene_arch_analysis_configs
from util.util import getGPU
from reproduce.article.code.widths import retrieve_image_width_from_settings


def test_architectures(sim_conversion_training_config,
                       save_dir: str,
                       generate_results: bool=True):
    """Tests Imagene reduced architectures given sim, conversion, and training configs"""

    getGPU()

    if generate_results:

        results_file = os.path.join(save_dir, 'arch_performance_analysis.csv')

        with open(results_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Architecture', 'Accuracy', 'AUC', 'Parameters'])

            for name, arch in imagene_arch_analysis_configs(width=retrieve_image_width_from_settings(sim_conversion_training_config)).items():
                config = {
                    'train': sim_conversion_training_config,
                    'model': arch
                }
                tf.keras.backend.clear_session()
                imagene = retrieve_model(config)(config)
                print(imagene.model.summary())
                prediction_vals, label_vals, acc_val = imagene.test(prediction_values=True, label_values=True,
                                                                    accuracy_value=True, test_in_batches=True)
                num_parameters = imagene.number_of_parameters()
                roc_auc = roc_auc_score(label_vals, prediction_vals)
                print(acc_val)
                print(roc_auc)
                writer.writerow([name, f'{acc_val:.3f}', f'{roc_auc:.3f}', num_parameters])
