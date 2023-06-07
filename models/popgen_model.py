import csv
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import tensorflow as tf
<<<<<<< HEAD
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam

from generator.data_generator import DataGenerator
from util.popgen_data_class import PopGenDataClass
=======
import csv
from math import isnan
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83


class PopGenModel(ABC, PopGenDataClass):
    """Abstract Class for models built to detect selective sweeps along portion of genome"""

    def __init__(self,
                 config: Dict,
                 root_dir: str = os.path.join(
                     os.getcwd(), 'models', 'saved_models'),
                 train_model: bool = True):
        """
        Parameters
        ----------
        config: (Dict) - Dictionary containing training settings, model settings, training
        settings, and potentially test settings
        root_dir: (str) - Location of root directory
        train_model: (bool) - If True, model will be trained.
        """
        super().__init__(config=config, root_dir=root_dir)
<<<<<<< HEAD
        print(self.data_dir)
=======
        #print(self.data_dir)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
        self.config = config.copy()
        self.train_model = train_model
        self.model = self._load_model()

    def _exclude_save_keys(self) -> List:
        """Model configuration does not depend on test set"""
        return ['test', 'validate']

    def _exclude_equality_test_keys(self) -> List:
        return ['test', 'validate']

    @abstractmethod
    def _model(self):
        """Returns model. Must implement standard functions such as predict on tensor object."""
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def _load(self, load_from_file=True):
=======
    def _load(self, load_from_file = True):
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
        """Loads the model"""
        raise NotImplementedError

    @abstractmethod
    def _classify(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _load_model(self):
        """Loads trained model, otherwise initializes the model and then trains it"""
        model, trained = self._load()
        if not trained:
            if self.train_model:
                model = self.train(model)
        return model

    def write_val_accs_to_file(self, accs):
        acc_csv = os.path.join(self.data_dir, 'validation_accs.csv')
        with open(acc_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Training', 'Validation Acc'])
            for i, acc in enumerate(accs):
                writer.writerow([i+1, acc])

<<<<<<< HEAD
    def reinitalize_ml_model(self):
        tf.keras.backend.clear_session()
        model, _ = self._load(load_from_file=False)
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

=======
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    def train(self, model):
        """Trains the model, saves it, and then returns it

        Parameters
        ----------
        model: Either an ML model or a statistic model
        """

<<<<<<< HEAD
        # Check settings if ML; if so, train using tensorflow
        data_generator = DataGenerator(self.config, load_training_data=True)
        if self.config['model']['type'] != 'statistic':

=======
        # Check settings if ML
        data_generator = DataGenerator(self.config, load_training_data=True)
        if self.config['model']['type'] != 'statistic':
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            val_x, val_y = data_generator.get_validation_data()
            num_trainings = 1
            if 'best_of' in self.config['train']['training']:
                num_trainings = self.config['train']['training']['best_of']
<<<<<<< HEAD
                print(
                    f'Training the model {num_trainings} times and using the best validation accuracy')
            best_acc = 0
            loss_file = os.path.join(self.base_dir, 'loss_log.csv')
            accs = []
            for i in range(num_trainings):  # Train the model num_trainings times

                # If number of trainings is 1, make sure we have better than 52% accuracy after first epoch, otherwise re-initialize model
                initial_epoch = 0
                if num_trainings == 1:
                    validation_acc = 0
                    count = 0
                    while validation_acc < .52 and count != 10:
                        count += 1
                        print(
                            'Re-initializing and checking to make sure that accuracy increases after first epoch')
                        model = self.reinitalize_ml_model()
                        csv_logger = CSVLogger(
                            loss_file, append=True, separator=',')
                        model.fit(data_generator, epochs=1, initial_epoch=0,
                                  verbose=1, callbacks=[csv_logger])
                        _, validation_acc = model.evaluate(val_x, val_y)
                    initial_epoch = 1
                else:
                    model = self.reinitalize_ml_model()
                    csv_logger = CSVLogger(
                        loss_file, append=True, separator=',')

                # Allow for early stopping if specified
                print('Training Model')
                if 'early_stopping' in self.config['train']['training']:
                    print('Using Early Stopping!')
                    model.fit(data_generator, epochs=self.config['train']['training']['epochs'], initial_epoch=initial_epoch, validation_data=(val_x, val_y),
                              verbose=1, callbacks=[csv_logger, EarlyStopping(patience=2)])
                else:
                    model.fit(data_generator, epochs=self.config['train']['training']['epochs'], initial_epoch=initial_epoch,
                              verbose=1, callbacks=[csv_logger])

                # Compute validation accuracy and take best model
=======
                print(f'Training the model {num_trainings} times and using the best validation accuracy')

            # Train the model
            best_acc = 0
            accs = []
            for i in range(num_trainings):
                tf.keras.backend.clear_session()
                model, _ = self._load(load_from_file=False)
                loss_file = os.path.join(self.base_dir, 'loss_log.csv')
                csv_logger = CSVLogger(loss_file, append=True, separator=',')
                model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                history = model.fit(data_generator, epochs=1, initial_epoch=0, verbose=1, callbacks=[csv_logger])
                model.fit(data_generator, epochs=self.config['train']['training']['epochs'], initial_epoch=1,
                          verbose=1, callbacks=[csv_logger])
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
                _, validation_acc = model.evaluate(val_x, val_y)
                accs.append(validation_acc)
                if validation_acc > best_acc:
                    best_acc = validation_acc
                    model.save(os.path.join(self.data_dir, 'model'))
            self.write_val_accs_to_file(accs)
            tf.keras.backend.clear_session()
            model, _ = self._load()
<<<<<<< HEAD

        else:  # call simple model fit if not an ML model
=======
        else:
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
            model.fit(data_generator)
            model.save(os.path.join(self.data_dir, 'model'))
        return model

    def _base_dir_surname(self) -> str:
        """Make all directories start with world model"""
        return 'model'

    def test(self,
             prediction_values: bool = False,
             classification_values: bool = False,
             label_values: bool = False,
             accuracy_value: bool = False,
             plot_cm: bool = False,
             ax=None,
             test_in_batches: bool = False,
             batch_size: int = 2048) -> List:
        """Tests the model on the test set. Then, returns the specified variables

        Parameters
        ----------
        prediction_values: (bool) - If True, returns model prediction values in return List
        classification_values: (bool) - If True, returns model classification values in return List
        label_values: (bool) - If True, returns test label values in return List
        accuracy_value: (bool) - If True, returns accuracy value in return List
        plot_cm: (bool) - If True, plots confusion matrix for model on ax
        ax: Matplotlib axis to plot confusion matrix
        test_in_batches: (bool) - If True, tests the data in batches (data must all be of the same size)
        batch_size: (int) - Size of batch to test at a time

        Returns
        -------
        List: List of values specified

        """
        outputs = []
        data_generator = DataGenerator(self.config, load_training_data=False)
        if prediction_values:
            predictions = []
        classifications = []
        labels = []
        if not test_in_batches or self.config['model']['type'] == 'statistic':
            generator = data_generator.generator('test', batch_size=1)
        else:
            generator = data_generator.generator('test', batch_size=batch_size)
        for x, y in generator:
            prediction = self.model.predict(x)
            if prediction.size != 0:
                classifications += list(self._classify(prediction))
                if prediction_values:
                    if not test_in_batches or self.config['model']['type'] == 'statistic':
                        prediction = [float(prediction)]
                    else:
                        prediction = list(np.squeeze(prediction))
                    predictions += prediction
                labels += list(y)
        if prediction_values:
            outputs.append(predictions)
        if classification_values:
            outputs.append(classifications)
        if label_values:
            outputs.append(labels)
        if accuracy_value:
            outputs.append(
                1 - np.mean(np.abs(np.asarray(classifications) - np.asarray(labels))))
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(labels,
                                                    classifications,
                                                    labels=[0, 1],
                                                    normalize='true',
                                                    cmap='Blues',
                                                    display_labels=[
                                                        'Neutral', 'Sweep'],
                                                    ax=ax, colorbar=False)
        return outputs
