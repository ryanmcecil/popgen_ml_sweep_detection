from abc import ABC, abstractmethod
from typing import Dict, List
import os
from util.popgen_data_class import PopGenDataClass
from generator.data_generator import DataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


class PopGenModel(ABC, PopGenDataClass):
    """Abstract Class for models built to detect selective sweeps along portion of genome"""

    def __init__(self,
                 config: Dict,
                 root_dir: str = os.path.join(os.getcwd(), 'models', 'saved_models'),
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
        self.config = config
        self.train_model = train_model
        self.model = self._load_model()

    def _exclude_save_keys(self) -> List:
        """Model configuration does not depend on test set"""
        return ['test']

    @abstractmethod
    def _model(self):
        """Returns model. Must implement standard functions such as predict on tensor object."""
        raise NotImplementedError

    @abstractmethod
    def _load(self):
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

    def train(self, model):
        """Trains the model, saves it, and then returns it

        Parameters
        ----------
        model: Either an ML model or a statistic model
        """
        data_generator = DataGenerator(self.config, load_training_data=True)
        if self.config['model']['type'] != 'statistic':
            loss_file = os.path.join(self.base_dir, 'loss_log.csv')
            csv_logger = CSVLogger(loss_file, append=True, separator=',')
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(data_generator, epochs=self.config['train']['training']['epochs'], verbose=1,
                      validation_data=data_generator.get_validation_data(), callbacks=[csv_logger])
        else:
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
        if test_in_batches:
            generator = data_generator.generator('test', batch_size=batch_size)
        else:
            generator = data_generator.generator('test', batch_size=1)
        for x, y in generator:
            prediction = self.model.predict(x)
            if prediction_values:
                predictions += list(np.squeeze(prediction))
            classifications += list(self._classify(prediction))
            labels += list(y)
        if prediction_values:
            outputs.append(predictions)
        if classification_values:
            outputs.append(classifications)
        if label_values:
            outputs.append(labels)
        if accuracy_value:
            outputs.append(1 - np.mean(np.abs(np.asarray(classifications) - np.asarray(labels))))
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(labels,
                                                    classifications,
                                                    labels=[0, 1],
                                                    normalize='true',
                                                    cmap='Blues',
                                                    display_labels=['Neutral', 'Sweep'],
                                                    ax=ax, colorbar=False)
        return outputs
