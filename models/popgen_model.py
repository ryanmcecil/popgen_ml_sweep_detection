from abc import ABC, abstractmethod
from typing import Dict, List, Iterable
import os
from util.popgen_data_class import PopGenDataClass
from generator.data_generator import DataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



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
        train: (bool) - If True, model will be trained.
        """
        super().__init__(config=config, root_dir=root_dir)
        self.config = config
        self.train_model = train_model
        self.model = self._load_model()

    def _exclude_save_keys(self) -> List:
        return ['test']

    @abstractmethod
    def _model(self):
        """Returns model. Must implement standard functions such as predict on tensor object."""
        raise NotImplementedError

    def _load_model(self):
        """Loads trained model, otherwise initializes the model and then trains it"""
        model = self._model()
        if os.path.exists(os.path.join(self.data_dir, 'model')):
            model.load(os.path.join(self.data_dir, 'model'))
        else:
            if self.train_model:
                model = self.train(model)
        return model

    def train(self, model):
        """Trains the inputted model, saves it, and then returns it

        Parameters
        ----------
        model: Either an ML model or a statistic model

        """
        loss_file = os.path.join(self.data_dir, 'loss_log.csv')
        csv_logger = CSVLogger(loss_file, append=True, separator=',')

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        data_generator = DataGenerator(self.config, load_training_data=True)
        model.fit(data_generator,epochs=self.config['train']['training']['epochs'], verbose=1,
                        validation_data = data_generator.get_validation_data(),callbacks=[csv_logger])
        model.save(os.path.join(self.data_dir), 'model')
        return model

    def _base_dir_surname(self) -> str:
        return 'model'

    def _test_predictions_and_labels(self) -> Iterable:
        """"""
        data_generator = DataGenerator(self.config, load_training_data=False)
        classifications = []
        labels = []
        for x,y in data_generator.generator('test'):
            classifications.append(self.model.classify(x))
            labels.append(y)
        return classifications, labels


    def test_and_plot_cnf(self) -> ConfusionMatrixDisplay:
        y_pred, y_true = self._test_predictions_and_labels()
        cmd = ConfusionMatrixDisplay.from_predictions(y_true,
                                                      y_pred,
                                                      labels=['neutral', 'sweep'],
                                                      normalize='true')
        return cmd


