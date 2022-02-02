from abc import ABC, abstractmethod
from typing import Dict, Generator
from util.util import standardize_and_save_data
from tensorflow.python.keras.engine import training


class MLModel(ABC):
    """Defines parent class for machine learning models."""

    def __init__(self, settings: Dict):
        self.settings = settings
        if self._saved_model_exists():
            self.model = self._load_saved_model()
            self.trained = True
        else:
            self.model = self._init_model()
            self.trained = False
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _name(self) -> str:
        """
        Returns name of CNN.
        """
        pass

    @abstractmethod
    def _init_model(self) -> training.Model:
        """
        Returns
        -------
        Keras model.
        """
        pass

    def _location_of_saved_model(self) -> str:
        """
        Returns location of saved model based on settings
        """
        raise NotImplementedError

    def _saved_model_exists(self):
        """
        Checks to see if model has already been trained and saved.
        """
        raise NotImplementedError

    def _load_saved_model(self):
        """
        Loads saved model from Keras file.
        """
        raise NotImplementedError

    def summary(self):
        """
        Prints Keras summary of model
        """
        print(self.model.summary())

