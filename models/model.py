from abc import ABC, abstractmethod
from typing import Dict
import os
from util.popgen_data_class import PopGenDataClass



class SweepDetectionModel(ABC, PopGenDataClass):
    def __init__(self,
                 config: Dict):
        super().__init__(config=config)
        self.config = config
        self.model = self._model()
        self.train()

    @abstractmethod
    def _model(self):
        """Returns model"""
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _root_dir_name(self) -> str:
        return os.path.join(f'{os.getcwd()}', 'models', 'saved_models')

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError