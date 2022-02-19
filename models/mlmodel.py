from abc import ABC, abstractmethod
from models.model import SweepDetectionModel
from typing import Dict
import os
from generator.data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger


class MLModel(SweepDetectionModel, ABC):
    def __init__(self, config: Dict):
        super().__init__(config=config)

    @abstractmethod
    def _model(self):
        pass

    def train(self):
        loss_file = os.path.join(self.data_dir, 'loss_log.csv')
        csv_logger = CSVLogger(loss_file, append=True, separator=',')

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        data_generator = DataGenerator(self.config, load_training_data=True)
        self.model.fit(data_generator,epochs=self.config['training']['epochs'], verbose=1,
                        validation_data = data_generator.get_validation_data(),callbacks=[csv_logger])
        self.model.evaluate(data_generator.generate_test_set())
