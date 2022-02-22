import numpy as np
from tensorflow import keras
from typing import Dict, Tuple
import random
from util.popgen_data_class import PopGenDataClass
from simulate.popgen_simulators import retrieve_simulator
from process.popgen_processors import retrieve_processor


class DataGenerator(keras.utils.Sequence, PopGenDataClass):
    '''Generates training and test converted simulation data for Keras'''

    def __init__(self, config: Dict,
                 load_training_data: bool = True):
        """Initializes the data generator

        Parameters
        ----------
        config: (Dict) - Dictionary containing data configuration
        load_training_data: (bool) - If True, loads specified training data and splits it into train, validation,
            and test. If False, instead loads all specified data as test data.
        """
        super(DataGenerator, self).__init__()
        self.config = config
        self.load_training_data = load_training_data
        self.train, self.validate, self.test = self._prepare_data()

    def _convert_label(self, label: str) -> int:
        """Converts data label into numeric value

        Parameters
        ----------
        label: (str) - Label of data

        Returns
        -------
        (int) - Numeric value of label

        """
        if label == 'sweep':
            return 1
        else:
            return 0

    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns validation data and labels for validation testing during training

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: (validation data, labels)
        """
        x = [[] for file in self.validate[0][1]]
        y = []
        for label, files in self.validate:
            for i, file in enumerate(files):
                x[i].append(self.load_data(full_file_path=file))
            y.append(self._convert_label(label))

        for i in range(len(x)):
            x[i] = np.asarray(x[i])
            x[i] = x[i][:, :, :, np.newaxis]

        return np.asarray(x), np.asarray(y)

    def generate_test_set(self):
        """Generator for test set
        """
        for label, files in self.test:
            x = []
            for i, file in enumerate(files):
                x.append(self.load_data(full_file_path=file))
            for i in range(len(x)):
                x[i] = x[i][np.newaxis, :, :, np.newaxis]
            yield x, np.asarray([self._convert_label(label)])

    def _prepare_data(self):
        """Prepares training and/or testing data
        """

        if self.load_training_data:
            assert self.config['training']['train_proportion'] + \
                   self.config['training']['validate_proportion'] + \
                   self.config['training']['test_proportion'] == 1

        # Load data for each simulation
        train, test, validate = [], [], []
        for label, sim_config_list in self.config['simulations'].items():
            for sim_config in sim_config_list:
                simulator = retrieve_simulator(sim_config['software'])(sim_config,
                                                                       parallel=True,
                                                                       max_sub_processes=8)
                simulator.run_simulations()

                conversion_files = []
                for processor_config in self.config['conversions']:
                    processor = retrieve_processor(processor_config['conversion_type'])(config=processor_config,
                                                                                        simulator=simulator,
                                                                                        parallel=True,
                                                                                        max_sub_processes=8)
                    processor.run_conversions()
                    conversion_files.append(processor.get_filenames(datatype=processor.conversion_datatype(),
                                                                     n=simulator.config['N']))

                # Pass converted data to train, validate, test based on specified proportions
                data = zip(*conversion_files)
                if self.load_training_data:
                    train_num = int(simulator.config['N']*self.config['training']['train_proportion'])
                    validate_num = int(simulator.config['N']*self.config['training']['validate_proportion']) + train_num
                    for i, tuple_of_files in enumerate(data):
                        if i < train_num:
                            train.append((label, tuple_of_files))
                        elif i < validate_num:
                            validate.append((label, tuple_of_files))
                        else:
                            test.append((label, tuple_of_files))
                else:
                    for i, tuple_of_files in enumerate(data):
                        test.append((label, tuple_of_files))

        # Return data
        if self.load_training_data:
            random.shuffle(train)
            return train, validate, test
        else:
            return None, None, test

    def on_epoch_end(self):
        """Shuffle the data at the end of every epoch"""
        random.shuffle(self.train)

    def __len__(self):
        '''Returns number of batches per epoch'''
        return int(np.floor(len(self.train) / self.config['training']['batch_size']))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        data = self.train[index*self.config['training']['batch_size']:(index+1)*self.config['training']['batch_size']]

        x = [[] for file in data[0][1]]
        y = []
        for label, files in data:
            for i, file in enumerate(files):
                x[i].append(self.load_data(full_file_path=file))
            y.append(self._convert_label(label))

        for i in range(len(x)):
            x[i] = np.asarray(x[i])
            x[i] = x[i][:, :, :, np.newaxis]

        return x,np.asarray(y)
