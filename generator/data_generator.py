import random
from typing import Dict, List, Tuple

import numpy as np
from tensorflow import keras

from process.popgen_processors import retrieve_processor
from simulate.popgen_simulators import retrieve_simulator
from util.popgen_data_class import PopGenDataClass


class DataGenerator(keras.utils.Sequence, PopGenDataClass):
    """Generates training and test converted simulation data for Keras"""

    def __init__(self, config: Dict,
                 load_training_data: bool = True,
                 retain_sim_location: bool = False):
        """Initializes the data generator

        Parameters
        ----------
        config: (Dict) - Dictionary containing data configuration
        load_training_data: (bool) - If True, loads specified training data and splits it into train, validation,
            and test. If False, if config['test'] is defined, instead loads all specified data as test data.
        retain_sim_location: (bool) - If True, all loaded data 'remembers' the simulator used to simulate it, and hence,
            its original location
        """
        super(DataGenerator, self).__init__()
        self.config = config
        self.retain_sim_location = retain_sim_location
        self.load_training_data = load_training_data
        self.train, self.validate, self.test = self._prepare_data()

    @staticmethod
    def _convert_label(label: str) -> int:
        """Converts data label into numeric value

        Parameters
        ----------
        label: (str) - Label of data

        Returns
        -------
        (int) - Numeric value of label

        """
        if label == 'sweep' or label == 'hard sweep':
            return 1
        elif label == 'neutral' or label == 'soft sweep':
            return 0
        else:
            raise NotImplementedError("Label Unknown")

    def generator(self,
                  dataset: str = 'test',
                  batch_size: int = 1):
        """Generates the specified dataset

        Parameters
        ----------
        dataset: (str) - Name of dataset, either 'train', 'validate', or 'test'
        batch_size: (int) - Size of batch in tensor form to be returned
        """
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'validate':
            dataset = self.validate
        elif dataset == 'test':
            dataset = self.test
        else:
            raise Exception(f'Dataset {dataset} does not exist.')
        count = 0
        while count + batch_size < len(dataset):
            yield self._get_data_pairs(dataset[count:count + batch_size])
            count += batch_size
        if count < len(dataset):
            yield self._get_data_pairs(dataset[count:])

    def _get_data_pairs(self,
                        data: List[str]) -> Tuple:
        """Retrieves batch of (x,y) pairs from data

        Parameters
        ----------
        data: (List[str, Tuple[str]]) - Data contain labels and filenames

        Returns
        -------
        Iterable: Batch of (x, y) or if retain_sim_location is set to true, batch of (x ,y, filenames, simulators)
        """

        if len(data) == 0:
            return None

        # Data is of form (label, tuple_of_files)
        y = [self._convert_label(label) for label, _ in data]
        y = np.asarray(y)

        if self.retain_sim_location:
            x = [[self.load_data(full_file_path=tuple_of_files[i][0], as_tensor=True) for _, tuple_of_files in data
                  ] for i in range(len(data[0][1]))]
            x = [np.concatenate(x_list) for x_list in x]

            filenames = [[tuple_of_files[i][0] for _, tuple_of_files in data
                          ] for i in range(len(data[0][1]))]
            sims = [[tuple_of_files[i][1] for _, tuple_of_files in data
                     ] for i in range(len(data[0][1]))]

            return x, y, filenames, sims

        else:
            x = [[self.load_data(full_file_path=tuple_of_files[i], as_tensor=True) for _, tuple_of_files in data
                  ] for i in range(len(data[0][1]))]
            x = [np.concatenate(x_list) for x_list in x]

            return x, y

    def get_validation_data(self) -> Tuple:
        """Returns validation data and labels for validation testing during training

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: (validation data, labels)
        """
        return self._get_data_pairs(self.validate)

    def __getitem__(self, index):
        """Generate one batch of data"""
        data = self.train[
            index * self.config['train']['training']['batch_size']:(index + 1) * self.config['train']['training'][
                'batch_size']]
        return self._get_data_pairs(data)

    def _prepare_data(self):
        """Prepares training and/or testing data"""

        if self.load_training_data or 'test' not in self.config:
            loc = 'train'
        else:
            loc = 'test'

        assert self.config[loc]['training']['train_proportion'] + \
            self.config[loc]['training']['validate_proportion'] + \
            self.config[loc]['training']['test_proportion'] == 1

        # Load data for each simulation
        train, test, validate = [], [], []
        for label, sim_config_list in self.config[loc]['simulations'].items():
            for sim_config in sim_config_list:
                simulator = retrieve_simulator(sim_config['software'])(sim_config,
                                                                       parallel=True,
                                                                       max_sub_processes=8)
                simulator.run_simulations()

                conversion_files = []
                for processor_config in self.config[loc]['conversions']:
                    processor = retrieve_processor(processor_config['conversion_type'])(config=processor_config,
                                                                                        simulator=simulator,
                                                                                        parallel=True,
                                                                                        max_sub_processes=8)
                    processor.run_conversions()
                    if self.retain_sim_location:
                        files = [(file, simulator) for file in
                                 processor.get_filenames(datatype=processor.conversion_datatype(),
                                                         n=simulator.config['N'])]
                        conversion_files.append(files)
                    else:
                        conversion_files.append(processor.get_filenames(datatype=processor.conversion_datatype(),
                                                                        n=simulator.config['N']))

                # Pass converted data to train, validate, test based on specified proportions
                data = zip(*conversion_files)
                train_num = int(
                    simulator.config['N'] * self.config[loc]['training']['train_proportion'])
                validate_num = int(
                    simulator.config['N'] * self.config[loc]['training']['validate_proportion']) + train_num
                for i, tuple_of_files in enumerate(data):
                    if i < train_num:
                        train.append((label, tuple_of_files))
                    elif i < validate_num:
                        validate.append((label, tuple_of_files))
                    else:
                        test.append((label, tuple_of_files))

        # Return data
        random.shuffle(train)
        return train, validate, test

    def on_epoch_end(self):
        """Shuffle the data at the end of every epoch"""
        random.shuffle(self.train)

    def __len__(self):
        """Returns number of batches per epoch"""
        return int(np.floor(len(self.train) / self.config['train']['training']['batch_size']))
