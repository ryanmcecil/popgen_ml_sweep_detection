import numpy as np
from tensorflow import keras
import os
import glob
from PIL import Image
from typing import Dict, List
import random
import sys
from util.popgen_data_class import PopGenDataClass
from simulators.simulators import retrieve_simulator
from processors.popgen_processors import retrieve_processor


class DataGenerator(keras.utils.Sequence, PopGenDataClass):
    '''Generates training and test data for Keras'''

    def __init__(self, config: Dict, load_training_data: bool = True):
        self.config = config
        self.load_training_data = load_training_data
        self.train, self.validate, self.test = self._prepare_data()

    def __len__(self):
        '''Returns number of batches per epoch'''
        return int(np.floor(len(self.train) / self.config['training']['batch_size']))

    def on_epoch_end(self):
        random.shuffle(self.train)

    def _convert_label(self, label):
        if label == 'sweep':
            return 1
        else:
            return 0

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

    def get_validation_data(self):
        x = [[] for file in self.validate[0][1]]
        y = []
        for label, files in self.validate:
            for i, file in enumerate(files):
                x[i].append(self.load_data(full_file_path=file))
            y.append(self._convert_label(label))

        for i in range(len(x)):
            x[i] = np.asarray(x[i])
            x[i] = x[i][:, :, :, np.newaxis]

        return x, np.asarray(y)

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
        train = []
        test = []
        validate = []
        for label, sim_config_list in self.config['simulations'].items():
            for sim_config in sim_config_list:
                simulator = retrieve_simulator(sim_config['software'])(sim_config, verbose_level=2,
                                                                       parallel=True,
                                                                       max_sub_processes=8)
                simulator.run_simulations()

                conversion_files = []
                for processor_config in self.config['conversions']:
                    processor = retrieve_processor(processor_config['conversion_type'])(config=processor_config,
                                                                                        simulator=simulator,
                                                                                        verbose_level=1,
                                                                                        parallel=True,
                                                                                        max_sub_processes=8)
                    processor.run_conversions()
                    conversion_files.append(processor.get_filenames(datatype=processor.conversion_datatype(),
                                                                     n=simulator.config['N']))
                data = []
                for j in range(len(conversion_files[0])):
                    entries = []
                    for i in range(len(conversion_files)):
                        entries.append(conversion_files[i][j])
                    data.append(tuple(entries))
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
        if self.load_training_data:
            random.shuffle(train)
            return train, validate, test
        else:
            return None, None, test
