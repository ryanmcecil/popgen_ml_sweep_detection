from abc import ABC, abstractmethod
from typing import Dict
from util.popgen_data_class import PopGenDataClass
from simulate.popgen_simulator import PopGenSimulator
import os
import numpy as np
import multiprocessing


class PopGenProcessor(PopGenDataClass, ABC):
    """Defines class for processing genetic data"""

    def __init__(self,
                 config: Dict,
                 simulator: PopGenSimulator,
                 root_dir: str = os.path.join(os.getcwd(), 'process', 'conversions'),
                 parallel=True,
                 max_sub_processes: int = 10):
        """Initializes processor class with configuration

        Parameters
        ----------
        config: (Dict) - Dictionary containing the configuration
        simulator: (Simulator) - Simulator that simulated the genetic data
        root_dir: (str) - Location of root directory
        parallel: (bool) - If True, run parallel processing to speed up simulate time
        max_sub_processes: (int) - Limit on maximum number of parallel subprocesses to run

        """

        self.simulator = simulator
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        root_dir = self.simulator._find_or_make_base_directory(root_dir)

        super().__init__(config=config, root_dir=root_dir)

        self.parallel = parallel
        self.max_sub_processes = max_sub_processes
        self._check_base_settings()

    def _base_dir_surname(self) -> str:
        """Surname of base directories"""
        return 'conv'

    @abstractmethod
    def _check_base_settings(self) -> Dict:
        """Checks that config contains appropriate settings"""
        raise NotImplementedError

    @abstractmethod
    def conversion_datatype(self) -> str:
        """Returns datatype of process"""
        raise NotImplementedError

    @abstractmethod
    def _convert(self, data: np.ndarray) -> np.ndarray:
        """Converts numpy data"""
        raise NotImplementedError

    def _convert_and_save(self, id_num: int):
        """Converts and saves the simulated data

        Parameters
        ----------
        id_num: (int) - ID number of data
        """
        print(f'Running Conversion {id_num}')
        datatype = self.conversion_datatype()
        data = self.simulator.load_data(id_num=id_num, datatype=datatype)
        data = self._convert(data)
        if datatype == 'popgen_image':
            if id_num == 1:
                self.plot_example_image(data)
        self.save_data(data, id_num, datatype)

    def run_conversions(self):
        """If data have not already been converted, run processing"""
        last_saved_id = self._last_saved_id(self.conversion_datatype())

        if last_saved_id < self.simulator.config['N']:
            if last_saved_id < self.simulator.config['N']:
                print('Running Conversions')
                if self.parallel:
                    pool = multiprocessing.Pool(self.max_sub_processes)
                    pool.map(self._convert_and_save, range(last_saved_id + 1, self.simulator.config['N'] + 1))
                else:
                    for i in range(last_saved_id + 1, self.simulator.config['N'] + 1):
                        self._convert_and_save(i)
            else:
                print('Simulations have already been processed')



