from abc import ABC, abstractmethod
from typing import Dict, Generator, Union, Iterable
import os
import random
import yaml
import numpy as np
from scipy import sparse
import multiprocessing
import glob

# Seed generated using os.urandom
# Defines seeds for all simulations dependant on number of sim directory
random.seed(434935399793924222151)


class Simulator(ABC):
    """Defines abstract class for simulators."""

    def __init__(self,
                 config: Dict,
                 verbose=False,
                 parallel=True,
                 max_sub_processes: int = 10):
        """Initializes simulator class with the simulation configuration

        Parameters
        ----------
        config: (Dict) - Dictionary containing the configuration for the simulation.
        verbose: (bool) - If true, print out steps of simulation process
        parallel: (bool) - If True, run parallel simulations to speed up simulation time
        max_sub_processes: (int) - Limit on maximum number of parallel subprocesses to run

        """

        # Initialize configuration
        self.config = config
        self.verbose = verbose
        self.parallel = parallel
        self.max_sub_processes = max_sub_processes
        self.template_settings = self._retrieve_base_template_settings()
        self.seeds = None

        # Initialize locations of directories to store data and template sim files
        self.sim_dir = None
        self.tmp_simulation_folder = f'{os.getcwd()}/simulators/tmp'
        if not os.path.isdir(self.tmp_simulation_folder):
            os.mkdir(self.tmp_simulation_folder)
        self.data_dir = f'{os.getcwd()}/simulators/data'
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        if "template" in self.config:
            file = f'{os.getcwd()}/simulators/{self.config["software"]}/settings/{self.config["template"]}'
            if os.path.isfile(file):
                self.template_file = file
            else:
                raise FileNotFoundError

        # Setup print statements
        global log
        if self.verbose:
            def log(x): print(x)
        else:
            def log(x): pass
        self.log = log
        self.log(f'Simulation object created with configuration: {self.config}')

    @abstractmethod
    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for template that do not change as dict.
        """
        pass

    @abstractmethod
    def _load_sim_data(self,
                       file: str) -> [np.ndarray, np.ndarray]:
        """Loads the simulation data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulation file

        Returns
        -------
        [np.ndarray, np.ndarray]: Genetic image and the loci positions as np arrays
        """
        pass

    @abstractmethod
    def _simulate(self,
                  id: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id: (int) - ID of current simulation

        """
        pass

    def _make_template_settings_file(self,
                                     template_settings: Dict,
                                     id: int) -> str:
        """Create temporary settings file for simulating the images

        Parameters
        ----------
        template_settings: (Dict) - Settings to be inserted into simulation template file.
        id: (int) - ID number of simulation to name the saved file.
        """
        template_file = self.config["template"]
        name, ext = template_file.split('.')
        tmp_template_file = f'{self.tmp_simulation_folder}/tmp_{id:09d}.{ext}'
        with open(self.template_file, 'r') as template_file:
            with open(tmp_template_file, 'w') as tmp_file:
                for line in template_file.readlines():
                    for setting_name, setting in template_settings.items():
                        line = line.replace(setting_name, setting)
                    tmp_file.write(line)
        self.log(f'Updated sim temporary settings file with settings: {template_settings}')
        return tmp_template_file

    def _erase_simulated_images(self):
        """
        Erases images that were originally simulated.
        """
        self.log('Erasing Simulated images')
        for f in os.listdir(self.tmp_simulation_folder):
            os.remove(os.path.join(self.tmp_simulation_folder, f))
            self.log(f)

    def _write_config_to_yaml_file(self,
                                   directory: str):
        """Writes simulation config to yaml file in directory, except for number of images we have already simulated.

        Parameters
        ----------
        directory: (str) - Directory that simulation config will be written to
        """
        with open(os.path.join(directory, 'config.yaml'), 'w') as config_file:
            num_sims = self.config.pop('n')
            yaml.dump(self.config, config_file)
            self.log(f'Writing config to yaml file: {self.config}')
            self.config['n'] = num_sims

    def _read_config_from_yaml_file(self,
                                    directory: str) -> Dict:
        """Reads simulation config from a yaml file

        Parameters
        ----------
        directory: (str) - Directory in which config.yaml is stored

        Returns
        -------
        settings: (Dict) - Returns a dict of the simulation config contained in the yaml file

        """
        with open(os.path.join(directory, 'config.yaml'), 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.log(f'Reading config from yaml file: {config}')
        return config

    def _create_data_directory(self):
        """Creates directory to store simulated data for this setting.
        Stores location of new directory in self.sim_dir"""
        new_dir = f'{self.data_dir}/sim_{len(os.listdir(self.data_dir)):04d}'
        # Get random base seed based on sim directory number
        self.config['base_seed'] = random.sample(range(100000000), len(os.listdir(self.data_dir)) + 1)[-1]
        os.mkdir(new_dir)
        self.sim_dir = new_dir
        self.config['n'] = 0
        self._write_config_to_yaml_file(new_dir)
        self.log(f'Created data directory: {new_dir}')

    def _count_num_images_in_sim_dir(self) -> int:
        """Reads simulation config from a yaml file

        Returns
        -------
        int: Number of image files that have already been simulated; must check for breaks in IDs due
            to parallel processing.

        """
        for i in range(self.config['N']):
            id = i+1
            if not os.path.isfile(f'{self.sim_dir}/{id:09d}.npz'):
                return i
        return self.config['N']

    def _search_for_data_directory(self):
        """Searches for data directory with this simulation config. If found, stores it in self.sim_dir.
        """
        self.log('Searching for data directory')
        for sim_dir in os.listdir(self.data_dir):
            config = self._read_config_from_yaml_file(os.path.join(self.data_dir, sim_dir))
            is_config = True
            for key in config:
                if key != 'N' and key != 'base_seed':
                    if key not in self.config:
                        is_config = False
                        break
                    if config[key] != self.config[key]:
                        is_config = False
                        break
            if is_config:
                self.sim_dir = os.path.join(self.data_dir, sim_dir)
                self.config['n'] = self._count_num_images_in_sim_dir()
                self.config['base_seed'] = config['base_seed']
                self.log(f'Data directory found: {self.sim_dir}')

    #
    # def _compress_data_dir(self):
    #     """Compresses data directory into as many single objects as possible"""
    #     # Add based on os.getmtime
    #    # list_of_files = sorted(list_of_files,
    #    #                        key=os.path.getmtime)
    #     # Iteratively add files to 3d matrix
    #     # Check for memory constraints
    #     raise NotImplementedError
    #
    # def _clean_data_dir(self):
    #     """Cleans data directory of all extra files"""
    #     # Remove simulated files
    #     raise NotImplementedError

    def _sparsify_and_save_data(self,
                                image: np.ndarray,
                                positions: np.ndarray,
                                id: int):
        """Standardizes and saves single image

        Parameters
        ----------
        image: (np.ndarray) - Image to be saved as sparsified matrix
        positions: (np.ndarray) - Positions of loci to be saved as npy file
        id: (int) - ID of current simulation

        """
        sparse_matrix = sparse.csr_matrix(image)
        sparse.save_npz(f'{self.sim_dir}/{id:09d}.npz', sparse_matrix, compressed=True)
        np.save(f'{self.sim_dir}/{id:09d}.npy', positions)

    def _simulate_and_save(self,
                           i: int):
        """Simulates the genetic data using appropriate software and then saves it as a compressed image

        Parameters
        ----------
        i: (int) - Current value iteration of simulations

        """
        print(f'Running simulation {i + 1}')
        id = i + 1
        image, positions = self._load_sim_data(self._simulate(id))
        self._sparsify_and_save_data(image, positions, id)
        os.remove(f'{self.tmp_simulation_folder}/{id:09d}.txt')

    def _run_simulations(self):
        """
        Calls appropriate file using bash commands to simulate images
        """
        self.log('Running Simulations')
        random.seed(self.config['base_seed'])
        self.seeds = random.sample(range(100000000), self.config['N'])
        self.log(f'First few seeds generated: {self.seeds[:10]}')
        if self.parallel:
            pool = multiprocessing.Pool(self.max_sub_processes)
            pool.map(self._simulate_and_save, range(self.config['n'], self.config['N']))
        else:
            for i in range(self.config['n'], self.config['N']):
                self._simulate_and_save(i)
        # self._compress_data_dir()
        # self._clean_data_dir()

    def simulate(self) -> str:
        """ Simulates the data if it has not already been simulated

        Returns
        -------
        str: - Location of simulated images
        """
        self._search_for_data_directory()
        if self.sim_dir is None:
            self.log('Data directory not found')
            self._create_data_directory()
        if self.config['n'] < self.config['N']:
            self._write_config_to_yaml_file(self.sim_dir)
            self._run_simulations()
        self.log('Simulation completed')
        return self.sim_dir
