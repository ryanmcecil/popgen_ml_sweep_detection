import multiprocessing
import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List

from simulate.software import settings
from util.popgen_data_class import PopGenDataClass

try:
    import importlib.resources as ilresources
except ImportError:
    try:
        import importlib_resources as ilresources
    except ImportError:
        raise ImportError(
            'Must install backport of importlib_resources if not using Python >= 3.7')
import math

# Seed generated using os.urandom
# Defines seeds for all simulations dependant on number of sim directory
random.seed(434935399793924222151)


class PopGenSimulator(PopGenDataClass, ABC):
    """Defines abstract class for simulate."""

    def __init__(self,
                 config: Dict,
                 root_dir: str = os.path.join(
                     os.getcwd(), 'simulate', 'simulations'),
                 parallel=True,
                 max_sub_processes: int = 10):
        """Initializes simulator class with the simulate configuration

        Parameters
        ----------
        config: (Dict) - Dictionary containing the configuration for the simulate.
        root_dir: (str) - Location of root directory
        parallel: (bool) - If True, run parallel simulations to speed up simulate time
        max_sub_processes: (int) - Limit on maximum number of parallel subprocesses to run

        """
        super().__init__(config=config,
                         root_dir=root_dir)

        self.parallel = parallel
        self.max_sub_processes = max_sub_processes

        # Simulation attributes
        self.tmp_simulation_folder = os.path.join(
            os.getcwd(), 'simulate', 'tmp')
        if not os.path.isdir(self.tmp_simulation_folder):
            os.mkdir(self.tmp_simulation_folder)
        self.template_settings = self._retrieve_base_template_settings()

        self.image_datatypes = []
        self.pos_datatypes = []
        if 'template' in self.config and 'schaffner' in self.config['template']:
            for k in [1, 2, 3]:
                self.image_datatypes.append(f'popgen_pop_image{k}')
                self.pos_datatypes.append(f'popgen_pop_positions{k}')
        else:
            self.image_datatypes.append('popgen_image')
            self.pos_datatypes.append('popgen_image')

        self.seeds = None
        sim_num = int(re.findall(
            f".*{self._base_dir_surname()}_(.*)", self.base_dir)[0])
        self.config['base_seed'] = random.sample(
            range(100000000),  len(os.listdir(self.root_dir)))[sim_num]

    @abstractmethod
    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for template that do not change as dict.
        """
        pass

    @abstractmethod
    def _load_and_save_sim_data(self,
                                file: str,
                                id_num: int):
        """Loads the simulate data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulate file
        """
        pass

    @abstractmethod
    def _simulate(self,
                  id_num: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id_num: (int) - ID of current simulate

        """
        pass

    def retrieve_max_or_min_width(self,
                                  pop: int = None,
                                  get_max: bool = True) -> int:
        """Returns the maximum or minimum width across all simulated images

        Parameters
        ----------
        pop: (int) - If None, the returned maximum width will be computed across all simulated images. If specified,
        returned maximum width will be across single population simulated.

        Returns
        -------
        int: Maximum simulation window width

        """

        if get_max:
            start_filename = 'max_width'
        else:
            start_filename = 'min_width'

        # Create max width directory to store max width values
        max_width_dir = os.path.join(self.base_dir, '')
        if not os.path.isdir(max_width_dir):
            os.mkdir(max_width_dir)

        # Check if we have already found max width
        max_width_file = os.path.join(
            max_width_dir, f'{start_filename}_{self.config["N"]}.txt')
        if os.path.isfile(max_width_file):
            with open(max_width_file, 'r') as f:
                max_width = int(f.readline())

        else:
            def roundup(x):
                """Rounds x to nearest 10th digit
                """
                return int(math.ceil(x / 10.0)) * 10

            def rounddown(x):
                return int(math.floor(x) / 10.0) * 10

            # Find maximum width if not already found
            widths = []
            if pop is None:
                for datatype in self.image_datatypes:
                    for id_num in range(1, self.config['N']+1):
                        widths.append(self.load_data(
                            id_num=id_num, datatype=datatype).shape[1])
            else:
                datatype = self.image_datatypes[pop-1]
                for id_num in range(1, self.config['N'] + 1):
                    widths.append(self.load_data(
                        id_num=id_num, datatype=datatype).shape[1])

            # Round up max width to nearest 10th digit
            if get_max:
                max_width = roundup(max(widths))
            else:
                max_width = rounddown(min(widths))

            # save max width so that it does not have to be recomputed
            with open(max_width_file, 'w') as f:
                f.write('{}'.format(max_width))

        return max_width

    def _base_dir_surname(self) -> str:
        """Surname of base directories"""
        return 'sim'

    def _exclude_equality_test_keys(self) -> List:
        """Excludes these seeds while testing for config equality"""
        return ['base_seed', 'N']

    def _make_template_settings_file(self,
                                     template_settings: Dict,
                                     id_num: int) -> str:
        """Create temporary settings file for simulating the images

        Parameters
        ----------
        template_settings: (Dict) - Settings to be inserted into simulate template file.
        id_num: (int) - ID number of simulate to name the saved file.
        """
        template_file = self.config["template"]
        name, ext = template_file.split('.')
        tmp_template_file = f'{self.tmp_simulation_folder}/tmp_{id_num:09d}.{ext}'
        txt = ilresources.open_text(settings, self.config['template']).read()
        formatted_txt = txt.format(**template_settings)
        fp = open(tmp_template_file, 'w')
        fp.write(formatted_txt)
        fp.close()
        return tmp_template_file

    def _erase_simulated_images(self):
        """
        Erases images that were originally simulated.
        """
        for f in os.listdir(self.tmp_simulation_folder):
            os.remove(os.path.join(self.tmp_simulation_folder, f))

    def _simulate_and_save(self,
                           id_num: int,
                           except_clause: bool = True):
        """Simulates the genetic data using appropriate software and then saves the output data

        Parameters
        ----------
        id_num: (int) - ID of current simulate

        """
        if except_clause:
            try:
                print(f'Running simulation {id_num}')
                self._load_and_save_sim_data(self._simulate(id_num), id_num)
            except Exception:
                print(f'Simulation {id_num} failed')
        else:
            self._load_and_save_sim_data(self._simulate(id_num), id_num)

    def test_simulations(self, parallel: bool = False, processes: int = 5):
        """Tests that simulations are working correctly"""
        self.seeds = random.sample(range(100000000), self.config['N'])
        if not parallel:
            self._simulate_and_save(1, except_clause=False)
        else:
            pool = multiprocessing.Pool(processes)
            pool.map(self._simulate_and_save, [i+1 for i in range(processes)])

    def run_simulations(self):
        """If simulations have not already been saved, runs the simulations"""
        last_saved_image_ids = [self._last_saved_id(
            datatype) for datatype in self.image_datatypes]
        last_saved_position_ids = [self._last_saved_id(
            datatype) for datatype in self.pos_datatypes]

        last_saved_id = min(last_saved_image_ids + last_saved_position_ids)
        if last_saved_id < self.config['N']:
            self.seeds = random.sample(range(100000000), self.config['N'])
            if self.parallel:
                pool = multiprocessing.Pool(self.max_sub_processes)
                total_range = range(last_saved_id+1, self.config['N']+1)
                to_sim = [i for i in total_range if not all([self.data_exists(id_num=i, datatype=datatype)
                                                             for datatype in self.image_datatypes+self.pos_datatypes])]
                print('Timing first simulation')
                start_time = time.time()
                self._simulate_and_save(to_sim[0])
                seconds = time.time() - start_time
                print(f'One simulation took {seconds:.2f} seconds')
                total_time = len(to_sim)*seconds / self.max_sub_processes
                print(
                    f'Simulating all {len(to_sim)} populations will take approximately {total_time / 60**2:.2f} hours')
                print(f"There are {len(to_sim)} left to simulate")
                pool.map(self._simulate_and_save, to_sim)
            else:
                for i in range(last_saved_id+1, self.config['N']+1):
                    self._simulate_and_save(i)
            self._erase_simulated_images()
