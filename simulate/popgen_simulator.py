from abc import ABC, abstractmethod
from util.popgen_data_class import PopGenDataClass
from typing import Dict, List
import os
import random
import numpy as np
import multiprocessing
import re
from simulate.software import settings
try:
    import importlib.resources as ilresources
except ImportError:
    try:
        import importlib_resources as ilresources
    except ImportError:
        raise ImportError('Must install backport of importlib_resources if not using Python >= 3.7')
# Seed generated using os.urandom
# Defines seeds for all simulations dependant on number of sim directory
random.seed(434935399793924222151)


class PopGenSimulator(PopGenDataClass, ABC):
    """Defines abstract class for simulate."""

    def __init__(self,
                 config: Dict,
                 root_dir: str = os.path.join(os.getcwd(), 'simulate', 'simulations'),
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
        self.tmp_simulation_folder = os.path.join(os.getcwd(),'simulate','tmp')
        if not os.path.isdir(self.tmp_simulation_folder):
            os.mkdir(self.tmp_simulation_folder)
        self.template_settings = self._retrieve_base_template_settings()

        self.seeds = None
        sim_num = int(re.findall(f".*{self._base_dir_surname()}_(.*)", self.base_dir)[0])
        self.config['base_seed'] = random.sample(range(100000000),  len(os.listdir(self.root_dir)))[sim_num]

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
        """Loads the simulate data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulate file

        Returns
        -------
        [np.ndarray, np.ndarray]: Genetic image and the loci positions as np arrays
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
                           id_num: int):
        """Simulates the genetic data using appropriate software and then saves the output data

        Parameters
        ----------
        id_num: (int) - ID of current simulate

        """
        if self.data_exists(id_num=id_num, datatype='popgen_image') and self.data_exists(id_num=id_num, datatype='popgen_positions'):
            pass
        else:
            print(f'Running simulation {id_num}')
            image, positions = self._load_sim_data(self._simulate(id_num))
            if id_num == 1:
                self.plot_example_image(image)
            self.save_data(image, id_num, 'popgen_image')
            self.save_data(positions, id_num, 'popgen_positions')

    def run_simulations(self):
        """If simulations have not already been saved, runs the simulations"""

        last_saved_image_id = self._last_saved_id('popgen_image')
        last_saved_positions_id = self._last_saved_id('popgen_positions')
        last_saved_id = min(last_saved_image_id, last_saved_positions_id)

        if last_saved_id < self.config['N']:
            self.seeds = random.sample(range(100000000), self.config['N'])
            if self.parallel:
                pool = multiprocessing.Pool(self.max_sub_processes)
                pool.map(self._simulate_and_save, range(last_saved_id+1, self.config['N']+1))
            else:
                for i in range(last_saved_id+1, self.config['N']+1):
                    self._simulate_and_save(i)
            self._erase_simulated_images()