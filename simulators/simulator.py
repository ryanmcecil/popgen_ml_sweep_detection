from abc import ABC, abstractmethod
from typing import Dict, Generator, Union
from util.util import standardize_and_save_data
import os
import random


class Simulator(ABC):
    """Defines parent class for simulators."""

    def __init__(self, settings: Dict, verbose=False):
        self.settings = settings
        self.tmp_simulation_folder = f'{os.getcwd()}/simulators/tmp'
        self.data_dir = f'{os.getcwd()}/simulators/data'
        self.sim_dir = None
        self.verbose = verbose
        file = f'{os.getcwd()}/simulators/{self.settings["software"]}/settings/{self.settings["template"]}'
        if os.path.isfile(file):
            self.template_file = file
        else:
            raise FileNotFoundError
        template_file = self.settings["template"]
        name, ext = template_file.split('.')
        self.tmp_template_file = f'{os.getcwd()}/simulators/{self.settings["software"]}/settings/{name}_tmp.{ext}'

    @abstractmethod
    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for template that do not change as dict.
        """
        pass

    def _update_settings_file(self, template_settings: Dict):
        """Updates the temporary simulation settings file with new settings

        Parameters
        ----------
        template_settings: (Dict) - Settings to be inserted into simulation template file.
        """
        with open(self.template_file, 'r') as template_file:
            with open(self.tmp_template_file, 'w') as tmp_template_file:
                for line in template_file.readlines():
                    for setting_name, setting in template_settings.items():
                        line.replace(setting_name, setting)
                    tmp_template_file.write(line)

    def _erase_simulated_images(self):
        """
        Erases images that were originally simulated.
        """
        for f in os.listdir(self.tmp_simulation_folder):
            os.remove(f)

    def _write_settings_to_txt_file(self, directory: str):
        """Writes simulation settings to text file in directory

        Parameters
        ----------
        directory: (str) - Directory that simulation settings will be written to
        """
        with open(os.path.join(directory, 'settings.txt'), 'w') as settings_file:
            for key,value in self.settings.values():
                # Exclude the number of images we are simulating from settings
                if key is not 'N':
                    settings_file.write(f'{key}: {value}')

    @staticmethod
    def _read_settings_from_txt_file(directory: str) -> Dict:
        """Reads simulation settings from a text file

        Parameters
        ----------
        directory: (str) - Directory in which settings.txt is stored

        Returns
        -------
        settings: (Dict) - Returns a dict of the simulation settings contained in the text file

        """
        settings = {}
        with open(os.path.join(directory, 'settings.txt'), 'r') as settings_file:
            for line in settings_file.readlines():
                line.split(': ')
                settings[line[0]] = line[1]
        return settings

    def _create_data_directory(self):
        """Creates directory to store simulated data for this setting.
        Stores location of new directory in self.sim_dir"""
        new_dir = f'{os.getcwd()}/simulators/sim{os.listdir(self.data_dir):04d}'
        os.mkdir(new_dir)
        self.sim_dir = new_dir
        self._write_settings_to_txt_file(new_dir)

    def _search_for_data_directory(self):
        """Searches for data directory with these simulation settings. If found, stores it in self.sim_dir.
        """
        for sim_dir in os.listdir(self.data_dir):
            settings = self._read_settings_from_txt_file(sim_dir)
            is_settings = True
            for key in settings:
                if key not in self.settings:
                    is_settings = False; break
                if settings[key] != self.settings[key]:
                    is_settings = False; break
            if is_settings:
                self.sim_dir = sim_dir

    def _find_number_of_simulated_images(self) -> int:
        """Finds number of images that have been simulated in simulation directory

        Returns
        -------
        int: Number of simulated images
        """
        raise NotImplementedError

    def _get_next_seed(self) -> int:
        raise NotImplementedError

    def _simulate(self):
        """Calls the simulation file"""
        os.system(f'{self.settings["software"]} {self.tmp_template_file}')

    def _run_simulations(self, starting_count: int):
        """
        Calls appropriate file using bash commands to simulate images
        """
        template_settings = self._retrieve_base_template_settings()
        for i in range(self.settings['N']):
            seed = self._get_next_seed()
            if i >= starting_count:
                template_settings['SEED'] = str(seed)
                template_settings['OUTPUT'] = f'{seed}.txt'
                self._update_settings_file(template_settings)
                self._simulate()
        # Get correct output location
        # save and standardize simulated data
        # erase temporary data
        raise NotImplementedError

    def simulate(self) -> str:
        """
        Calls appropriate bash file to simulate the images. Loads images and puts them into standard form.
        Erases images that were generated during simulation. Saves standard form of images.
        Returns location of standard form images for processing.
        """
        self._search_for_data_directory()
        if self.sim_dir is None:
            self._create_data_directory()
        count = self._find_number_of_simulated_images()
        if count < self.settings['N']:
            self._run_simulations(count)
        return self.sim_dir
