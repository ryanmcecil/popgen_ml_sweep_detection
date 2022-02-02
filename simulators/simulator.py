from abc import ABC, abstractmethod
from typing import Dict, Generator
from util.util import standardize_and_save_data
import os
import glob


# Option to use generators for storing data
# Option to load everything in
class Simulator(ABC):
    """Defines parent class for simulators."""

    def __init__(self, settings: Dict):
        self.settings = settings

    @abstractmethod
    def _name(self) -> str:
        """
        Returns name of simulator.
        """
        pass

    @abstractmethod
    def _tmp_simulation_folder(self) -> str:
        """
        Returns location of temporary simulation folder.
        """
        pass

    @abstractmethod
    def _settings_file(self) -> str:
        """
        Returns name of settings file.
        """
        pass

    @abstractmethod
    def _update_settings_file(self):
        """
        Updates appropriate settings file with new settings for current simulation.
        """
        pass

    @abstractmethod
    def _simulated_image_generator(self) -> Generator:
        """
        Returns a generator to process the simulated images.
        """
        pass

    def _erase_simulated_images(self):
        """
        Erases images that were originally simulated.
        """
        for f in glob.glob(self._tmp_simulation_folder()):
            os.remove(f)

    def _location_of_saved_images(self) -> str:
        raise NotImplementedError

    def _simulate(self):
        """
        Calls bash file to simulate the images.
        """
        raise NotImplementedError

    def _simulated_images_exist(self):
        """
        Checks to see if images have already been simulated on this setting.
        """
        raise NotImplementedError

    def simulate(self) -> str:
        """
        Calls appropriate bash file to simulate the images. Loads images and puts them into standard form.
        Erases images that were generated during simulation. Saves standard form of images.
        Returns location of standard form images for processing.
        """
        if not self._simulated_images_exist():
            self._update_bash_settings()
            self._simulate()
            standardize_and_save_data(self._simulated_image_generator())
            self._erase_simulated_images()
        return self._location_of_saved_images()

