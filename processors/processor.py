from abc import ABC, abstractmethod
from typing import Dict, Generator
from util.util import load_standardized_data, standardize_and_save_data


# Option to use generators for storing data
# Option to load everything in
# When loading data, slowly load it, if we pass threshold of max memory, switch to generator code
class Processor(ABC):
    """Defines parent class for processors."""

    def __init__(self, settings: Dict):
        raise NotImplementedError

    @abstractmethod
    def _name(self) -> str:
        """
        Returns name of the processor.
        """
        pass

    @abstractmethod
    def _process(self, data):
        """Process the given data.

        Parameters
        ----------
        data:
            Data that should be processed.
        """
        pass

    def _processed_files_exist(self):
        """
        Checks to see if images have already been processed on this setting.
        """
        raise NotImplementedError

    def _location_of_saved_files(self) -> str:
        raise NotImplementedError

    def process(self) -> str:
        if not self._processed_files_exist():
            x = self._process(load_standardized_data(self._location_of_saved_files()))
            standardize_and_save_data(x)
        return self._location_of_saved_files()

