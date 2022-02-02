from abc import ABC, abstractmethod
from typing import Dict, Generator


class Statistic(ABC):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def _name(self) -> str:
        """
        Returns name of the processor.
        """
        pass

    @abstractmethod
    def _compute(self, data):
        """
        Computes the statistic from the data.
        """
        pass



