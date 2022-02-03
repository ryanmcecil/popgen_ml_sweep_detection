from typing import Generator, Dict
import os
from abc import ABCMeta
from simulators.simulator import Simulator


def retrieve_simulator(name: str) -> ABCMeta:
    """Retrieves simulator class by name.

    Parameters
    ----------
    name: str
        Name of the simulator class.
    """
    raise NotImplementedError


class SLiM(Simulator):

    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for slim template that do not change as dict.
        """
        if self.settings['template'] == 'msms_match':
            return {'NININDV': self.settings['num_of_individuals']}
        elif self.settings['template'] == 'msms_match_selection':
            return {'NININDV': self.settings['num_of_individuals'],
                    'SELCOEFF': self.settings['sel_coeff']}
        else:
            raise NotImplementedError