from typing import Generator, Dict
import os

from simulators.simulator import Simulator


def retrieve_simulator(name: str) -> Simulator:
    """Retrieves simulator class by name.

    Parameters
    ----------
    name: str
        Name of the simulator class.
    """
    raise NotImplementedError



"""
Make directory with initial settings for neutral and sweep

Process of simulation for SLIM:
- If sweep
    - Replace settings in sweep with settings dictated by settings
    - Simulated N times based on number specified in settings
- If neutral
    - Replace settings in sweep with settings dictated by settings
    - Simulate N times based on number specified in settings
    
- Make way so that exact settings can be reproduced?
"""

class SLiM(Simulator):

    def _name(self) -> str:
        return 'slim'

    def _tmp_simulation_folder(self):
        return f'{os.getcwd()}/simulators/slim/tmp'

    def _settings_file(self) -> str:
        if self.settings['sweep']:
            return f'{os.getcwd()}/simulators/slim/settings/sweep.slim'
        else:
            return f'{os.getcwd()}/simulators/slim/settings/neutral.slim'

    def _replace_line_of_settings_file(self, new_line: str, line_num: int):
        lines = open(self._settings_file(), 'r').readlines()
        lines[line_num] = new_line
        out = open(self._settings_file(), 'w')
        out.writelines(lines)
        out.close()

    def _update_settings_file(self):
        self._replace_line_of_settings_file(f'	initializeMutationRate(1e-8);',
                                            3)

        if self.settings['sweep']:
            return f'{os.getcwd()}/simulators/slim/settings/sweep.slim'
        else:
            return f'{os.getcwd()}/simulators/slim/settings/neutral.slim'

    def _simulated_image_generator(self) -> Generator:
        raise NotImplementedError

    def _location_of_saved_images(self) -> str:
        raise NotImplementedError

    def _simulate(self):

        raise NotImplementedError

    def _simulated_images_exist(self):
        raise NotImplementedError