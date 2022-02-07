from typing import Generator, Dict
import os
from abc import ABCMeta
from simulator import Simulator
import numpy as np
import subprocess


def retrieve_simulator(name: str) -> ABCMeta:
    """Retrieves simulator class by name.

    Parameters
    ----------
    name: str
        Name of the simulator class.
    """
    if name == 'slim':
        return SLiM
    elif name == 'msms':
        return MSMS
    else:
        raise NotImplementedError


class SLiM(Simulator):

    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for slim template that do not change as dict.
        """
        if self.config['template'] == 'msms_match.slim':
            return {'NINDIV': self.config['NINDIV']}
        elif self.config['template'] == 'msms_match_selection.slim':
            return {'NINDIV': self.config['NINDIV'],
                    'SELCOEFF': self.config['SELCOEFF']}
        else:
            raise NotImplementedError

    def _simulate(self, id: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id: (int) - ID of current simulation

        """
        template_settings = self.template_settings.copy()
        template_settings['SEED'] = str(self.seeds[id-1])
        template_settings['OUTPUT'] = f'\"{self.tmp_simulation_folder}/{id:09d}.txt\"'
        tmp_settings_file = self._make_template_settings_file(template_settings, id)
        self.log('Beginning simulation')
        os.system(f'{self.config["software"]} {tmp_settings_file} >/dev/null 2>&1')
        os.remove(tmp_settings_file)
        return f'{self.tmp_simulation_folder}/{id:09d}.txt'

    def _load_sim_data(self, file: str) -> [np.ndarray, np.ndarray]:
        """Loads the simulation data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulation file

        Returns
        -------
        [np.ndarray, np.ndarray]: Genetic image and the loci positions as np arrays
        """
        with open(file, 'r') as sim_file:
            lines = sim_file.readlines()
            segsites = int(lines[1].replace('segsites: ', '').replace('\n', ''))
            image = np.zeros((int(self.config['NINDIV'])*2, segsites))
            positions = lines[2].replace('positions: ', '').replace('\n', '').split()
            positions = np.asarray([float(position) for position in positions])
            for i, line in enumerate(lines[3:]):
                image[i, :] = [int(d) for d in line.replace('\n', '')]
        return image, positions


class MSMS(Simulator):

    def _retrieve_base_template_settings(self) -> Dict:
        """
        Returns
        -------
        Dict: Returns settings for msms template that do not change as dict.
        """
        return {'NREF': self.config['NREF'],
                    'DEMO': self.config['DEMO'],
                    'LEN': self.config['LEN'],
                    'THETA': self.config['THETA'],
                    'RHO': self.config['RHO'],
                    'NCHROMS': self.config['NCHROMS'],
                    'SELPOS': self.config['SELPOS'],
                    'FREQ': self.config['FREQ'],
                    'SELTIME': self.config['SELTIME'],
                    'SELCOEFF': self.config['SELCOEFF']
                    }

    def _simulate(self, id: int) -> str:
        """Simulates the genetic data. Returns location of file.

        Parameters
        ----------
        id: (int) - ID of current simulation

        """
        ts = self.template_settings.copy()
        ts['SEED'] = str(self.seeds[id-1])
        ts['OUTPUT'] = f'{self.tmp_simulation_folder}/{id:09d}.txt'
        self.log('Beginning simulation')
        tmp_bash_file = f'{self.tmp_simulation_folder}/{id:09d}.sh'
        with open(tmp_bash_file, "w") as bash_file:
            bash_file.write('#!/bin/bash\n')
            bash_file.write(f'java -jar $PWD/simulators/msms/lib/msms.jar -seed {ts["SEED"]} -N {ts["NREF"]} -ms {ts["NCHROMS"]} 1 '
                     f'-t {ts["THETA"]} -r {ts["RHO"]} {ts["LEN"]} -Sp {ts["SELPOS"]} -SI {ts["SELTIME"]} 1 {ts["FREQ"]} '
                     f'-SAA $(({int(float(float(ts["SELCOEFF"])*int(ts["NREF"])*2))})) -SAa $(({int(float(float(ts["SELCOEFF"])*int(ts["NREF"])))})) -Saa 0 '
                     f'-Smark {ts["DEMO"]} -thread 4 > {ts["OUTPUT"]}\n')
        os.system(f'bash {tmp_bash_file}')
        os.remove(tmp_bash_file)
        return f'{self.tmp_simulation_folder}/{id:09d}.txt'

    def _load_sim_data(self, file: str) -> [np.ndarray, np.ndarray]:
        """Loads the simulation data from given file and returns the image and positions as np array

        Parameters
        ----------
        file: (str) - Location of simulation file

        Returns
        -------
        [np.ndarray, np.ndarray]: Genetic image and the loci positions as np arrays
        """
        with open(file, 'r') as sim_file:
            lines = sim_file.readlines()
            segsites = int(lines[4].replace('segsites: ', '').replace('\n', ''))
            image = np.zeros((int(self.config['NCHROMS']), segsites))
            positions = lines[5].replace('positions: ', '').replace('\n', '').split()
            positions = np.asarray([float(position) for position in positions])
            for i, line in enumerate(lines[6:]):
                if line != '\n':
                    image[i, :] = [int(d) for d in line.replace('\n', '')]
        return image, positions
