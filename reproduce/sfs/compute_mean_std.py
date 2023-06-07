from typing import Dict
from simulate.popgen_simulators import retrieve_simulator
import csv
import numpy as np
import os


def msms_neutral_config() -> Dict:
    """Returns configuation for msms neutral simulation"""
    return {'software': 'msms',
             'NREF': '10000',
             'N': 50000,
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ': '`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': '0',
             }


def msms_sweep_config() -> Dict:
    """Returns configuation for msms sweep simulation with selection coefficient of 0.01"""
    return {'software': 'msms',
             'N': 50000,
             'NREF': '10000',
             'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
             'LEN': '80000',
             'THETA': '48',
             'RHO': '32',
             'NCHROMS': '128',
             'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
             'FREQ': '`bc <<< \'scale=6; 1/100\'`',
             'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
             'SELCOEFF': '0.01',
             }


def slim_neutral_config() -> Dict:
    """Returns configuation for slim neutral simulation"""
    return {'software': 'slim',
             'template': 'msms_match.slim',
             'N': 50000,
             'NINDIV': '64'
             }


def slim_sweep_config() -> Dict:
    """Returns configuation for slim sweep simulation with selection coefficient of 0.01"""
    return {'software': 'slim',
             'template': 'msms_match_selection.slim',
             'N': 50000,
             'NINDIV': '64',
             'SELCOEFF': '0.01',
             }


if __name__ == '__main__':
    sims = {'msms_neutral': msms_neutral_config(),
            'msms_sweep': msms_sweep_config(),
            'slim_neutral': slim_neutral_config(),
            'slim_sweep': slim_sweep_config()
            }

    with open(os.path.join(os.getcwd(),'reproduce', 'sfs','results', 'results/means_and_stds.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['sim', 'mean', 'std'])
        for sim, config in sims.items():
            print(f'Computing Mean and Stand Deviation for sim {sim}')
            simulator = retrieve_simulator(config['software'])(config)
            widths = []
            for i in range(config['N']):
                print(id)
                id = i+1
                widths.append(simulator.load_data(id_num=id, datatype='popgen_image').shape[1])
            writer.writerow([sim, f'{np.mean(widths):.2f}', f'{np.std(widths):.2f}'])
