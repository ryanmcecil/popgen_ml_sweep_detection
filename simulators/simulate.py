from typing import Dict, List
from simulators import retrieve_simulator


def simulate(settings: Dict[str, List[Dict[str, str]]],
             verbose: bool = False,
             parallel: bool = True,
             max_sub_processes: int = 10):
    """Simulates the data specified by settings

    Parameters
    ----------
    settings: Dict[List[Dict]] - Dictionary containing the data to be simulated. The first level of the dictionary
        specifies the overall class labels of the data. The List of Dicts at the next level specifies the
        different simulations to run for each of the labels. Each Dict at the last level represents a different
        simulation setting. If data has already been simulated and saved, then this function will not waste time
        re-simulating the data.
    verbose: (bool) - If true, print out updates on the current simulation.
    parallel: (bool) - If true, simulations are done in parallel
    max_sub_processes: (int) - Defines max number of simulations that can be run in parallel if parallel is true

    """
    print('Beginning simulations')
    for label, sim_config_list in settings.items():
        print(f'Simulating {label}s')
        for sim_config in sim_config_list:
            if verbose:
                print(f'Simulating config: {sim_config}')
            simulator = retrieve_simulator(sim_config['software'])(sim_config, verbose=verbose, parallel=parallel,
                                                                   max_sub_processes=max_sub_processes)
            sim_config['sim_files_location'] = simulator.simulate()
            sim_config['simulated'] = True


if __name__ == '__main__':
    '''For Testing'''
    # settings = {
    #     'neutral': [
    #         {'software': 'msms',
    #          'NREF': '10000',
    #          'N': 1000,
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '128',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ':'`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0',
    #          }
    #     ],
    #     'sweep': [
    #         {'software': 'msms',
    #          'N': 1000,
    #          'NREF': '10000',
    #          'DEMO': '-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2',
    #          'LEN': '80000',
    #          'THETA': '48',
    #          'RHO': '32',
    #          'NCHROMS': '128',
    #          'SELPOS': '`bc <<< \'scale=2; 1/2\'`',
    #          'FREQ': '`bc <<< \'scale=6; 1/100\'`',
    #          'SELTIME': '`bc <<< \'scale=4; 600/40000\'`',
    #          'SELCOEFF': '0.01',
    #          }
    #     ]
    # }
    # simulate(settings, verbose=True)

    settings = {
        'neutral': [
            {'software': 'slim',
             'template': 'msms_match.slim',
             'N': 100,
             'NINDIV': '64'
             }
        ],
        'sweep': [
            {'software': 'slim',
             'template': 'msms_match_selection.slim',
             'N': 100,
             'NINDIV': '64',
             'SELCOEFF': '0.01',
             }
        ]
    }
    simulate(settings, verbose=False)


