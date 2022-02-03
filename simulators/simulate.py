from typing import Dict, List
from simulators.simulators import retrieve_simulator


def simulate(settings: Dict[List[Dict]]):
    """Simulates the data specified by settings

    Parameters
    ----------
    settings: Dict[List[Dict]] - Dictionary containing the data to be simulated. The first level of the dictionary
        specifies the overall class labels of the data. The List of Dicts at the next level specifies the
        different simulations to run for each of the labels. Each Dict at the last level represents a different
        simulation setting. If data has already been simulated and saved, then this function will not waste time
        re-simulating the data.

    """
    for label, sim_settings_list in settings.items():
        for sim_settings in sim_settings_list:
            simulator = retrieve_simulator(sim_settings['software'])(sim_settings)
            sim_settings['sim_files_location'] = simulator.simulate()
            sim_settings['simulated'] = True


if __name__ == '__main__':
    '''For Testing'''
    raise NotImplementedError


