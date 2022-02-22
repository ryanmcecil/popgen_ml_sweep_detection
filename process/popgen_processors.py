from typing import Dict
import skimage.transform
import numpy as np
from process.popgen_processor import PopGenProcessor


def retrieve_processor(name: str):
    """Retrieves processor class by name.

    Parameters
    ----------
    name: str
        Name of the processor class
    """
    if name == 'imagene':
        return ImaGeneProcessor
    else:
        raise NotImplementedError


class ImaGeneProcessor(PopGenProcessor):

    def _check_base_settings(self):
        if 'sorting' not in self.config or 'min_minor_allele_freq' not in self.config or 'resize_dimensions' not in self.config:
            raise ValueError

    def conversion_datatype(self) -> str:
        return 'popgen_image'

    def _sort(self, data: np.ndarray, ordering: str) -> np.ndarray:
        """Imagene type sorting modifed from https://github.com/mfumagalli/ImaGene/blob/master/ImaGene.py

        Parameters
        ----------
        data: (np.ndarray) - Binary image data to be sorted
        ordering: (str) - either 'rows_freq', 'cols_freq', 'rows_dist', 'cols_dist'

        Returns
        -------
        np.ndarray: Sorted array
        """

        if ordering == 'rows_freq':
            uniques, counts = np.unique(data, return_counts=True, axis=0)
            counter = 0
            for j in counts.argsort()[::-1]:
                for z in range(counts[j]):
                    data[counter, :] = uniques[j, :]
                    counter += 1
        elif ordering == 'cols_freq':
            uniques, counts = np.unique(data, return_counts=True, axis=1)
            counter = 0  #
            for j in counts.argsort()[::-1]:
                for z in range(counts[j]):
                    data[:, counter] = uniques[:, j]
                    counter += 1
        elif ordering == 'rows_dist':
            uniques, counts = np.unique(data, return_counts=True, axis=0)
            # most frequent row in float
            top = uniques[counts.argsort()[::-1][0]].transpose().astype('float32')
            # distances from most frequent row
            distances = np.mean(np.abs(uniques[:, :, 0] - top), axis=1)
            # fill in from top to bottom
            counter = 0
            for j in distances.argsort():
                for z in range(counts[j]):
                    data[counter, :] = uniques[j, :]
                    counter += 1
        elif ordering == 'cols_dist':
            uniques, counts = np.unique(data, return_counts=True, axis=1)
            # most frequent column
            top = uniques[:, counts.argsort()[::-1][0]].astype('float32')
            # distances from most frequent column
            distances = np.mean(np.abs(uniques[:, :, 0] - top), axis=0)
            # fill in from left to right
            counter = 0
            for j in distances.argsort():
                for z in range(counts[j]):
                    data[:, counter] = uniques[:, j]
                    counter += 1
        else:
            raise NotImplementedError
        return data

    def _majorminor(self, data: np.ndarray):
        """Modifed from Imagene. Converts to major minor polarization

        Parameters
        ----------
        data: (np.ndarray) - Binary image data to be converted

        Returns
        -------
        np.ndarray: Converted data

        """
        idx = np.where(np.mean(data, axis=0) > 0.5)[0]
        data[:,idx] = np.abs(1 - data[:,idx])
        return data

    def _filter_freq(self, data: np.ndarray):
        """Filters by minor allele frequency. Modifed from Imagene

        Parameters
        ----------
        data: (np.ndarray) - data to be filters

        Returns
        -------
        np.ndarray: Filtered data

        """
        idx = np.where(np.mean(data, axis=0) >= self.config['min_minor_allele_freq'])[0]
        data = data[:,idx]
        return data

    def _resize(self, data:np.ndarray):
        """Resize all images to same dimensions. Modified from Imagene.

        Parameters
        ----------
        data: (np.ndarray) - data to be resized

        Returns
        -------
        np.ndarray: resized data

        """
        dimensions = (self.config['resize_dimensions'], self.config['resize_dimensions'])
        data = skimage.transform.resize(data, dimensions, anti_aliasing=True, mode='reflect')
        data = np.where(data < 0.5, 0, 1)
        return data

    def _convert(self, data: np.ndarray) -> np.ndarray:

        data = self._majorminor(data)
        data = self._filter_freq(data)

        if self.config['sorting'] == 'Rows' or self.config['sorting']== 'RowsCols':
            data = self._sort(data, 'rows_freq')

        if self.config['sorting']== 'Cols' or self.config['sorting']== 'RowsCols':
            data = self._sort(data, 'cols_freq')

        if self.config['sorting'] != 'None' and self.config['sorting'] != 'Cols' \
                and self.config['sorting'] != 'Rows' and self.config['sorting'] != 'RowsCols':
            raise Exception('A valid sorting option was not specified')

        data = self._resize(data)

        return data


if __name__ == '__main__':
    from simulate.popgen_simulators import retrieve_simulator
    """For Testing"""
    def simulate_and_process(settings,
                 parallel: bool = True,
                 max_sub_processes: int = 10):
        """Simulates the data specified by settings

        Parameters
        ----------
        settings: - Dictionary containing the settings for simulation and processing.
        parallel: (bool) - If true, simulations are done in parallel
        max_sub_processes: (int) - Defines max number of simulations that can be run in parallel if parallel is true
        """

        for label, sim_config_list in settings['simulations'].items():
            for sim_config in sim_config_list:
                simulator = retrieve_simulator(sim_config['software'])(sim_config,
                                                                       parallel=parallel,
                                                                       max_sub_processes=max_sub_processes)
                simulator.run_simulations()

                for processor_config in settings['conversions']:
                    print(processor_config)
                    processor = retrieve_processor(processor_config['conversion_type'])(config=processor_config,
                                                                                        simulator=simulator,
                                                                                        parallel=parallel,
                                                                                        max_sub_processes=max_sub_processes)
                    processor.run_conversions()


    sim_settings = {
        'neutral': [
            {'software': 'msms',
             'NREF': '10000',
             'N': 100,
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

        ],
        'sweep': [
            {'software': 'msms',
             'N': 100,
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
        ]
    }

    conversion_settings = [{'conversion_type': 'imagene',
                            'sorting': 'Rows',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            },
                           {'conversion_type': 'imagene',
                            'sorting': 'Cols',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            },
                           {'conversion_type': 'imagene',
                            'sorting': 'RowsCols',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            },
                           {'conversion_type': 'imagene',
                            'sorting': 'None',
                            'min_minor_allele_freq': 0.01,
                            'resize_dimensions': 128
                            }
                           ]

    settings = {
        'simulations': sim_settings,
        'conversions': conversion_settings
    }

    simulate_and_process(settings, parallel=True, max_sub_processes=4)
