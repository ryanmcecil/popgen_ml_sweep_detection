from abc import ABCMeta
from typing import Dict
import skimage.transform
import numpy as np
from processors.popgen_processor import PopGenProcessor


def retrieve_processor(name: str) -> ABCMeta:
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

    def _check_base_settings(self) -> Dict:
        if 'sorting' not in self.config or 'min_minor_allele_freq' not in self.config or 'resize_dimensions' not in self.config:
            raise ValueError

    def conversion_datatype(self) -> str:
        return 'popgen_image'

    def _sort(self, data: np.ndarray, ordering: str) -> np.ndarray:
        """Imagene type sorting modifed from

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
