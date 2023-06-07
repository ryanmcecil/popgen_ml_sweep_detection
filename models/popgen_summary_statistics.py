import os
from math import isnan
from typing import List

import allel
import numpy as np

from generator.data_generator import DataGenerator
<<<<<<< HEAD
from models.popgen_model import PopGenModel


def all_statistics():
    return ['ihs_maxabs', 'tajima_d', 'garud_h1', 'garud_h12', 'garud_h2_h1', 'n_columns']
=======
import os
from math import isnan


def all_statistics():
    return ['ihs_maxabs', 'tajima_d', 'garud_h1', 'n_columns']
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83


def all_image_and_position_statistics():
    return ['ihs_maxabs']


class SummaryStatPopGenModel(PopGenModel):
    def _model(self):
        """Initializes the model"""
        if 'standardized' in self.config['model'] and self.config['model']['standardized']:
            return StandardizedStatistic(self.config['model']['name'])
        else:
            return Statistic(self.config['model']['name'])

    def _load(self):
        model = self._model()
        file = os.path.join(self.data_dir, 'model.npy')
        if os.path.exists(file):
            model.load(file)
            return model, True
        else:
            return model, False

    def _classify(self, data: np.ndarray) -> np.ndarray:
        """Classifies input data"""
        if self.model.threshold is None:
            raise Exception('The statistic has not been trained')
        return np.where(data < self.model.threshold, 0, 1)


def compute_threshold(statistics: np.ndarray,
<<<<<<< HEAD
                      labels: np.ndarray) -> float:
=======
        labels: np.ndarray) -> float:
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    """Computes optimal decision threshold based on computed statistics from training dataset

    Parameters
    ----------
    statistics (np.ndarray): np arrays of computed statistics for each training sample
    labels (np.ndarray): np array of labels for each sample
    """

    # thresholds in descending order
    potential_thresholds = np.sort(statistics)[::-1]
    best_threshold = 0
    best_acc = 0
    for threshold in potential_thresholds:
        classifications = np.where(statistics >= threshold, 1, 0)
        errors = np.abs(labels - classifications)
        acc = 1 - np.mean(errors)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold


class Statistic:
    """Implements sweep detection test statistics"""

    def __init__(self,
                 statistic: str):
        """Initializes the statistic

        Parameters
        ----------
        statistic: (str) - Name of the statistic
        """
        self.name = statistic
        self.threshold = None
        if 'ihs_maxabs' in statistic:
            self._predict = self._predict_ihs_max
        elif 'tajima_d' in statistic:
            self._predict = self._predict_td
        elif 'nsl' in statistic:
            self._predict = self._predict_nsl
        elif 'garud_h1' in statistic:
            self._predict = self._predict_garud_h1
        elif 'garud_h12' in statistic:
            self._predict = self._predict_garud_h12
        elif 'garud_h123' in statistic:
            self._predict = self._predict_garud_h123
        elif 'garud_h2_h1' in statistic:
            self._predict = self._predict_garud_h2_h1
        elif 'n_columns' in statistic:
            self._predict = self._predict_n_columns
        else:
            raise Exception(f'The test statistic {statistic} is unknown.')

    def save(self,
             filename: str):
        """Saves the statistic with threshold

        Parameters
        ----------
        filename: (str) - Filename that statistic settings will be saved to
        """
        if self.threshold is None:
            raise Exception("Statistic should be trained before it is saved")
        np.save(filename, self.threshold)

    def load(self,
             filename: str):
        """Loads threshold from file

        Parameters
        ----------
        filename: (str) - Name of file that threshold has been saved to
        """
        if self.threshold is not None:
            raise Exception("Threshold has already been trained or loaded")
        self.threshold = np.load(filename)

    def set_threshold(self,
                      threshold: float):
        """Sets the threshold for classification

        Parameters
        ----------
        threshold: (float) - If prediction is above threshold, it will be classifed as sweep
        """
        self.threshold = threshold

    def fit(self,
            datagenerator: DataGenerator,
            **kwargs):
        """Trains the statistic model using the data supplied by the data generator and then saves it

        Parameters
        ----------
        datagenerator: (DataGenerator) - Data generator class supplying training data
        kwargs: Extra parameters that might be passed in due to Keras fit
        """
        print('============================================')
        print(f'Training Statistic {self.name} on data')
        statistics = []
        labels = []
        for x, y in datagenerator.generator('train'):
            statistics += list(self.predict(x))
            labels += list(y)
<<<<<<< HEAD
            # print(f'{len(statistics)} stats have been computed')

        self.threshold = compute_threshold(
            np.asarray(statistics), np.asarray(labels))
=======
            print(f'{len(statistics)} stats have been computed')

        self.threshold = compute_threshold(np.asarray(statistics), np.asarray(labels))
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

    def predict(self, data: List[np.ndarray]) -> np.ndarray:
        stats = []
        for i in range(data[0].shape[0]):
            input_data = []
            for item in data:
                input_data.append(item[i, ...])
            stats.append(self._predict(input_data))
        return np.asarray(stats)

    @staticmethod
    def _predict_ihs_max(data: List[np.ndarray]) -> float:
        """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

        Parameters
        ----------
        data List[np.ndarray]: [genetic data, genetic positions]

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        if not isinstance(data, list):
            raise Exception('The ihs test statistic has multiple inputs')
        genetic_data = data[0][:, :, 0]
        positions = data[1][:]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        ihs = allel.ihs(h1, pos=positions, include_edges=True)
        output = float(np.nanmax(np.abs(ihs)))
        return output

    @staticmethod
    def _predict_td(data: List[np.ndarray]) -> float:
        """Computes tajima d test statistic of genetic data

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        ac = h1.count_alleles()
        output = allel.tajima_d(ac)
        return output

    @staticmethod
    def _predict_nsl(data: List[np.ndarray]) -> float:
        """Computes nsl test statistic of genetic data

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        nsl = allel.nsl(h1)
        output = np.nanmax(np.abs(nsl))
        return output

    @staticmethod
    def _predict_garud_h1(data: List[np.ndarray]) -> float:
        """Computes garud's H1 test statistic

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        h1, _, _, _ = allel.garud_h(h1)
        return h1

    @staticmethod
    def _predict_garud_h12(data: List[np.ndarray]) -> float:
        """Computes garud's H12 test statistic

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        _, h12, _, _ = allel.garud_h(h1)
        return h12

    @staticmethod
    def _predict_garud_h123(data: List[np.ndarray]) -> float:
        """Computes garud's H123 test statistic

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of
        positions (Iterable): Iterable of positions if needed

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        _, _, h123, _ = allel.garud_h(h1)
        return h123

    @staticmethod
    def _predict_garud_h2_h1(data: List[np.ndarray]) -> float:
        """Computes garud's H2/H1 test statistic

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of
        positions (Iterable): Iterable of positions if needed

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        _, _, _, h2_h1 = allel.garud_h(h1)
        return -h2_h1  # Make negative to flip sides of classification threshold

    def _predict_n_columns(self, data: List[np.ndarray]) -> float:
        """Computes statistic based on number of columns in image

        Parameters
        ----------
        data (np.ndarray): Genetic data to compute test statistic of
        positions (Iterable): Iterable of positions if needed

        Returns
        -------
        output (float): Returns the statistic of the data

        """
        genetic_data = data[0][:, :, 0]
        return -genetic_data.shape[1]


class StandardizedStatistic:
    """Implements sweep detection test statistics which are standardized in some form"""

    def __init__(self,
                 statistic: str):
        """Initializes the statistic

        Parameters
        ----------
        statistic: (str) - Name of the statistic
        """
        self.name = statistic
        self.threshold = None
        self.means = None
        self.stds = None
        if 'ihs_maxabs' in statistic:
            self._col_statistics = self._col_statistics_ihs
            self._predict = self._predict_ihs_max
        else:
            raise Exception(f'The test statistic {statistic} is unknown.')

    def save(self,
             filename: str):
        """Saves the statistic with threshold, mean, and std values

        Parameters
        ----------
        filename: (str) - Filename that statistic settings will be saved to
        """
        if self.threshold is None:
            raise Exception("Statistic should be trained before it is saved")
        np.save(filename, self.threshold)
        np.save(f'{filename}_means', self.means)
        np.save(f'{filename}_stds', self.stds)

    def load(self,
             filename: str):
        """Loads threshold, means, and stds from file

        Parameters
        ----------
        filename: (str) - Name of file that settings have been stored to
        """
        if self.threshold is not None:
            raise Exception("Threshold has already been trained or loaded")
        self.threshold = np.load(filename)
        self.means = np.load(filename.replace('.npy', '_means.npy'))
        self.stds = np.load(filename.replace('.npy', '_stds.npy'))

    def set_means_stds(self,
                       means: np.ndarray,
                       stds: np.ndarray):
        """Sets the mean, stds settings for classification

        Parameters
        ----------
        means: (np.ndarray) - 1D array of bin means for standardization
        stds: (np.ndarray) - 1D array of bin standard deviations
        """
        self.means = means
        self.stds = stds

    def fit(self,
            datagenerator: DataGenerator,
            **kwargs):
        """Trains the statistic model using the data supplied by the data generator and then saves it

        Parameters
        ----------
        datagenerator: (DataGenerator) - Data generator class supplying training data
        kwargs: Extra parameters that might be passed in due to Keras fit
        """
        print('============================================')
        print(f'Training Statistic {self.name} on data')

        # First process all neutral images and compute standardization settings based on derived allele frequency
        print('Computing Standardization')
        bins = [[] for _ in range(128)]
        num = 0
        for x, y in datagenerator.generator('train'):
            num += 1
            for i in range(x[0].shape[0]):
                if y[i] == 0:
                    input_data = [item[i, ...] for item in x]
                    ihs = self._col_statistics(input_data)
                    counts = self.derived_allele_frequency(input_data[0])
                    for j, count in enumerate(counts):
                        bins[count - 1].append(ihs[j])
            print(num)
<<<<<<< HEAD
        means = np.nan_to_num(np.asarray(
            [np.nanmean(binn) for binn in bins], dtype=float))
        stds = np.nan_to_num(np.asarray(
            [np.nanstd(binn) for binn in bins], dtype=float), nan=1.0)
=======
        means = np.nan_to_num(np.asarray([np.nanmean(binn) for binn in bins], dtype=float))
        stds = np.nan_to_num(np.asarray([np.nanstd(binn) for binn in bins], dtype=float), nan=1.0)
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
        self.set_means_stds(means, stds)

        print('Finding Threshold')
        statistics = []
        labels = []
        # Now compute threshold
        for x, y in datagenerator.generator('train'):
            prediction = self.predict(x)
            if prediction.size != 0:
                statistics += list(prediction)
                labels += list(y)
            print(f'{len(statistics)} stats have been computed')

<<<<<<< HEAD
        self.threshold = compute_threshold(
            np.asarray(statistics), np.asarray(labels))
=======
        self.threshold = compute_threshold(np.asarray(statistics), np.asarray(labels))
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

    def predict(self, data: List[np.ndarray]) -> np.ndarray:
        stats = []
        for i in range(data[0].shape[0]):
            input_data = [item[i, ...] for item in data]
            statistic = self._predict(input_data)
            if statistic is not None:
                stats.append(statistic)
        return np.asarray(stats)

    @staticmethod
    def derived_allele_frequency(image_data: np.ndarray) -> np.ndarray:
        """Computes derived allele frequency for each column

        Parameters
        ----------
        image_data: (np.ndarray) - Genetic image data

        Returns
        -------
        np.ndarray: np array with allele frequency counts for each column

        """
        genetic_data = image_data[:, :, 0]
        return np.sum(genetic_data.astype(np.int), axis=0)

    @staticmethod
    def _col_statistics_ihs(data: List[np.ndarray]) -> np.ndarray:
        if not isinstance(data, list):
            raise Exception('The ihs test statistic has multiple inputs')
        genetic_data = data[0][:, :, 0]
        positions = data[1][:]
        haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
        h1 = allel.HaplotypeArray(haplos)
        ihs = allel.ihs(h1, pos=positions, include_edges=True)
        return ihs

    def _predict_ihs_max(self, data: List[np.ndarray]) -> float:
        """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

        Parameters
        ----------
        data List[np.ndarray]: [genetic data, genetic positions]

        Returns
        -------
        output (float): Returns the statistic of the data
        """
        # Compute raw statistics
        ihs = self._col_statistics_ihs(data)
        # Standardize statistics based on bin
        counts = self.derived_allele_frequency(data[0])
        ihs = (ihs - self.means[counts - 1]) / self.stds[counts - 1]
        # Compute final prediction
        output = float(np.nanmax(np.abs(ihs)))
        if isnan(output):
            return None
        else:
            return output
