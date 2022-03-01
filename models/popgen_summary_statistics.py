from models.popgen_model import PopGenModel
import allel
import numpy as np
from typing import List
from generator.data_generator import DataGenerator
import os


def all_statistcs():
    return ['ihs_maxabs', 'tajima_d', 'garud_h1']


def all_image_and_position_statistics():
    return ['ihs_maxabs']


class SummaryStatPopGenModel(PopGenModel):
    def _model(self):
        """Initializes the model"""
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
        for x,y in datagenerator.generator('train'):
            statistics += list(self.predict(x))
            labels += list(y)
            print(f'{len(statistics)} stats have been computed')
        statistics = np.asarray(statistics)
        labels = np.asarray(labels)
        potential_thresholds = np.linspace(start=np.min(statistics), stop=np.max(statistics), num=1000)
        best_threshold = 0
        best_acc = 0
        for threshold in potential_thresholds:
            classifications = np.where(statistics > threshold, 1, 0)
            errors = np.abs(labels - classifications)
            acc = 1 - np.mean(errors)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        self.threshold = best_threshold

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
        output = np.nanmax(np.abs(ihs))[0]
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
        return h2_h1

