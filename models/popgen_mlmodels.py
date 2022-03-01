from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import Input, regularizers
from tensorflow.keras import initializers
import tensorflow as tf
from models.popgen_model import PopGenModel
import numpy as np
from abc import ABC, ABCMeta
from tensorflow.keras.models import load_model
import os
from util.util import save_grey_image
from generator.data_generator import DataGenerator
from tensorflow.keras import backend


class MLPopGenModel(PopGenModel, ABC):
    """Abstract class for supervised machine learning models built to detect selective sweeps"""

    def _load(self):
        """Loads the ML model"""
        model = self._model()
        file = os.path.join(self.data_dir, 'model')
        if os.path.exists(file):
            return model.load(file), True
        else:
            return model, False

    def _classify(self, data: np.ndarray) -> np.ndarray:
        """Classifies input data"""
        data = data[:, 0]
        return np.where(data < 0.5, 0, 1)


class MLSweepModel(tf.keras.Model, metaclass=ABCMeta):
    """Keras Model for detecting selective sweeps"""

    @staticmethod
    def load(filename):
        """Loads the model"""
        return load_model(filename)


def retrieve_ml_model(name: str) -> MLPopGenModel:
    """Retrieves machine learning model type by name

    Parameters
    ----------
    name: (str) - Name of machine learning model

    Returns
    -------

    """
    if name == 'imagene':
        return ImaGene
    else:
        raise NotImplementedError


class ImaGene(MLPopGenModel):
    """Implements Imagene Convolutional Neural Network as detailed in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x#ref-CR30"""

    def _model(self) -> MLSweepModel:
        """

        Returns
        -------
        MLSweepModel: The Imagene CNN with specified settings dictated by self.config

        """
        inpt = Input(shape=(128, 128, 1), name='input')

        x = inpt
        for i in range(self.config['model']['depth']):
            x = Conv2D(filters=self.config['model']['filters'],
                       kernel_size=(self.config['model']['kernel_size'],
                                    self.config['model']['kernel_size']),
                       strides=(1, 1),
                       activation='relu', padding='valid',
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=1),
                       kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005))(x)

            if self.config['model']['max_pooling']:
                x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        for i in range(self.config['model']['num_dense_layers']):
            x = Dense(units=64, activation='relu')(x)

        x = Dense(units=1, activation='sigmoid')(x)
        model = MLSweepModel(inputs=inpt, outputs=x)
        return model

    def produce_layer_outputs(self,
                              base_filename: str,
                              processed_image: np.ndarray,
                              unprocessed_image: np.ndarray):
        """Visualizes the layer outputs given base filename, processed image, and unprocessed image

        Parameters
        ----------
        base_filename: (str) - Base_filename to be appended to beginning of the plots of the visualizations
        processed_image: (np.ndarray) - Genetic image that has been processed. Will be visualized and then used
            to generate the layer outputs
        unprocessed_image: (np.ndarray) - Genetic image that has not been processed. Will be visualized
        """

        # Save base images
        save_grey_image(unprocessed_image, base_filename + '_unprocessed.png')
        save_grey_image(processed_image[0, :, :, 0], base_filename + '_processed.png')

        # Visualize layer outputs
        inp = self.model.input  # input placeholder
        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        functor = backend.function([inp], outputs)  # evaluation function
        names = [layer.name for layer in self.model.layers]  # all layer names
        layer_outs = functor(processed_image)
        layer_num = 1
        for i, name in enumerate(names):
            if 'conv' in name or 'max_pooling' in name:
                save_grey_image(layer_outs[i][0, :, :, 0], base_filename + f'_processed_{layer_num}_{name}.png')
                layer_num += 1

    def visualize_layer_outputs(self, output_filename: str, num: int):
        """Visualizes layer outputs of Imagene model. Currently only works for 'Tiny' Imagene

        Parameters
        ----------
        output_filename: (str) - Base filename which will be appended to the beginning of the saved visualizations
        num: (int) - Number of file to visualize outputs from in test set. Will visualize for the nth neutral image
            and nth sweep image.
        """

        if not num > 0:
            raise Exception('If we are to visualize the nth image, n must be greater than 0.')

        # Load data
        data_generator = DataGenerator(self.config, load_training_data=False, retain_sim_location=True)

        # Count up until we find nth neutral and sweep
        num_sweeps = 0
        num_neutral = 0
        for x, y, filenames, simulators in data_generator.generator('test'):
            get_outputs = False
            label = None
            if y == 0:
                num_neutral += 1
                if num_neutral == num:
                    get_outputs = True
                    label = 'neutral'
            else:
                num_sweeps += 1
                if num_sweeps == num:
                    get_outputs = True
                    label = 'sweep'
            if get_outputs:
                simulator = simulators[0][0]
                file = filenames[0][0].split('/')[-1]
                self.produce_layer_outputs(output_filename + f'_{label}',
                                           x[0],
                                           simulator.load_data(file=file))
            if num_neutral > num and num_sweeps > num:
                break

    def visualize_parameters(self, output_filename: str):
        """Plots the parameters of 'Tiny' Imagene model

        Parameters
        ----------
        output_filename: (str) - Name of output file, dense and conv are added to it
        """
        # Check that we have Tiny Imagene config
        if self.config['model']['filters'] > 1:
            raise NotImplementedError
        if self.config['model']['num_dense_layers'] > 0:
            raise NotImplementedError

        # Plot the dense layer
        for layer in self.model.layers:
            if 'dense' in layer.name:
                weights = layer.get_weights()[0]
                num_weights = weights.shape[0]
                image_size = int(np.sqrt(num_weights))
                image = np.reshape(weights, (image_size, image_size))
                save_grey_image(image, output_filename + '_dense.png')

        # Plot the convolution layer
        for layer in self.model.layers:
            if 'conv' in layer.name:
                kernel_weights = layer.get_weights()[0][:, :, 0, 0]
                save_grey_image(kernel_weights, output_filename + '_conv.png')
