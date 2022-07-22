from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
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
import shap
from matplotlib import pyplot as plt


class MLPopGenModel(PopGenModel, ABC):
    """Abstract class for supervised machine learning models built to detect selective sweeps"""

    def _load(self, load_from_file: bool=True):
        """Loads the ML model"""
        model = self._model()
        file = os.path.join(self.data_dir, 'model')
        if load_from_file and os.path.exists(file):
            return model.load(file), True
        else:
            return model, False

    def number_of_parameters(self):
        """Returns number of trainable parameters in ML model."""
        trainable_count = np.sum([backend.count_params(w) for w in self.model.trainable_weights])
        return trainable_count

    def _classify(self, data: np.ndarray) -> np.ndarray:
        """Classifies input data"""
        data = data[:, 0]
        return np.where(data < 0.5, 0, 1)

    def apply_shap(self,
                   save_dir: str,
                   generate_results: bool = True):
        """Utilizes Shap package: https://github.com/slundberg/shap to generate explanations for model predictions
        for a single sweep and neutral image, and across the whole dataset

        Parameters
        ----------
        save_dir: (str) - Directory to save plots to
        generate_results: (bool) - If True, re-computes shap values and saves them before generating plots
        """

        # Generate and save shap values
        if generate_results:

            num_shap_deep_explainer_images = 20
            num_images = 1000
            batch_size = 100
            assert num_images > num_shap_deep_explainer_images

            # Load neutral and sweep images from test set
            data_generator = DataGenerator(self.config, load_training_data=False)
            neutrals = []
            sweeps = []
            to_explain = []
            for x, y, in data_generator.generator('test'):
                if y == 0:
                    neutrals.append(x[0])
                    if len(to_explain) == 0:
                        to_explain.append(x[0])
                if y == 1:
                    sweeps.append(x[0])
                    if len(to_explain) == 1:
                        to_explain.append(x[0])

            to_explain = np.concatenate(to_explain, axis=0)
            neutrals = np.concatenate(neutrals, axis=0)[0:num_images]
            sweeps = np.concatenate(sweeps, axis=0)[0:num_images]
            X = np.concatenate([neutrals[0:num_shap_deep_explainer_images], sweeps[0:num_shap_deep_explainer_images]], axis=0)

            e = shap.DeepExplainer(self.model, X)

            sweep_shap_values = []
            neutral_shap_values = []
            i = 0
            while i*batch_size < num_images:
                print(i)
                sweep_shap_values.append(e.shap_values(sweeps[i*batch_size:(i+1)*batch_size])[0])
                neutral_shap_values.append(e.shap_values(neutrals[i*batch_size:(i+1)*batch_size])[0])
                i += 1
            sweep_shap_values = np.mean(np.concatenate(sweep_shap_values, axis=0), axis=0, keepdims=True)
            neutral_shap_values = np.mean(np.concatenate(neutral_shap_values, axis=0), axis=0, keepdims=True)
            to_explain_shap_values = e.shap_values(to_explain)

            np.save(os.path.join(self.data_dir, 'sweep_shap_values.npy'), sweep_shap_values)
            np.save(os.path.join(self.data_dir, 'neutral_shap_values.npy'), neutral_shap_values)
            np.save(os.path.join(self.data_dir, 'to_explain_shap_values.npy'), to_explain_shap_values)
            np.save(os.path.join(self.data_dir, 'to_explain.npy'), to_explain)

        # Load saved shap values and plot results
        sweep_shap_values = np.load(os.path.join(self.data_dir, 'sweep_shap_values.npy'))
        neutral_shap_values = np.load(os.path.join(self.data_dir, 'neutral_shap_values.npy'))
        to_explain_shap_values = np.load(os.path.join(self.data_dir, 'to_explain_shap_values.npy'))
        to_explain_images = np.load(os.path.join(self.data_dir, 'to_explain.npy'))

        # Plot mean shap vals
        shap_values = np.concatenate([neutral_shap_values, sweep_shap_values], axis=0)
        zero_image = np.zeros(shape=shap_values.shape)
        shap.image_plot(shap_values, zero_image, np.asarray([['Neutral', 'Sweep'], ['Sweep', 'Neutral']]), show=False)
        plt.savefig(os.path.join(save_dir, 'shap_mean_explanations.png'))
        plt.clf()

        # Plot individual shap vals
        shap.image_plot(to_explain_shap_values[0],
                        to_explain_images,
                        np.asarray([['Neutral'],['Sweep']]),
                        show=False,
                        true_labels=['Neutral', 'Sweep'])
        plt.savefig(os.path.join(save_dir, 'shap_explanations.png'))
        plt.clf()


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
    elif name == 'deepset':
        return DeepSet
    elif name == 'imasortgene':
        return ImaSortGene
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
        inpt = Input(shape=(self.config['model']['image_height'], self.config['model']['image_width'], 1), name='input')

        conv_regularizer = regularizers.l1_l2(l1=0.005, l2=0.005)
        dense_regularizer = None
        if 'l1_regularize' in self.config['model']:
            if self.config['model']['l1_regularizer']:
                conv_regularizer = regularizers.l1(0.2)
                dense_regularizer = regularizers.l1(0.2)

        x = inpt
        for i in range(self.config['model']['depth']):
            if self.config['model']['convolution']:
                x = Conv2D(filters=self.config['model']['filters'],
                           kernel_size=(self.config['model']['kernel_height'],
                                        self.config['model']['kernel_width']),
                           strides=(1, 1),
                           activation=None, padding='valid', use_bias=False,
                           kernel_initializer=initializers.RandomNormal(mean=0, stddev=1),
                           kernel_regularizer=conv_regularizer)(x)

            if self.config['model']['relu']:
                x = Activation('relu')(x)

            if self.config['model']['max_pooling']:
                x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        for i in range(self.config['model']['num_dense_layers']):
            x = Dense(units=64, activation='relu', kernel_regularizer=dense_regularizer, use_bias=False)(x)

        x = Dense(units=1, activation='sigmoid', kernel_regularizer=dense_regularizer, use_bias=False)(x)
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
            if 'conv' in name or 'max_pooling' in name or 'activation' in name:
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
                self.produce_layer_outputs(output_filename + f'{label}',
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
                initial_width = self.config['model']['image_width']
                image_width = initial_width - (self.config['model']['kernel_width'] - 1)
                image_height = layer.input_shape[1] // image_width
                image = np.reshape(weights, (image_height, image_width))
                save_grey_image(image, output_filename + 'dense.png')

        # Plot the convolution layer
        for layer in self.model.layers:
            if 'conv' in layer.name:
                kernel_weights = layer.get_weights()[0][:, :, 0, 0]
                save_grey_image(kernel_weights, output_filename + 'conv.png')


class DeepSet(MLPopGenModel):

    def _model(self) -> MLSweepModel:
        """

        Returns
        -------
        DeepSet Model

        """

        input = Input(shape=(self.config['model']['image_height'], self.config['model']['image_width'], 1), name='input')

        x = input
        for i in range(self.config['model']['depth']):
            x = Conv2D(filters=self.config['model']['filters'], kernel_size=(1,self.config['model']['kernel_size']),
                       activation='relu', padding='valid')(x)

        x = tf.reduce_mean(x, axis=1)

        x = Flatten()(x)

        for i in range(self.config['model']['num_dense_layers']):
            x = Dense(units=self.config['model']['num_units'], activation='relu')(x)

        x = Dense(units=1, activation='sigmoid')(x)
        model = MLSweepModel(inputs=input, outputs=x)
        return model
