from tensorflow.keras.layers import Dropout, Conv1D, Conv2D,MaxPooling2D, MaxPooling1D, Dense, GlobalMaxPooling2D, \
    GlobalMaxPooling1D, AveragePooling2D, AveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling1D, Flatten, \
    concatenate, Conv3D, Reshape, Lambda, ReLU, Add
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras import initializers
import tensorflow as tf
from models.model import SweepDetectionModel


def retrieve_ml_model(name: str):
    if name == 'imagene':
        return ImaGene
    else:
        raise NotImplementedError


class MLModel(SweepDetectionModel):
    '''Abstract class for supervised machine learning models built to detect selective sweeps'''

    def _model(self):
        if self.config['name'] == 'imagene':
            return


class MLSweepModel(tf.keras.Model):

    def classify(self, data) -> int:
        """Classifies input data"""
        prediction = self.predict(data)
        if prediction > 0.5:
            return 1
        else:
            return 0


class ImaGene(MLModel):
    """Implements Imagene Convolutional Neural Network as detailed in
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2927-x#ref-CR30"""

    def _model(self) -> tf.keras.Model:
        """

        Returns
        -------
        tf.keras.Model: The Imagene CNN with specified settings dictated by self.config

        """
        input = Input(shape=(128, 128, 1), name='input')

        x = input
        for i in range(self.config['model']['depth']):
            x = Conv2D(filters=self.config['model']['filters'], kernel_size=(self.config['model']['kernel_size'],
                                                                      self.config['model']['kernel_size']), strides=(1, 1),
                       activation='relu', padding='valid',
                       kernel_initializer=initializers.RandomNormal(mean=0, stddev=1),
                       kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005))(x)

            if self.config['model']['max_pooling'] == True:
                x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        for i in range(self.config['model']['num_dense_layers']):
            x = Dense(units=64, activation='relu')(x)

        x = Dense(units=1, activation='sigmoid')(x)
        model = MLSweepModel(inputs=input, outputs=x)
        return model