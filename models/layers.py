from tensorflow.keras.layers import Layer
import tensorflow as tf


class DM(Layer):
    def __init__(self, units = 64, setting = 'cols', **kwargs):
        super(DM, self).__init__()
        self.units = units
        self.setting = setting
        super(DM, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.setting == 'cols':
            self.dense_weights = self.add_weight(name='dw', shape=[input_shape[2], self.units], initializer='random_normal')
        elif self.setting == 'rows':
            self.dense_weights = self.add_weight(name='dw', shape=[input_shape[1], self.units],
                                                 initializer='random_normal')
        super(DM, self).build(input_shape)

    def call(self, input):
        x = input
        if self.setting == 'cols':
            x = x * self.dense_weights
            x = tf.reduce_sum(x, axis=2, keepdims=True)
            x = tf.transpose(x, perm=[0, 1, 3, 2])
        elif self.setting == 'rows':
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            x = x * self.dense_weights
            x = tf.reduce_sum(x, axis=2, keepdims=True)
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        else:
            raise Exception()
        return x

    def get_config(self):
        config = {
            'units' : self.units,
        }

        base_config = super(DM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))