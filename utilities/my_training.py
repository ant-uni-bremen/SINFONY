#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import backend as KB

from packaging import version

# Only imported for Gaussiannoise2 layer
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from tensorflow.keras.layers import Layer


# Training and Custom Layers ------------------------------------------------------------------

# Custom callbacks

class BatchTrackingCallback(K.callbacks.Callback):
    '''Log training losses and accuracies after each single batch iteration
    '''

    def __init__(self):
        self.batch_end_loss = []
        self.batch_end_acc = []
        # self.batch = []
    # def on_train_begin(self, logs = {}):
    # 	self.batch_end_loss = []
    # 	self.batch_end_acc = []
    # 	# self.batch = []

    def on_train_batch_end(self, batch, logs=None):
        '''Log losses and accuracies on training batch end
        NOTE: batch is required as an input here
        '''
        self.batch_end_loss.append(logs['loss'])
        self.batch_end_acc.append(logs['accuracy'])
        # self.batch.append(batch)

# Convenience Functions


def gpu_select(number=0, memory_growth=True, cpus=0):
    '''Select/deactivate GPU in Tensorflow 2
    Configure to use only a single GPU and allocate only as much memory as needed
    For more details, see https://www.tensorflow.org/guide/gpu
    '''
    if number >= 0:
        # Choose GPU
        gpus = tf.config.list_physical_devices('GPU')
        print('Number of GPUs available :', len(gpus))
        if gpus:
            gpu_number = number  # Index of the GPU to use
            try:
                tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
                print('Only GPU number', gpu_number, 'used.')
                tf.config.experimental.set_memory_growth(
                    gpus[gpu_number], memory_growth)
            except RuntimeError as error:
                print(error)
    elif number == -1:
        # Deactivate GPUs and use CPUs
        try:
            tf.config.experimental.set_visible_devices([], 'GPU')
            print('GPUs deactivated.')
        except RuntimeError as error:
            print(error)
        if cpus > 0:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(cpus)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                print(cpus, 'CPUs used.')
            except RuntimeError as error:
                print(error)
    else:
        print('Will choose GPU or CPU automatically.')


# Custom Layer Functions


def normalize_input_legacy(inputs, axis=0, eps=0):
    '''Normalize power of inputs to one, legacy version
    axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since inputs=0, then NaN!
    '''
    out = inputs / KB.sqrt(KB.mean(KB.square(inputs) +
                                   eps, axis=axis, keepdims=True))         # Keras backend version
    # out = inputs / \
    #     tf.math.sqrt(tf.reduce_mean(
    #         inputs ** 2 + eps, axis=axis, keepdims=True))
    return out


@tf.function
def normalize_input_keras3(inputs, axis=0, eps=0):
    '''Normalize power of inputs to one
    axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since inputs=0, then NaN!
    '''
    out = inputs / \
        K.ops.sqrt(K.ops.mean(K.ops.square(inputs) +
                   eps, axis=axis, keepdims=True))
    return out


if version.parse(tf.__version__) >= version.parse("2.16.0"):
    print('Normalization layer successful!')
    normalize_input = normalize_input_keras3
else:
    print('Choose legacy normalization...')
    normalize_input = normalize_input_legacy


@K.utils.register_keras_serializable()
class NormalizeInputLayer(Layer):
    '''Normalize power of inputs to one
    axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since inputs=0, then NaN!
    '''

    def __init__(self, axis=0, eps=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.eps = eps

    def call(self, inputs):
        return normalize_input(inputs, axis=self.axis, eps=self.eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "eps": self.eps
        })
        return config


class GaussianNoise2tf26(Layer):
    """Modified GaussianNoise(Layer)
    1. to be active in evaluation and 2. to allow SNR range in training
    Can be used in Tensorflow1 and 2
    Version used in tensorflow 2.6!!!
    Input
    stddev: Standard deviation range is saved as weights to be changable in evaluation

    Original description:
    Apply additive zero-centered Gaussian noise.

    Args:
    stddev: Float, standard deviation of the noise distribution.

    Call arguments:
    inputs: Input tensor (of any rank).

    Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

    Output shape:
    Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoise2tf26, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev0 = stddev
        init = K.initializers.Constant(value=self.stddev0)
        self.stddev = self.add_weight(
            "stddev", trainable=False, shape=(2), initializer=init)
        # self.stddev = tf.Variable(name = "stddev", trainable = False, initial_value = stddev, shape = ())

    # def build(self, inputs):
    @tf.function
    def call(self, inputs):
        def noised_tf26():
            # tf.cond() # tf-alternative to switch
            stddev = KB.switch(self.stddev[0] == self.stddev[1],
                               lambda: self.stddev[0],
                               lambda: KB.exp(
                KB.random_uniform(tf.concat([array_ops.shape(inputs)[0][tf.newaxis], tf.ones(tf.shape(array_ops.shape(inputs)[1:]), dtype='int32')], axis=0),  # [array_ops.shape(inputs)[0]] + tf.ones(tf.shape(array_ops.shape(inputs))[0] - 1, dtype = tf.int16).tolist(),   # stdv only varies for each batch
                                  minval=KB.log(
                    self.stddev[0]),
                    maxval=KB.log(self.stddev[1]))
            )
            )
            output = inputs + stddev * KB.random_normal(
                shape=array_ops.shape(inputs),
                mean=0.,
                stddev=1,
                dtype=inputs.dtype)
            return output
        return noised_tf26()

    def get_config(self):
        config = {'stddev': self.stddev0}
        base_config = super(GaussianNoise2tf26, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


def noised_keras3(inputs, stddev_range):
    # if stddev_range[0] == stddev_range[1]:
    #     stddev = stddev_range[0]
    # else:
    # Dynamic shape
    shape = K.ops.shape(inputs)
    batch_size = K.ops.expand_dims(shape[0], axis=0)

    # Ones for all dims after batch
    ones_shape = K.ops.ones_like(shape[1:], dtype="int32")
    stddev_shape = K.ops.concatenate([batch_size, ones_shape], axis=0)

    # Sample log stddev uniformly from log range, then exponentiate
    log_stddev = K.random.uniform(
        shape=stddev_shape,
        minval=K.ops.log(stddev_range[0]),
        maxval=K.ops.log(stddev_range[1]),
        dtype=inputs.dtype
    )
    stddev = K.ops.exp(log_stddev)

    # Generate noise with mean=0, std=1, same shape as inputs
    noise = K.random.normal(
        shape=shape,
        mean=0.0,
        stddev=1.0,
        dtype=inputs.dtype
    )

    return inputs + stddev * noise


@tf.function
def noised_legacy(inputs, stddev_range):
    # if stddev_range[0] == stddev_range[1]:
    #     stddev = stddev_range[0]
    # else:
    # Compute shape for stddev: [batch_size, 1, 1, ...] matching inputs rank
    shape = tf.shape(inputs)
    batch_size = shape[0]
    # Ones for all dims after batch
    ones_shape = tf.ones_like(shape[1:], dtype=tf.int32)
    stddev_shape = tf.concat([[batch_size], ones_shape], axis=0)

    # Sample log stddev uniformly from log range, then exponentiate
    log_stddev = tf.random.uniform(
        shape=stddev_shape,
        minval=tf.math.log(stddev_range[0]),
        maxval=tf.math.log(stddev_range[1]),
        dtype=inputs.dtype
    )
    stddev = tf.exp(log_stddev)

    # Generate noise with mean=0, std=1, same shape as inputs
    noise = tf.random.normal(
        shape=shape,
        mean=0.0,
        stddev=1.0,
        dtype=inputs.dtype
    )

    # Add scaled noise to inputs
    return inputs + stddev * noise


if version.parse(tf.__version__) >= version.parse("2.16.0"):
    print('Keras3 Gaussian layer successful!')
    noised = noised_keras3
else:
    print('Choose legacy Gaussian layer...')
    noised = noised_legacy


@tf.function
def gaussian_noise3(inputs, stddev):
    '''Tensorflow 2 Gaussian Noise layer as function, for RL-SINFONY compatibility
    1. to be active in evaluation and 2. to allow SNR range in training
    '''
    output = noised(inputs, stddev)
    # stddev2 = KB.switch(stddev[0] == stddev[1],
    #                     lambda: stddev[0],
    #                     lambda: KB.exp(
    #     KB.random_uniform(tf.concat([tf.shape(inputs)[0][tf.newaxis], tf.ones(tf.shape(tf.shape(inputs)[1:]), dtype='int32')], axis=0),  # [array_ops.shape(inputs)[0]] + tf.ones(tf.shape(array_ops.shape(inputs))[0] - 1, dtype = tf.int16).tolist(),   # stdv only varies for each batch
    #                       minval=KB.log(stddev[0]),
    #                       maxval=KB.log(stddev[1]))
    # )
    # )
    # output = inputs + stddev2 * KB.random_normal(
    #     shape=tf.shape(inputs),
    #     mean=0.,
    #     stddev=1,
    #     dtype=inputs.dtype)
    return output


@K.utils.register_keras_serializable()
class GaussianNoise2(Layer):
    """Modified GaussianNoise(Layer) for Tenorflow >= 2.10
    1. to be active in evaluation and 2. to allow SNR range in training
    Can be used in Tensorflow1 and 2
    Input
    stddev: Standard deviation range is saved as weights to be changable in evaluation

    Original description:
    Apply additive zero-centered Gaussian noise.

    Args:
    stddev: Float, standard deviation of the noise distribution.

    Call arguments:
    inputs: Input tensor (of any rank).

    Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

    Output shape:
    Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        if isinstance(stddev, (int, float)):
            self.stddev0 = [stddev, stddev]
        else:
            self.stddev0 = list(stddev)
        init = K.initializers.Constant(value=self.stddev0)
        self.stddev = self.add_weight(
            name="stddev", trainable=False, shape=(2,), initializer=init)

    def call(self, inputs):
        return noised(inputs, self.stddev)

    def get_config(self):
        config = super().get_config()
        # Conversion to numpy array necessary for serialization
        config.update({"stddev": self.stddev0})
        return config

    # @shape_type_conversion
    # def compute_output_shape(self, input_shape):
    #     return input_shape


def test_equal(value1, value2, tol=1e-7):
    """
    Test if two values or arrays are equal within a tolerance.

    Args:
        value1: First value (scalar or numpy array).
        value2: Second value (scalar or numpy array).
        tol: Tolerance for floating-point comparison.

    Raises:
        AssertionError if values differ beyond tolerance.
    """
    import numpy as np

    if isinstance(value1, (float, int)) and isinstance(value2, (float, int)):
        assert abs(
            value1 - value2) < tol, f"Values differ: {value1} != {value2}"
    else:
        # Assume numpy arrays or array-like
        np.testing.assert_allclose(value1, value2, rtol=tol, atol=tol)


def test_gaussian_noise_layer(snr_limits=[-4, 6], experiment_size=1e9):
    '''Test the GaussianNoise2 Layer
    '''
    import my_math_operations as mops
    # snr_limits = [-20, 6]
    sigma_limits = []
    for el in snr_limits:
        sigma_limits.append(mops.snr2standard_deviation(el))
    sigma_limits = np.array(sigma_limits[::-1], np.float32)

    mean_snr = np.mean(snr_limits)
    mean_var = (sigma_limits[1]**2-sigma_limits[0]**2) / 2 / \
        (np.log(sigma_limits[1])-np.log(sigma_limits[0]))

    stddev_target_shape = [int(experiment_size)]
    inputs = np.zeros(stddev_target_shape, np.float32)

    outputs = gaussian_noise3(inputs, sigma_limits)
    mean_var_emp = np.var(outputs.numpy())

    # Test random_uniform and exp-log transform
    try:
        stddev = K.ops.exp(
            K.random.uniform(
                shape=stddev_target_shape,
                minval=K.ops.log(sigma_limits[0]),
                maxval=K.ops.log(sigma_limits[1])
            )
        )
    except (ImportError, AttributeError):
        stddev = KB.exp(KB.random_uniform(
            stddev_target_shape,
            minval=KB.log(sigma_limits[0]),
            maxval=KB.log(sigma_limits[1]))
        )

    mean_snr_emp = np.mean(10 * np.log10(1 / stddev.numpy() ** 2))
    # mean_var = np.mean(stddev.numpy() ** 2)

    # Emperical tests of output
    outputs = GaussianNoise2(sigma_limits)(inputs)
    mean_var_emp2 = np.var(outputs.numpy())

    print(mean_snr)
    print(mean_snr_emp)
    print(mean_var)
    print(mean_var_emp)
    print(mean_var_emp2)
    return test_equal(mean_snr, mean_snr_emp, tol=1e-4), test_equal([mean_var, mean_var], [mean_var_emp, mean_var_emp2], tol=1e-4)


# EOF
