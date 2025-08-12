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

from my_functions import print_time


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

# ------ Training functions - Tensorflow 1 (CMDNet Research) ---------------------


def tf_enable_gpu(num_gpu, num_cores):
    '''Select/deactivate GPU in Tensorflow 1
    num_GPU: Number of GPUs (0)
    num_cores: Number of CPU cores (8)
    '''
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': 1,
                                          'GPU': num_gpu}
                            )
    session = tf.Session(config=config)
    KB.set_session(session)
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return


class TrainingHistory():
    '''Object for recording training history
    Written in Tensorflow 1 for CMDNet research
    '''
    # Class Attribute
    name = 'Training history'
    # Initializer / Instance Attributes

    def __init__(self):
        self.epoch = []
        self.train_loss = []
        self.val_loss = []
        self.params = []
        self.add_params = []  # optional additional parameters to be saved, but immutable
        self.opt_params = []
        self.opt_config = []
        self.train_time = []
        self.total_time = 0
        self.filename = ''
        self.estop_epoch = 0
    # Instance methods

    def __call__(self, epoch, train_loss, val_loss, params, opt, time, add_params=None):
        '''Record train step values
        epoch: Epoch / iteration
        train_loss: Training loss
        val_loss: Validation loss
        params: List of tensorflow tensors
        add_params: List of numpy (!) tensors
        opt: Optimizer (keras)
        time: training and validation time
        '''
        # Save losses
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        # Convert list of tensors to list of numpy arrays
        # nplist = []
        # for el in params:
        #     nplist.append(KB.eval(el))
        # self.params.append(nplist)
        self.params.append(KB.batch_get_value(params))
        if add_params is not None:
            self.add_params.append(add_params)
        # Save state of optimizer
        self.opt_params.append(opt.get_weights())
        self.opt_config.append(opt.get_config())
        # Save time
        self.train_time.append(time)
        self.total_time = self.np_total_time()
        return self

    def sel_best_weights(self):
        '''Select best parameters according to val_loss history
        '''
        ind = np.argmin(self.val_loss)
        return self.params[ind], ind

    def set_best_weights(self, params):
        '''Set params to best parameters according to val_loss history
        params: Trainable parameters to be set
        '''
        val_params, _ = self.sel_best_weights()
        # for el, el2 in zip(params, val_params):
        #    el.assign(el2)
        KB.batch_set_value(list(zip(params, val_params)))
        return params

    def sel_weights(self, params, sel_epoch):
        '''Set params to parameters according to sel_epoch
        params: Trainable parameters to be set
        sel_epoch: epoch to be selected
        '''
        KB.batch_set_value(list(zip(params, self.params[sel_epoch])))
        return params

    def early_stopping(self, esteps):
        '''Early stopping according to val_loss history
        esteps: Number of epochs w/o improvement until early stopping
        '''
        _, ind = self.sel_best_weights()
        # Reset best epoch after early stopping for dynamic lr
        if self.estop_epoch > self.epoch[ind]:
            bepoch = self.estop_epoch
        else:
            bepoch = self.epoch[ind]
        # training epochs w/o improvement
        stepswi = np.abs(self.epoch[-1] - bepoch)
        if esteps > 0 and stepswi >= esteps:        # set estop flag if more than esteps epochs w/o improvement
            estop = True
            # track epoch when early stopping for dynamic lr
            self.estop_epoch = self.epoch[-1]
        else:
            estop = False
        return estop

    def printh(self):
        '''Prints current training status
        '''
        print_str = f"Epoch: {self.epoch[-1]}, Train Loss: {self.train_loss[-1]:.6f}, Val Loss: {self.val_loss[-1]:.6f}, Time: {self.train_time[-1]:04.2f}s, Tot. time: {print_time(self.total_time)}"
        print(print_str)
        return print_str

    def np_total_time(self):
        '''Compute total training time
        '''
        self.total_time = np.sum(self.train_time)
        return self.total_time

    def obj2dict(self):
        '''Save object data to dict
        '''
        hdict = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "params": self.params,
            "opt_params": self.opt_params,
            "opt_config": self.opt_config,
            "train_time": self.train_time,
            "total_time": self.total_time,
            "add_params": self.add_params,
        }
        return hdict

    def dict2obj(self, hdict, json=0):
        '''Import from dict to object data
        '''
        def json2par(dic_json):
            '''Converts params list of lists in json dictionary back to list of arrays
            '''
            dic = dic_json
            for key, value in dic_json.items():
                if isinstance(value, list):
                    list1 = []
                    for l in value:
                        list0 = []
                        if isinstance(l, list):
                            for el in l:
                                if isinstance(el, int):
                                    list0.append(np.int64(el))
                                else:
                                    list0.append(np.array(el))
                            list1.append(list0)
                    if list1:
                        dic[key] = list1
            return dic

        def npz2dict(npz_file):
            '''Converts npz_file object with object type arrays to dictionary with lists
            '''
            npz_file.allow_pickle = True
            hdict = {}
            for key, value in npz_file.items():
                if isinstance(value, np.ndarray):
                    hdict[key] = value.tolist()
            return hdict

        if hdict is not None:
            # If .npz-file and array, convert back to list
            if isinstance(hdict, np.lib.npyio.NpzFile):
                hdict = npz2dict(hdict)
            if json == 1:
                hdict = json2par(hdict)
            self.epoch = hdict['epoch']
            self.train_loss = hdict['train_loss']
            self.val_loss = hdict['val_loss']
            self.params = hdict['params']
            self.opt_params = hdict['opt_params']
            self.opt_config = hdict['opt_config']
            self.train_time = hdict['train_time']
            self.total_time = hdict['total_time']
            if 'add_params' in hdict:
                self.add_params = hdict['add_params']
        return self


# Data set handling

def create_batch(data, batch_size, batch_number):
    '''Create one batch for each element in dataset list
    '''
    data_batch = []
    batch_index = batch_number * batch_size
    for datum in data:
        data_batch.append(datum[batch_index:batch_index+batch_size, ...])
    return data_batch


def get_batch(data, batch_size):
    '''Feed batch data into generator
    '''
    for batch_number in range(0, len(data[0]) // batch_size):
        data_batch = create_batch(data, batch_size, batch_number)
        yield data_batch


def get_batch_dataset(train_input, train_labels, batch_size):
    '''Feed batch data into generator
    '''
    data = train_input.copy()
    data.append(train_labels)
    for batch_number in range(0, len(data[0]) // batch_size):
        data_batch = create_batch(data, batch_size, batch_number)
        input_batch = data_batch[0:-1]
        labels_batch = data_batch[-1:][0]
        yield input_batch, labels_batch


def shuffle_data(datasets):
    '''Random permutation of datasets list along first dimension
    '''
    perm = np.random.permutation(datasets[0].shape[0])
    for ii, dataset in enumerate(datasets):
        datasets[ii] = dataset[perm, ...]
    return datasets


def shuffle_dataset(input_data, labels):
    '''Shuffle a dataset consisting of input list and labels
    '''
    data = input_data.copy()
    data.append(labels)
    dataset = shuffle_data(data)
    shuffled_input = dataset[0:-1]
    shuffled_labels = dataset[-1:][0]
    return shuffled_input, shuffled_labels


def dataset_split(datasets, validation_split):
    '''Splits each dataset in a list of datasets into two parts along dim=0
    with percentage of data according to val_split
    '''
    if validation_split != 1:
        if isinstance(datasets, list):
            # List of Arrays
            datasets_train = []
            datasets_test = []
            for dataset in datasets:
                dataset_size = dataset.shape[0]
                datasets_train.append(
                    dataset[:int(dataset_size * validation_split)])
                datasets_test.append(
                    dataset[int(dataset_size * validation_split):])
        else:
            # Array
            dataset_size = datasets.shape[0]
            datasets_train = datasets[:int(dataset_size * validation_split)]
            datasets_test = datasets[int(dataset_size * validation_split):]
    else:
        datasets_train = datasets
        datasets_test = []
    return datasets_train, datasets_test


def test_get_batches():
    '''Test get_batch functions
    '''
    import semantic_communication.datasets as datasets
    train_input, train_labels, _, _ = datasets.load_dataset('mnist')
    train_input, _ = datasets.preprocess_pixels(train_input, [])
    training_batch_size = 500
    epoch = 0
    epochs = 3
    number_batches = len(train_input[0]) // training_batch_size
    number_image_inputs = len(train_input)
    for epoch in range(epochs):
        batch_number = 0
        train_input, train_labels = shuffle_dataset(
            train_input, train_labels)
        if not dimensions_same:
            break
        for batch_x, _ in get_batch_dataset(train_input, train_labels, training_batch_size):
            print_str = f'[Rx] Epoch: {epoch + 1}/{epochs}, Batch: {batch_number + 1}/{number_batches}'
            print(print_str)
            batch_number = batch_number + 1
            dimensions_same = len(batch_x) == number_image_inputs
            if not dimensions_same:
                break
    return dimensions_same

# EOF
