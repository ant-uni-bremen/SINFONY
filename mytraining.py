#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as KB

# Only imported for Gaussiannoise2 layer
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from tensorflow.keras.layers import Layer

from myfunc import print_time


## Training and Custom Layers ------------------------------------------------------------------

# Convenience Functions

def GPU_sel(num = 0, memory_growth = 'True'):
    '''Select/deactivate GPU in Tensorflow 2
    Configure to use only a single GPU and allocate only as much memory as needed
    For more details, see https://www.tensorflow.org/guide/gpu
    '''
    if num >= 0:
        # Choose GPU
        gpus = tf.config.list_physical_devices('GPU')
        print('Number of GPUs available :', len(gpus))
        if gpus:
            gpu_num = num # Index of the GPU to use
            try:
                tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
                print('Only GPU number', gpu_num, 'used.')
                tf.config.experimental.set_memory_growth(gpus[gpu_num], memory_growth)
            except RuntimeError as e:
                print(e)
    elif num == -1:
        # Deactivate GPUs and use CPUs
        # TODO: Not yet possible to choose number of cores
        # TODO: "Visible devices cannot be modified after being initialized"
        cpus = tf.config.list_physical_devices('CPU')
        print('Number of CPUs available :', len(cpus))
        if cpus:
            cpu_num = 0 # Index of the CPU to use
            try:
                tf.config.experimental.set_visible_devices([], 'GPU')
                print('GPUs dectivated.')
                tf.config.set_visible_devices(cpus[cpu_num], 'CPU')
                print('Only CPU number', cpu_num, 'used.')
                # tf.config.experimental.set_memory_growth(cpus[cpu_num], memory_growth)
            except RuntimeError as e:
                print(e)
    else:
        print('Will choose GPU or CPU automatically.')


# Custom Layer Functions


def normalize_input(x, axis = 0, eps = 0):
	'''Normalize power of input x to one
	x: input
	axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since x=0, then NaN!
	'''
	# out = x / tf.keras.backend.sqrt(tf.keras.backend.mean(x ** 2 + eps, axis = axis, keepdims = True))
	out = x / tf.math.sqrt(tf.reduce_mean(x ** 2 + eps, axis = axis, keepdims = True))
	return out


class GaussianNoise2(Layer):
	"""Modified GaussianNoise(Layer)
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
		super(GaussianNoise2, self).__init__(**kwargs)
		self.supports_masking = True
		self.stddev0 = stddev

	def build(self, inputs):
		init = tf.keras.initializers.Constant(value = self.stddev0)
		self.stddev = self.add_weight("stddev", trainable = False, shape = (2), initializer = init)
		# self.stddev = tf.Variable(name = "stddev", trainable = False, initial_value = stddev, shape = ())
	
	@tf.function
	def call(self, inputs):
		def noised():
            # tf.cond() # tf-alternative to switch
			stddev = tf.keras.backend.switch(self.stddev[0] == self.stddev[1],
                    lambda: self.stddev[0],
                    lambda: tf.keras.backend.exp(
					tf.keras.backend.random_uniform(tf.concat([array_ops.shape(inputs)[0][tf.newaxis], tf.ones(tf.shape(array_ops.shape(inputs)[1:]), dtype = 'int32')], axis = 0), # [array_ops.shape(inputs)[0]] + tf.ones(tf.shape(array_ops.shape(inputs))[0] - 1, dtype = tf.int16).tolist(),   # stdv only varies for each batch
					minval = tf.keras.backend.log(self.stddev[0]),
					maxval = tf.keras.backend.log(self.stddev[1]))
			        )
                    )
			output = inputs + stddev * tf.keras.backend.random_normal(
				shape = array_ops.shape(inputs),
				mean = 0.,
				stddev = 1,
				dtype = inputs.dtype)
			return output
		return noised()

	def get_config(self):
		config = {'stddev': self.stddev0}
		base_config = super(GaussianNoise2, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@shape_type_conversion
	def compute_output_shape(self, input_shape):
		return input_shape



@tf.function
def GaussianNoise3(inputs, stddev): 
    '''Tensorflow 2 Gaussian Noise layer as function, for RL-SINFONY compatibility
    1. to be active in evaluation and 2. to allow SNR range in training
    '''
    stddev2 = tf.keras.backend.switch(stddev[0] == stddev[1],
            lambda: stddev[0],
            lambda: tf.keras.backend.exp(
            tf.keras.backend.random_uniform(tf.concat([tf.shape(inputs)[0][tf.newaxis], tf.ones(tf.shape(tf.shape(inputs)[1:]), dtype = 'int32')], axis = 0), # [array_ops.shape(inputs)[0]] + tf.ones(tf.shape(array_ops.shape(inputs))[0] - 1, dtype = tf.int16).tolist(),   # stdv only varies for each batch
            minval = tf.keras.backend.log(stddev[0]),
            maxval = tf.keras.backend.log(stddev[1]))
            )
            )
    output = inputs + stddev2 * tf.keras.backend.random_normal(
        shape = tf.shape(inputs),
        mean = 0.,
        stddev = 1,
        dtype = inputs.dtype)
    return output




## ------ Training functions - Tensorflow 1 (CMDNet Research) ---------------------


def tf_enable_GPU(num_GPU, num_cores):
    '''Select/deactivate GPU in Tensorflow 1
    num_GPU: Number of GPUs (0)
    num_cores: Number of CPU cores (8)
    '''
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
    inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
    device_count = {'CPU' : 1, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    return KB.set_session(session)


class training_history():
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
        self.add_params = [] # optional additional parameters to be saved, but immutable
        self.opt_params = []
        self.opt_config = []
        self.train_time = []
        self.total_time = 0
        self.filename = ''
        self.estop_epoch = 0
    # Instance methods
    def __call__(self, epoch, train_loss, val_loss, params, opt, time, add_params = None):
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
        if add_params != None:
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
        #for el, el2 in zip(params, val_params):
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
        stepswi = np.abs(self.epoch[-1] - bepoch)   # training epochs w/o improvement
        if esteps > 0 and stepswi >= esteps:        # set estop flag if more than esteps epochs w/o improvement
            estop = True
            self.estop_epoch = self.epoch[-1]       # track epoch when early stopping for dynamic lr
        else:
            estop = False
        return estop
    def printh(self):
        '''Prints current training status
        '''
        print_str = "Epoch: {}, Train Loss: {:.6f}, Val Loss: {:.6f}, Time: {:04.2f}s, Tot. time: ".format(self.epoch[-1], self.train_loss[-1], self.val_loss[-1], self.train_time[-1]) + print_time(self.total_time)
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
    def dict2obj(self, hdict, json = 0):
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
                    if list1 != []:
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
        
        if hdict != None:
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


def create_batch(batch, batch_size, data):
    '''Create batches of data set
    '''
    data_batch = []
    # for ii, datum in enumerate(data):
    for datum in data:
        data_batch.append(datum[batch * batch_size : (batch + 1) * batch_size])
    return data_batch

def permute_data(data):
    '''Random permutation of data along first dimension
    '''
    perm = np.random.permutation(data[0].shape[0])
    for ii, datum in enumerate(data):
        data[ii] = datum[perm]
    return data

def data_split(data_list, val_split):
    '''Splits array y of list data_list into two parts along dim=0 with percentage of data according to val_split
    '''
    if val_split != 1:
        if isinstance(data_list, list):
        # List of Arrays
            ytrain = []
            ytest = []
            for y in data_list:
                ytrain.append(y[:int(y.shape[0] * val_split)])
                ytest.append(y[int(y.shape[0] * val_split):])
        else:
        # Array
            ytrain = data_list[:int(y.shape[0] * val_split)]
            ytest = data_list[int(y.shape[0] * val_split):]
    else:
        ytrain = data_list
        ytest = []
    return ytrain, ytest


# EOF