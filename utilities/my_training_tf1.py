#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as KB

from my_functions import print_time


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


# EOF
