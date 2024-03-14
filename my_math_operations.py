#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np

# Basic Mathematical Operations


def np_softmax(arg, axis=-1):
    '''Accurate softmax implementation in numpy
    '''
    exp_arg = np.exp(arg - np.max(arg, axis=axis, keepdims=True))
    softmax = exp_arg / np.expand_dims(np.sum(exp_arg, axis=axis), axis=axis)
    return softmax


def sigmoid(x):
    '''sigmoid function
    '''
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    '''derivative of sigmoid function
    '''
    return sigmoid(x) * (1 - sigmoid(x))  # np.exp(-x) / (1 + np.exp(-x)) ** 2 # exact form


def batch_dot(a, b):
    '''Computes the
    matrix vector product: A*b
    vector matrix product: a*B
    matrix product: A*B
    for a batch of matrices and vectors along dimension 0
    Shape of tensors decides operation
    '''
    if len(a.shape) == 3 and len(b.shape) == 2:
        y = np.einsum('nij,nj->ni', a, b)  # A*b
    elif len(a.shape) == 2 and len(b.shape) == 3:
        y = np.einsum('nj,nji->ni', a, b)  # b*A
    elif len(a.shape) == 3 and len(b.shape) == 3:
        y = np.einsum('nij,njk->nik', a, b)  # A*B
    return y


def matim2re(x, mode=1):
    '''Converts imaginary vector/matrix to real
    '''
    if mode == 1:  # matrix conversion
        if len(x.shape) == 3:
            x = np.concatenate((np.concatenate((np.real(x), np.imag(x)), axis=1), np.concatenate(
                (-np.imag(x), np.real(x)), axis=1)), axis=-1)
        else:
            x = np.concatenate((np.concatenate((np.real(x), np.imag(x))), np.concatenate(
                (-np.imag(x), np.real(x)))), axis=1)
    else:  # vector conversion
        x = np.concatenate((np.real(x), np.imag(x)), axis=1)
    return x


def tvec2diag(x):
    '''Creates tensor with diagonal matrices diagx out of tensor of vectors x
    '''
    diagt = np.zeros((x.shape[-1], x.shape[-1])) + \
        np.expand_dims(np.eye(x.shape[-1]), 0)
    diagx = x[:, np.newaxis] * diagt
    # di = np.diag_indices(x.shape[-1])
    # diagx = np.zeros((x.shape[0], x.shape[-1], x.shape[-1]))# + np.expand_dims(np.eye(x.shape[-1]), 0)
    # diagx[:, di[0], di[1]] = x
    return diagx


def tdiag2vec(A):
    '''Extracts diagonal elements diag(A) from batch of matrices A and writes to batch of vectors x
    '''
    x = np.diagonal(A, axis1=1, axis2=-1).copy()
    return x


def dbinv(in_x):
    '''Converts SNR in dB back to normal scale
    '''
    return 10 ** (in_x / 10)


def csigma(in_x):
    '''Converts SNR in dB to standard deviation sigma
    Only for backward compatability
    '''
    return np.sqrt(1 / dbinv(in_x))


def snr2standard_deviation(snr):
    '''Converts SNR in dB to standard deviation sigma
    '''
    return csigma(snr)


def snr_range2snrlist(snr_range=[-30, 20], snr_step_size=1):
    '''Computes SNR array from given SNR range and SNR step size
    '''
    snr = np.linspace(snr_range[0], snr_range[1], int(
        (snr_range[1] - snr_range[0]) / snr_step_size) + 1)
    return snr


def int2bin(x, N):
    '''Convert a positive integer num into an N-bit bit vector
    Limited up to N = 64 bits and 2 ** 64 numbers (!!!)
    '''
    return np.unpackbits(np.reshape(x, (-1, 1)).astype(np.uint64).byteswap().view(np.uint8), axis=1)[:, -N:]


def bin2int(b, axis=-1, dtype='uint64'):
    '''Convert a N-bit bit vector [b] across dimension [axis] into positive integer num [cl]
    Maximum integer number with uint64 is 9223372036854775807
    '''
    b_power = 2 ** np.arange(b.shape[axis], dtype=dtype)[::-1]
    dims = np.ones(len(b.shape), dtype=dtype).tolist()
    dims[axis] = b.shape[axis]
    cl = np.sum(b * np.reshape(b_power, dims), axis=axis).astype(dtype)
    return cl


# EOF
