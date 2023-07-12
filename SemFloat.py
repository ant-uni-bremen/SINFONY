#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: beck
Simulation framework for numerical results of the articles:
1. E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022
2. E. Beck, B.-S. Shin, S. Wang, T. Wiedemann, D. Shutin, and A. Dekorsy,
“Swarm Exploration and Communications: A First Step towards Mutually-Aware Integration by Probabilistic Learning,”
Electronics, vol. 12, no. 8, p. 1908, Apr. 2023
"""

# LOADED PACKAGES
# Tensorflow packages
import tensorflow as tf
from tensorflow.keras import backend as KB
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam, Nadam
# Keras layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Add, Lambda, Concatenate, Layer, Concatenate, GaussianNoise#, Activation
from tensorflow.keras.callbacks import EarlyStopping
# Python packages
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import tikzplotlib as tplt
# Own packages
import sys
sys.path.append('..')	# Include parent folder, where own packages are
sys.path.append('.')	# Include current folder, where start simulation script and packages are
import mytraining as mt
import myfloat as mfl
import mymathops as mop
import mycom as com
import myfunc as mf


## Functions exclusive to this file

# System model: Floating point source + Bit transmission

class float_dataset_gen():
    '''Data object for floating point data set generation
    data = [y, sigma, s, b]
    '''
    # Class Attribute
    name = 'Floating point dataset generator'
    # Initializer / Instance Attributes
    def __init__(self, Nb, floatx, mod, snr_min, snr_max, p_s, huffman = 0):
        '''Input -----------------------------------------
        Nb: Batch size
        floatx: Floating point object
        mod: Modulation object
        snr_min: Minimum SNR
        snr_max: Maximum SNR
        p_s: Probabilities of floating point data
        huffman: Huffman source coding object
        '''
        # Inputs
        self.N_batch = Nb
        self.floatx = floatx
        self.mod = mod
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p_s = p_s
        self.huffman = huffman
        # Outputs
        self.y = 0
        self.sigma = 0
        self.data = []
    # Instance methods
    def __call__(self, Nb = None):
        return self.gen_data(Nb)
    def gen_data(self, Nb = None):
        '''Generates floating point data in AWGN
        Input -----------------------------------------
        Nb: Batch size (optional)
        Output ----------------------------------------
        data = [y, sigma, s, b]
        y: Received signal vector
        sigma: Standard deviation vector of AWGN
        s: Floating point values
        b: Transmitted bit representation of s
        '''
        if Nb == None:
            Nb = self.N_batch
        [y, sigma, s, b] = float_model(Nb, self.p_s, self.floatx, self.mod, self.snr_min, self.snr_max, self.huffman)
        self.s = s
        self.b = b  
        self.sigma = sigma
        self.y = y
        self.data = [self.y, self.sigma, self.s, self.b]
        return self.data


def float_source(Nb, p_s, floatx):
    '''Generates floating point data distributed according to p_s
    Input -----------------------------------------
    Nb: Batch size
    floatx: Floating point object
    p_s: A priori probabilities
	Output ----------------------------------------
    s: Floating point values
    b: Transmitted bit representation of s
    '''
    # Floating point data generation
    if floatx.float_name == 'float32' or floatx.float_name == 'float64':
        # float32/64 far too complex for the usual procedure (below)
        if p_s[0] == 0:
            # Gaussian distribution
            _, s = floatx.float2bitint(np.random.normal(p_s[1], np.sqrt(p_s[2]), (Nb)))
        elif p_s[0] == 1:
            # Uniform continuous distribution
            _, s = floatx.float2bitint(np.random.uniform(p_s[1], p_s[2], (Nb)))
        else:
            # Uniform discrete distribution
			# Both isfinite sets for +/- are of equal size
            bint = np.random.randint(0, 2 ** (floatx.N_bits - 1) - 2 ** floatx.N_sig, Nb)
            indrand = np.random.randint(0, 1 + 1, Nb)
            bint[indrand == 1] = np.random.randint(2 ** (floatx.N_bits - 1), 2 ** floatx.N_bits - 2 ** floatx.N_sig, bint[indrand == 1].shape)
            s = floatx.bitint2float(bint)
    else:
        # Usual random number generation with s_poss and p_s
        s_poss = floatx.x_poss
        s = np.random.choice(s_poss, p = p_s, size = (Nb))
    b = floatx.float2bit(s)
    return s, b

def semantic_channel(b, mod, snr_min, snr_max):
    '''Simulates simple transmission system with modulation of bits and transmission over AWGN channel
    Semantic channel for semantic information
    INPUT
    b: Bit array
    mod: Modulation object
    snr_min: Minimum SNR value of the AWGN channel
    snr_max: Maximum SNR value
    OUTPUT
    y: Received signal
    sigma: Standard deviations of AWGN channel
	'''
    # Modulation
    x = mod.modulate(b[..., np.newaxis], axis = -1)
    # Channel
    sigma = mop.csigma(np.random.uniform(snr_min, snr_max, x.shape[0]))[..., np.newaxis]
    y = com.awgn(x, np.repeat(sigma, x.shape[-1], axis = -1), compl = 0)
    return y, sigma

def float_model(Nb, p_s, floatx, mod, snr_min, snr_max, huffman = 0):
	'''Generates p_s distributed floating point data and simulates digital transmission over AWGN channel (with source encoding)
	Input -----------------------------------------
	Nb: Batch size
	s_var: Variance of Gaussian distributed floating point values s
	floatx: Floating point object
	huffman: Huffman code object (0: default, source coding turned off)
	mod: Modulation object
	snr_min: Minimum SNR
	snr_max: Maximum SNR
	Output ----------------------------------------
	data = [y, sigma, s, b]
	y: Received signal vector
	sigma: Standard deviation vector of AWGN
	s: Floating point values
	b: Transmitted bit representation of s
	'''
	# Floating point data generation
	s, b = float_source(Nb, p_s, floatx)
	if huffman != 0:
        # Include Huffman source coding
		sint, _ = floatx.float2bitint(s)
		bl, _ = huffman.encoding(sint)
		b = np.array(bl, dtype = 'bool')[..., np.newaxis]
	y, sigma = semantic_channel(b, mod, snr_min, snr_max)
	return y, sigma, s, b


## PRIOR probabilities for different data --------------------------------

def gaussian(x, mu, sigma):
	'''Gaussian function
    INPUT
	x: Function is evaluated at point x
    mu: Expected value
    sigma: Standard deviation
	OUTPUT
	Gaussian function values
	'''
	return 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.) / 2)

def compute_prior(floatx, mode = 0, fmode = True):
	'''Compute prior probability p_s of any bit sequence s and p_ba of any bit, if floating point value is
	INPUT
	floatx: Floatint point object
	mode = 0: Gaussian distributed
	mode = 1: Uniformly distributed (continuous)
    mode = 2: Data-based prior from DLR seismic exploration
	mode = 3: For convenience: Both data sets from DLR merged
	mode = 4: Data-based prior from SINFONY article
	mode = 5: Just for debug purposes: All floating point values have same probability of occurence (uniform discrete distribution)
	fmode: True - Adapt prior generation with higher resolution floating point numbers
	OUTPUT
	p_ba: Probabilities of each single bit individually
	p_s: Probabilities of all sequences
	'''
	# Adapt prior to floating point number resolution due to computational intractability
	if fmode == True and (floatx.float_name == 'float32' or floatx.float_name == 'float64'):
		# Define distribution parameters to use built-in distribution samplers for high-resolution floating point numbers
		if mode == 0:
			# Gaussian with mean 0 and variance 1
			p_s = np.array([mode, 0, 1])
		elif mode == 1:
			# Uniform continuous distribution
			floatmax = 2 ** floatx.N_bits - 2 ** (floatx.N_sig + 1)
			p_s = np.array([mode, -floatmax, floatmax])
		else:
			# Uniform discrete distribution
			p_s = np.array([mode, 2 ** (floatx.N_bits - 1) - 2 ** floatx.N_sig, 2 ** floatx.N_bits - 2 ** floatx.N_sig])
		# Set prior of each bit to uniform -> exact computation intractable
		p_ba = 0.5 * np.ones((floatx.N_bits, 2))
	else:
		# Default: Define probability for each sequence and sample it
		if mode == 0:
			# Gaussian distribution
			p_s = sequence_prior_gaussian(floatx)
		elif mode == 1:
			# Uniform continuous distribution
			p_s = sequence_prior_uniform(floatx)
		elif mode == 2:
			# Data-based prior from DLR seismic exploration
			p_s = mfl.sequence_prior_data(floatx, load_data_explseismic(floatx.N_bits, mode = 1)) # choose dataset/path with mode
		elif mode == 3:
			# For convenience: Both data sets from DLR merged
			p_s = mfl.sequence_prior_data(floatx, load_data_explseismic(floatx.N_bits, mode = 0))
		elif mode == 4:
			# Data-based prior from SINFONY article
			p_s = mfl.sequence_prior_data(floatx, load_data_semcom())
		else:
			# Just for debug purposes: All floating point values have same probability of occurence (uniform discrete distribution)
			p_s = sequence_prior_uniform_float(floatx)
		# A piori probability for each bit b separately
		p_ba = mfl.compute_single_bitprob(floatx, p_s)
	return p_ba, p_s


def sequence_prior_uniform_float(floatx):
    '''Compute prior of bit sequence for uniform distributed s_poss (all floating point values have same probability of occurence)
    [For debugging purposes]
    INPUT
    floatx: Floating point object -> possible floating point values / bit sequences
    OUTPUT
    p_s: Prior probability of sequences
    '''
    s_poss = floatx.x_poss
    # Note: +/-inf or NaN are not transmitted, exclude from possibilities
    isfin = np.isfinite(s_poss)
    p_s = np.zeros(s_poss.shape, dtype = 'float64')
    p_s[isfin] = p_s[isfin] + 1 / isfin.shape[0]
    p_s = p_s / np.sum(p_s)
    return p_s

def sequence_prior_uniform(floatx):
    '''Probabilities p(s) of bit sequences or s computed via numerical integration of continuous uniform distribution
    INPUT
    floatx: Floating point object
    OUTPUT
    p_s: Prior probability of sequences
	'''
    # Integrated probabilities of an uniform distribution discretized to floatx resolution
    s_poss = floatx.x_poss
    isfin = np.isfinite(s_poss)
    s_poss_fin = s_poss[isfin]
    s_int = integral_rec(s_poss_fin, np.ones(s_poss_fin.shape, dtype = np.int8))
    p_sint = s_int / np.sum(s_int)
    p_s = np.zeros(s_poss.shape[0])
    p_s[isfin] = p_sint
    return p_s

def sequence_prior_gaussian(floatx, s_mean = 0, s_var = 1, mode = 0):
    '''Probabilities p(s) of bit sequences or s computed via numerical integration (trapezoid rule) of continuous Gaussian distribution
    INPUT
    floatx: Floating point object
    s_mean: Mean of Gaussian distribution p(s)
    s_var:  Variance of Gaussian distribution p(s)
    mode: [0: Trapez rule / 1: Rectangle approx]
    OUTPUT
    p_s: Probabilities for each possible finite floating point value
    '''
    # Integrated probabilities of a Gaussian distribution discretized to floatx resolution
    s_poss = floatx.x_poss
    isfin = np.isfinite(s_poss)
    s_poss_fin = s_poss[isfin]
    f_s = gaussian(s_poss_fin.astype('float32'), s_mean, s_var)
    if mode == 0:
        # Tectangle approx.
        s_int = integral_rec(s_poss_fin, f_s)
    else:
        # Trapezoid rule
        s_int = integral_trapz(s_poss_fin, f_s)
    p_sint = s_int / np.sum(s_int)
    p_s = np.zeros(s_poss.shape[0])
    p_s[isfin] = p_sint
    return p_s

def load_data_explseismic(float_Nbits, mode = 0):
	'''Load dataset of seismic exploration from DLR in Oberpfaffenhofen
	Provided by Ban-Sok Shin (ban-sok.shin@dlr.de)
    INPUT
	float_Nbits: Number of floating point bits / resolution
	mode: Choose data set 0 (gradients, models), 1 (gradients), 2 (models)
	OUTPUT
	data: Loaded data
	'''
	# Exploration data set: Gradients + Models
	# Only given for one iteration of the seismic exploration algorithm
	# Flatten 3D data set with grid index (x, z) and 20 geophones/agents (3D)
	# Load datasets
	path = os.path.join('Datasets', 'seismic_exploration_DLR')
	pathfile = os.path.join(path, 'gradients.npy')
	data1 = np.load(pathfile).flatten()
	pathfile = os.path.join(path, 'models.npy')
	data2 = np.load(pathfile).flatten()
	if float_Nbits < 16:
        # Delete mean of models to make data values smaller for low-resolution floating point values (< 16 bit)
		data2 = data2 - np.mean(data2)
	if mode == 0:
        # Gradients + models
		data = np.concatenate((data1, data2))
	elif mode == 1:
		# Gradients
		data = data1
	elif mode == 2:
		# Models
		data = data2
	else:
		print('Error: Dataset not available')
	return data

def load_data_semcom(flatten = True, train = True):
	'''Loaded dataset is encoder output of semantic communication system SINFONY - Perfect Com.
	INPUT
	flatten: Flatten the multidimensional output of the multiple encoders 
	train: Use the training data to produce the encoder output
    OUTPUT
	data_out: Loaded training data
	'''
	# Load model and reevaluate
	filename = 'ResNet14_MNIST2'	# ResNet14_MNIST#N, ResNet20_CIFAR#N
	path = os.path.join('models_sem')
	pathfile = os.path.join(path, filename)
	# Loading the model back:
	print('Loading model ' + filename + '...')
	model = tf.keras.models.load_model(pathfile)
	print('Model loaded.')
	# Data set
	dataset = 'mnist'				# mnist, cifar10, fashion_mnist
	import SemCom.SINFONY as sc
	trainx, _, testx, _ = sc.load_dataset(dataset)
	train_norm, test_norm = sc.prep_pixels(trainx, testx)
	if train == True:
		data = model.layers[1].predict(train_norm) # .numpy() # for total test set
	else:
		data = model.layers[1].predict(test_norm)
	if flatten == True:
		data_out = data.flatten()
	else:
		data_out = data
	return data_out

def integral_rec(s, f_s):
    '''Integral approximation of f(s) with sampling points s according to rectangle rule
    INPUT
    s: Sampling points
    f_s: Function f(s) evaluated for s
    OUTPUT
    s_int: Integral of f(s) between two sampling point borders = rectangle approximation
    '''
    # Ordering
    indord = np.argsort(s)
    s_ord = s[indord]
    # Rectangle approx.
    interv = np.diff(s_ord)
    interv2 = np.zeros(s_ord.shape[0])
    interv2[1:-1] = interv[:-1] / 2 + interv[1:] / 2
    interv2[0] = interv[0] / 2
    interv2[-1] = interv[-1] / 2
    s_int_ord = interv2 * f_s[indord]
    # Reordering
    indord2 = np.argmax(np.arange(0, s_ord.shape[0])[:, np.newaxis] == indord[np.newaxis, :], axis = -1)
    s_int = s_int_ord[indord2]
    return s_int


def integral_trapz(s, f_s):
    '''Integral approximation of f(s) with sampling points s according to trapezoid rule
    INPUT
    s: Sampling points
    f_s: Function f(s) evaluated for s
    OUTPUT
    s_int: Integral approximation of f(s) according to trapezoid rule
    '''
    # Ordering
    indord = np.argsort(s)
    s_ord = s[indord]
    # Trapezoid rule
    trap = np.diff(s_ord) * (f_s[indord][1:] + f_s[indord][:-1]) * 0.5
    s_int_ord = np.zeros(s_ord.shape[0])
    s_int_ord[1:-1] = trap[:-1] / 2 + trap[1:] / 2
    s_int_ord[0] = trap[0] / 2
    s_int_ord[-1] = trap[-1] / 2
    # Reordering
    indord2 = np.argmax(np.arange(0, s_ord.shape[0])[:, np.newaxis] == indord[np.newaxis, :], axis = -1)
    s_int = s_int_ord[indord2]
    return s_int


def sequence_prior_sampled_gaussian(floatx, N_s = 10 ** 9, s_mean = 0, s_var = 1, step = 100000):
    '''Probabilities p(s) of bit sequences of Gaussian distributed s computed via Sampling
    [For evaluation of manual implementation]
    INPUT
    floatx: Floating point object
    N_s: Total sample size
    s_mean: Mean of Gaussian distribution
    s_var: Variance of Gaussian distributed floating point values s
    step: Sample size per iteration
    OUTPUT
    p_s: Sequence probabilities approximated via sampling
    '''
    p_s = np.zeros(2 ** floatx.N_bits)
    for _ in range(0, int(N_s / step)):
        bint, _ = floatx.float2bitint(np.random.normal(s_mean, np.sqrt(s_var), (step)))
        b_count = np.bincount(bint)
        p_s[0:b_count.shape[0]] = p_s[0:b_count.shape[0]] + b_count
    p_s = p_s / np.sum(p_s)
    return p_s

def sequence_prior_sampled_uniform(floatx, N_s = 10 ** 9, step = 100000):
    '''Probabilities p(s) of bit sequences of uniform distributed s computed via Sampling
    [For evaluation of manual implementation]
    INPUT
    floatx: Floating point object
    N_s: Total sample size
    step: Sample size per iteration
    OUTPUT
    p_s: Sequence probabilities approximated via sampling
    '''
    p_s = np.zeros(2 ** floatx.N_bits)
    s_max = (1 - 2 ** (- floatx.N_sig - 1)) * 2 ** (2 ** floatx.N_exp - 1 - floatx.expbias)
    for _ in range(0, int(N_s / step)):
        bint, _ = floatx.float2bitint(np.random.uniform(-s_max, s_max, (step)))
        b_count = np.bincount(bint)
        p_s[0:b_count.shape[0]] = p_s[0:b_count.shape[0]] + b_count
    p_s = p_s / np.sum(p_s)
    return p_s

def sequence_prior_comparison(floatx, N_s = 10 ** 9, tikzplt = 0):
	'''Comparison of sampled and computed sequence prior
	[Evaluation of manual implementation]
	INPUT
	floatx: Floating point object
	N_s: Sample size
	tikzplt: Save in tikz picture with [1]
	OUTPUT
	p_su: Manual uniform prior
	p_su2: Sampled uniform prior
	p_s: Manual Gaussian prior
	p_s2: Sampled Gaussian prior
	'''
	s_poss = floatx.x_poss

	# Plot Uniform
	p_su = sequence_prior_uniform(floatx)
	p_su2 = sequence_prior_sampled_uniform(floatx, N_s)
	plt.figure(1)
	plt.plot(s_poss, p_su, 'r-o', label = 'p(s)')
	plt.plot(s_poss, p_su2, 'b-x', label = 'p(s) sampled')
	plt.legend()
	# Tikz plot
	if tikzplt == 1:
		tplt.save("plots/comsem_sequence_prior_uniform.tikz")

	# Plot Gaussian
	p_s = sequence_prior_gaussian(floatx, mode = 0) # mode 0/1
	p_s2 = sequence_prior_sampled_gaussian(floatx, N_s)
	plt.figure(2)
	plt.plot(s_poss, p_s, 'r-o', label = 'p(s)')
	plt.plot(s_poss, p_s2, 'b-x', label = 'p(s) sampled')
	# plt.plot(s_poss, gaussian(s_poss, 0, 1), 'g-', label = 'Gaussian')
	# plt.xlim(-3, 3)
	# plt.ylim(0, 5 * 10 ** -6)
	plt.legend()
	# TikZ plot
	if tikzplt == 1:
		## Preprocessing for float16, since 2 ** 16 data points are too much for visualization
		if floatx.N_bits >= 16:
			x_lim = 5
			s_possl5 = s_poss[np.abs(s_poss) <= x_lim]
			p_sl5 = p_s[np.abs(s_poss) <= x_lim]
			p_s_list = []
			s_poss_list = []
			for ii, p_s0 in enumerate(p_s):
				if ii + 1 < len(p_s) and np.abs(p_s0 - p_s[ii + 1]) / p_s0 > 0.1:
					p_s_list.append(p_s0)
					s_poss_list.append(s_poss[ii])
			p_s_list = p_s_list + p_sl5[::20].tolist()
			s_poss_list = s_poss_list + s_possl5[::20].tolist()
			plt.figure(2)
			plt.plot(s_poss_list, p_s_list, 'r-o', label = 'p(s)')
			# plt.plot(s_poss, p_s, 'b-x', label = 'p(s) full')
			plt.xlim(-x_lim, x_lim)
			plt.legend()
		tplt.save("plots/comsem_sequence_prior_gaussian.tikz")    
	return p_su, p_su2, p_s, p_s2



## Estimators/Posteriors -----------------------------------------------


def sequence_est(p_s, y, sigma, mod, floatx, step = 100, mode = 0, tikzplt = 0, it_print = 0):
	'''Maximum Likelihood Sequence Estimator
	INPUT
	p_s: Probability of sequence s
	y: Received signal
	sigma: Noise standard deviation
	mod: Modulation object
	floatx: Floating point object
	step: Batch of sequences to process (home: 100, work: 1000)
	mode: MAP (1) / Mean (2) estimation or both (0)
	tikzplt: Plot posterior of index indx
	it_print: Print each it_print finished iteration
	OUTPUT
	s_r: Most likely transmitted floating point values
	b_r: Most likely transmitted bit sequences
	s_mean: Mean estimate of transmitted floating point values (only MAP mode)
	s_var: Variance of mean estimate s_mean (only MAP mode)
	'''
	# Note: +/-inf or NaN are not transmitted, exclude from possibilities
	# -> Otherwise Mean estimator leads to NaNs *(1)
	isfin = np.isfinite(floatx.x_poss)
	b_poss = floatx.b_poss[isfin]
	s_poss = floatx.x_poss[isfin]
	p_s = p_s[isfin]
	isfin = 0

	b_poss2 = mop.int2bin(b_poss, floatx.N_bits)
	x = mod.modulate(b_poss2[..., np.newaxis], axis = -1)

	# Initialize output tensors
	s_r = np.zeros(y.shape[0])
	s_mean = np.zeros(y.shape[0])
	s_var = np.zeros(y.shape[0])
	b_r = np.zeros((y.shape[0], b_poss2.shape[-1]))
	# Divide y into batches due to extensive computation
	for indj in range(0, int(y.shape[0] / step)):
		yj = y[indj * step:(indj + 1) * step]
		sigmaj = sigma[indj * step:(indj + 1) * step, :]
		norm2 = np.sum((yj[:, np.newaxis, :] - x.reshape((1, x.shape[0], x.shape[1]))) ** 2, axis = -1)
		arg = - 1 / 2 / (sigmaj ** 2) * norm2 + np.log(p_s[np.newaxis, :])
		if mode == 0 or mode == 1:
			# MAP estimator, alternative: indmax = np.argmax(p_sr, axis = -1)
			indmax = np.argmax(arg, axis = -1)
			s_r[indj * step:(indj + 1) * step] = s_poss[indmax]
			b_r[indj * step:(indj + 1) * step, :] = b_poss2[indmax, :]
		if mode == 0 or mode == 2:
			# Mean estimator
			p_sr = mop.np_softmax(arg)
			# s_mean does not have to be of same data type/resolution since it is averaged across possible values
			# s_mean[indj * step:(indj + 1) * step] = np.sum(p_sr[:, np.isfinite(s_poss)] * s_poss[np.newaxis, np.isfinite(s_poss)], axis = -1) # Mean estimator *(1)
			s_mean[indj * step:(indj + 1) * step] = np.sum(p_sr * s_poss[np.newaxis, :], axis = -1)
			b_r[indj * step:(indj + 1) * step, :] = floatx.float2bit(s_mean[indj * step:(indj + 1) * step])
			s_var[indj * step:(indj + 1) * step] = np.sum(p_sr * (s_poss[np.newaxis, :] - s_mean[indj * step:(indj + 1) * step, np.newaxis]) ** 2, axis = -1)
			# -------- Plot the posterior -------------------------------------------------------
			if tikzplt == 1 and indj == 0:
				indx = 0
				plot_posterior(s_poss, p_sr[indx, :], s_mean[indx], s_var[indx], x_lim = 4)
		if (it_print != 0) and (ii % it_print == 0):
			print('Estimator it.:' + str(indj))
	return s_r, b_r, s_mean, s_var


def plot_posterior(s_poss_fin, p_sr, s_mean, s_var, x_lim = 4):
    '''Plot the posterior pmf p_sr with its continuous M-projection onto Gaussian pdf (VI)
    INPUT
    s_poss_fin: Possible finite floats s
    p_sr: True posterior probabilities for each floating point value
    s_mean: Expected value of p_sr
    s_var: Variance of p_sr
    x_lim: Limits of x-axis
    '''
    ## Plot variational M projection
    # [::-1] is for correct -0/+0 ordering
    ind_ord = np.argsort(s_poss_fin[::-1])
    s_poss_ord = s_poss_fin[::-1][ind_ord]
    q_sr = gaussian(s_poss_ord, s_mean, s_var)
    # Transform pmf into step-wise continuous pdf
    s_int_reord = integral_rec(s_poss_fin, np.ones(s_poss_fin.shape, dtype = np.int8))
    p_srcon = p_sr[::-1][ind_ord] / s_int_reord[::-1][ind_ord]
    # Preprocessing for >=float16, since 2 ** 16 data points are too much
    if floatx.N_bits >= 16:
        s_possl = s_poss_fin[np.abs(s_poss_fin) <= x_lim]
        s_poss_ordl = s_poss_ord[np.abs(s_poss_ord) <= x_lim]
        p_srl = p_sr[np.abs(s_poss_fin) <= x_lim]
        s_possl2 = s_possl[p_srl >= 0.01 * np.max(p_srl)]
        p_srl2 = p_srl[p_srl >= 0.01 * np.max(p_srl)]
    # Plots are normalized to one at max, since pmf is compared to pdf
    plt.figure(1)
    plt.stem(s_poss_fin, p_sr / np.max(p_sr), 'b-x', markerfmt='bx', basefmt=" ", label = r'$p(\tilde{s}|\mathbf{y})$')
    plt.plot(s_poss_ord, p_srcon / np.max(p_srcon), 'r-o', label = r'$p(s|\mathbf{y})$ step-wise')
    plt.plot(s_poss_ord, q_sr / np.max(q_sr), 'k-', label = r'$q(s|\mathbf{y})$')
    plt.xlim(-x_lim, x_lim)
    plt.xlabel(r"$\tilde{s}$")
    plt.ylabel(r"$p(\tilde{s}|\mathbf{y})$")
    plt.legend()
    tplt.save("plots/comsem_posterior_example.tikz")
    return


## AEs/DNNs/Training functions -----------------------------------------

def data_generator(data_gen, outbin = 0, llr = 0, hmode = 0):
    '''Data generator for Tensorflow learning with signal s
    INPUT
    data_gen: Model data generator
    outbin: Select binary output data b or signal s
    llr: Select llr input data
    hmode: Select heteroscedastic model mode
    OUTPUT
    y, data[ind]: Received signal and chosen output data b or s either as list or tuple
    '''
    if outbin == 0:
        # signal: data[2]
        ind = 2
    else:
        # bits: data[3]
        ind = 3
    if hmode == 1:
		# If heteroscedastic, output a list instead of tuple
        while True:
            data = data_gen()
            y = dnn_input(data[0], data[1], mode = llr)
            yield [y, data[ind]]
    else:
        while True:
            data = data_gen()
            y = dnn_input(data[0], data[1], mode = llr)
            yield (y, data[ind])


def dnn_input(y, sigma, mode = 0):
    '''Compute DNN input
    INPUT
    y: Received signal
    sigma: Noise standard deviation
    mode: (0) Received signal, (1) LLR, (2) y scaled by sigma
    OUTPUT
    inputDNN: Input for DNN
    '''
    if mode == 1:
        # LLR: 2 * y / sigma ** 2 -> * 2 not necessary, can be learned by DNN / only scales input
        inputDNN = y / sigma ** 2
    elif mode == 2:
        # Alternative option: Sigma scaling
        inputDNN = y / sigma
    else:
        # Raw channel output/receiver input y
        inputDNN = y
    return inputDNN


def data_generator_ae(data_gen, input_shape = 1, hmode = 0):
    '''Data generator for tensorflow learning with signal s for AE
    INPUT
	data_gen: Model data generator
    input_shape: Input shape
    hmode: Select heteroscedastic model mode
    OUTPUT
    data[ind], data[2]: Chosen input data b or s and output data s either as list or tuple
    '''
    if input_shape == 1:
        # signal: data[2]
        ind = 2
    else:
        # bits: data[3]
        ind = 3
    if hmode == 1:
        # If heteroscedastic, output a list instead of tuple
        while True:
            data = data_gen()
            yield [data[ind], data[2]]
    else:
        while True:
            data = data_gen()
            yield (data[ind], data[2])



def ml_receiver_bits(M, n):
    '''Machine learning receiver directly estimating bits
    INPUT
	m: Layer width
    n: Number of channel uses
    OUTPUT
	rx: Receiver model
    '''
    inputs = Input(shape = (n, ))
    layer1 = Dense(M, activation = 'relu', kernel_initializer = 'he_uniform')(inputs)
    layer2 = Dense(M, activation = 'relu', kernel_initializer = 'he_uniform')(layer1)
    outputs = Dense(M, activation = 'sigmoid')(layer2)
    rx = Model(inputs = inputs, outputs = outputs)
    return rx

def ml_receiver_reg(m, n, output_shape = 1):
    '''Machine learning receiver for regression
    INPUT
	m: Layer width
    n: Number of channel uses
    output_shape: Output dimension of receiver
    OUTPUT
	rx: Receiver model
    '''
    inputs = Input(shape = (n, ))
    layer1 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(inputs)
    layer2 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(layer1)
    outputs = Dense(output_shape, activation = 'linear')(layer2)
    rx = Model(inputs = inputs, outputs = outputs)
    return rx

def ml_receiver_reg_heteroscedastic(m, n):
    '''Machine learning receiver for regression - heteroscedastic: with noise variance layer
    INPUT
	m: Layer width
    n: Number of channel uses
    OUTPUT
	rx: Receiver model
    '''
    inputs = Input(shape = (n, ))
    layer1 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(inputs)
    layer2 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(layer1)
    # mu NN
    mu_layer1 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(layer2)
    mu_layer2 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(mu_layer1)
    mu_out = Dense(1, activation = 'linear')(mu_layer2)
    # sigma NN
    sigma_layer1 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(layer2)
    sigma_layer2 = Dense(m, activation = 'relu', kernel_initializer = 'he_uniform')(sigma_layer1)
    sigma_out = Dense(1, activation = 'linear')(sigma_layer2)
    sigma_out_abs = Lambda(lambda x: tf.math.abs(x))(sigma_out)

    labels = Input(shape = (1,))
    rx = Model(inputs = [inputs, labels], outputs = [mu_out, sigma_out_abs])
    
    # Construct your custom loss as a tensor
    loss = mse_heteroscedastic(labels, mu_out, sigma_out_abs)
    # Add loss to model
    rx.add_loss(loss)
    rx.add_metric(mse_heteroscedastic_metric(labels, mu_out), name = 'mseh', aggregation = 'mean')
    return rx

def ml_transmitter(m, n, input_shape = 1, axnorm = 0):
    '''Machine learning transmitter
    INPUT
	m: Layer width
    n: Number of channel uses
    input_shape: number of inputs
    axnorm: axis for normalization
    OUTPUT
	tx: Transmitter model
    '''
    tx_in = Input(shape=(input_shape))
    tx_layer1 = Dense(m, activation = 'relu', kernel_initializer='he_uniform')(tx_in) # , kernel_initializer='he_uniform' # for RELU
    tx_layer2 = Dense(m, activation = 'relu', kernel_initializer='he_uniform')(tx_layer1)
    tx_layer3 = Dense(n, activation = 'linear')(tx_layer2)
    tx_out = Lambda(lambda  x: mt.normalize_input(x, axis = axnorm))(tx_layer3)
    tx = Model(inputs = tx_in, outputs = tx_out)
    return tx


def ml_com_reg(m, n, sigma, input_shape = 1, output_shape = 1, axnorm = 0, hmode = 0):
	'''Communication System as Autoencoder for continuous floating point outputs s - with noise variance layer in hmode (heteroscedastic)
	INPUT
	m: Layer width
	n: Number of channel uses
	sigma: Standard deviation limits [sigma_min, sigma_max]
	input_shape: Input shape (1 for continuous input)
	axnorm: Axis for normalization at tx
	hmode: Default (0), heteroscedastic (1)
	OUTPUT
	model: AE model
	tx: Transmitter model
	rx: Receiver model
	'''
	tx = ml_transmitter(m, n, input_shape = input_shape, axnorm = axnorm)
	if hmode == 1:
		rx = ml_receiver_reg_heteroscedastic(m, n)
	else:
		rx = ml_receiver_reg(m, n, output_shape = output_shape)

	# Model for autoencoder training
	ae_in = Input(shape = (input_shape))
	ch_in = tx(ae_in)
	ch_out = mt.GaussianNoise2(sigma)(ch_in)
	if hmode == 1:
		label_in = Input(shape = (input_shape))
		ae_out = rx([ch_out, label_in])
		model = Model(inputs = [ae_in, label_in], outputs = ae_out)
	else:
		ae_out = rx(ch_out)
		model = Model(inputs = ae_in, outputs = ae_out)

	return model, tx, rx

def mse_heteroscedastic(y_true, mu, sigma):
    '''Heteroscedastic version of MSE loss
    INPUT
	y_true: True outputs
    mu: Predicted output y_pred[0]
    sigma: Standard deviation y_pred[1] (in second list entry)
    OUTPUT
	mse_heteroscedastic: Mean square error loss in heteroscedastic model, i.e., with variance depending on input data
    '''
    # For numerical stability, avoiding NaNs, e.g., 1e-4
    const = 1e-4 
    sigma2 = sigma + const
    # Two versions of the same function: numerically different?
    # mse_heteroscedastic = tf.keras.backend.log(sigma2) + tf.keras.backend.square(y_true - mu) / (2 * tf.keras.backend.square(sigma2))
    mse_heteroscedastic = tf.keras.backend.mean(tf.keras.backend.square((y_true - mu) / sigma2) / 2 + tf.keras.backend.log(sigma2) + 0.5 * tf.keras.backend.log(2 * np.pi), axis = 0)
    return mse_heteroscedastic


def mse_heteroscedastic_metric(y_true, mu):
	'''MSE metric for heteroscedastic training
	INPUT
	y_true: True outputs
	mu: Predicted output y_pred[0]
	OUTPUT
	mse: Mean square error loss computed manually
	'''
	mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - mu), axis = 0)
	return mse

def lin_det_soft(x_est, Phi_ee, m, alpha):
    '''Calculation of soft information of symbols from linear detectors
    INPUT
    x_est: Equalized symbols by linear equalizer
    Phi_ee: Covariance matrix of error in x_est
    m: modulation alphabet
    alpha: modulation probabilities
    OUTPUT
    p_x: symbol probabilities
    '''
    # Neglection of non-diagonal entries of Sigma
    arg = - 1 / 2 / mop.tdiag2vec(Phi_ee)[..., np.newaxis] * (x_est[..., np.newaxis] - m[np.newaxis, np.newaxis, ...]) ** 2 + np.log(alpha[np.newaxis, ...])
    p_x = mop.np_softmax(arg)
    return p_x




if __name__ == '__main__':
	#     my_func_main()
	# def my_func_main():

	# Initialization
	tf.keras.backend.clear_session()          		# Clearing graphs
	tf.keras.backend.set_floatx('float32')			# Computation accuracy: 'float16', 'float32', or 'float64'
	mt.GPU_sel(num = -2, memory_growth = 'True')	# Choose/disable GPU: (-2) default, (-1) CPU, (>=0) GPU
	np.random.seed()            					# Random seed in every run, predictable random numbers for debugging with np.random.seed(0)

	# Simulation
	fn_ext = '_test'   								# _Nb100000it100
	algo = 'DNN'									# hardbit , MAPseq, Meanseq, analog, DNN, DNNh, AE, AEb, AEh, hardbitNN
													# Names in article 2 [SCIL]: Single-bit detector, MAP detection, Mean estimator, Analog transmission, DNN estimator
													# Article 1 [SINFONY_DRAFT]: AE=DNN transceiver
	load_set = 0									# 1: Load evaluation data and proceed
	prior = 3               						# 0: Gaussian, 1: Uniform (continuous), 2: Data-based DLR (gradients), 3: Data-based DLR (gradients + models), 4: Data-based SINFONY, 5: Uniform (per class)
	it_max = 100            						# Maximum number of iterations per SNR value: 100
	Nb = 100000             						# Evalution batch size: 10000, 100000
	seq_step = 1000         						# Sequence estimator steps: [home: 100, work: 1000]
	step_size = 1									# EbN0 step size
	EbN0_range = [-6, 14]   						# Validation Eb/N0: [-6, 14]
	R_c = 1											# Provide code rate for saves
	floatx = mfl.float_toolbox('float16')			# Floating point object: float4/8/16
	print('# of bits: ' + '{}'.format(Nb * floatx.N_bits * it_max))
	print('# of floats: ' + '{}'.format(Nb * it_max))
	mod = com.modulation('BPSK')					# Modulation object: BPSK, ...
	mod.alpha = 1 / mod.M * np.ones((int(floatx.N_bits / np.log2(mod.M)), mod.M))	# Set correct alpha dimensions
	sim_par = mf.simulation_parameters(1, 1, 1, mod, 1, EbN0_range, rho = 0)		# Simulation parameters object
	sim_par.snr_gridcalc(step_size)
	p_ba, p_s = compute_prior(floatx, mode = prior)


	# Path and save
	path = os.path.join('curves_float', mod.mod_name, floatx.float_name, 'prior' + str(prior))
	filename = 'RES_' + algo + fn_ext
	pathfile = os.path.join(path, filename)
	saveobj = mf.savemodule(form = 'npz')

	perf_meas = mf.performance_measures(Nerr_min = 100, it_max = it_max, sel_crit = 'it')
	if load_set == 1:
		perf_meas.load_results(saveobj.load(pathfile))

	## TRAINING of DNN-based approaches
	Ntx = floatx.N_bits         # Number of channel uses: floatx.N_bits
	NL = 2 * Ntx            	# Intermediate layer width: 2 * Ntx
	llr = 0                 	# llr-Rx input: 0 (False, default)
	axnorm = 0              	# AE-Tx output normalization axis
	Ne = 10000              	# Number of batch iterations, default: 10000
								# Note: With datasets, this measure needs to be converted to number of epochs by use of dataset size, e.g., for seismic DLR dataset Ntrain = 1,147,000
	steps_per_epoch = 10		# SGD steps per batch iteration with same data samples
	bs = 500                	# Training batch size: 1000, 500
	bval = 1000             	# Validation batch size: 1000
	# Training SNR
	snr_min_train = 6      		# 6, -4
	snr_max_train = 16       	# 16, 6
	opt = Adam()				# Optimizer: Adam, SGD
	# opt = SGD(learning_rate = 0.001, momentum = 0.9) # learning_rate = 0.001, momentum = 0.9, nesterov = True
	float_data = float_dataset_gen(bs, floatx, mod, snr_min_train, snr_max_train, p_s)
	data_val = float_data(bval)	# Generate validation data
	
	# Learn posterior float distribution as Rx (with heteroscedastic model)
	if algo == 'DNN' or algo == 'DNNh':
		start_time = time.time()
		if algo == 'DNN':
			print('Start training Rx DNN...')
			model = ml_receiver_reg(2 * floatx.N_bits, floatx.N_bits)
			model.compile(optimizer = opt, loss = 'mean_squared_error') # metrics=['mean_squared_error']) # 'categorical_crossentropy', 'binary_crossentropy', 'mse'
			val_data = (dnn_input(data_val[0], data_val[1], mode = llr), data_val[2])
			hmode = 0
		elif algo == 'DNNh':
			# Learn posterior floating point value distribution with heteroscedastic model
			print('Start training Rx heteroscedastic DNN...')
			model = ml_receiver_reg_heteroscedastic(2 * floatx.N_bits, floatx.N_bits)
			model.compile(optimizer =  opt)
			val_data = [dnn_input(data_val[0], data_val[1], mode = llr), data_val[2]]
			hmode = 1
		else:
			print('DNN not defined.')
		history = model.fit(data_generator(float_data, outbin = 0, llr = llr, hmode = hmode),
							steps_per_epoch = steps_per_epoch,
							epochs = Ne,
							shuffle = False,
							batch_size = None,
							validation_data = val_data, #validation_split = 0)
							validation_steps = 1,
							verbose = 2, # 2
							)
		print('Tot. time: ' + mf.print_time(time.time() - start_time))
	
	# Learn whole communication system as one AE (with heteroscedastic model)
	if algo == 'AE' or algo == 'AEh' or algo == 'AEb' or algo == 'AEbh':
		if algo == 'AEb' or algo == 'AEbh':
			# Bit input
			input_shape = floatx.N_bits
			val_data = (data_val[3], data_val[2])
		else:
			# Floating point input
			input_shape = 1
			val_data = (data_val[2], data_val[2])
		if algo == 'AE' or algo == 'AEb':
			print('Start training AE...')
			model, _, _ = ml_com_reg(NL, Ntx, mop.csigma(np.array([snr_min_train, snr_max_train]))[::-1], input_shape = input_shape, axnorm = axnorm)
			model.compile(optimizer = opt, loss = 'mean_squared_error')
			hmode = 0
		elif algo == 'AEh' or algo == 'AEbh':
			print('Start training heteroscedastic AE...')
			model, _, _ = ml_com_reg(NL, Ntx, mop.csigma(np.array([snr_min_train, snr_max_train]))[::-1], input_shape = input_shape, axnorm = axnorm, hmode = 1)
			model.compile(optimizer =  opt)
			val_data = list(val_data)
			hmode = 1
		else:
			print('AE not defined.')
		start_time = time.time()
		history = model.fit(data_generator_ae(float_data, input_shape = input_shape, hmode = hmode),
							steps_per_epoch = steps_per_epoch,
							epochs = Ne,
							shuffle = False,
							batch_size = None,
							validation_data = val_data,
							validation_steps = 1,
							verbose = 2,
							)
		print('Tot. time: ' + mf.print_time(time.time() - start_time))
	
	# Learn posterior bit distribution
	if algo == 'hardbitNN':
		print('Start training...')
		model2 = ml_receiver_bits(floatx.N_bits, floatx.N_bits)
		model2.compile(optimizer = opt, loss = 'binary_crossentropy')
		start_time = time.time()
		history2 = model2.fit(data_generator(float_data, outbin = 1, llr = llr),
							steps_per_epoch = steps_per_epoch,
							epochs = Ne,
							shuffle = False,
							batch_size = None,
							validation_data = (dnn_input(data_val[0], data_val[1], mode = llr), data_val[3]),
							validation_steps = 1,
							verbose = 0, # 2
							)
		print('Tot. time: ' + mf.print_time(time.time() - start_time))


	## SIMULATION
	float_data.N_batch = Nb

	for ii, snr in enumerate(sim_par.SNR):
		if snr not in perf_meas.SNR:
			while perf_meas.stop_crit():    # Simulate until 1000 errors or stop after it_max iterations
				
				float_data.snr_min = snr
				float_data.snr_max = snr
				[y, sigma, s, b] = float_data()
				z = np.concatenate((1 - b[..., np.newaxis], b[..., np.newaxis]), axis = -1)

				# Stochastically independent prior Equalizer
				if algo == 'hardbit':
					x_est = y
					Phi_ee = mop.tvec2diag(sigma ** 2)
					p_x = lin_det_soft(x_est, Phi_ee, mod.m, p_ba) # p_ba = mod.alpha
					_, p_b0 = com.symprob2llr(p_x, mod.M)
					p_b = np.concatenate((p_b0, 1 - p_b0), axis = -1)
						
				# NN equalizer
				if algo == 'hardbitNN':
					p_b1 = model2.predict(dnn_input(y, sigma, mode = llr))[..., np.newaxis]
					p_b = np.concatenate((1 - p_b1, p_b1), axis = -1)
					b_r = (p_b[..., 0] < 0.5) * 1

				# 1. Hard bit decision estimator, Single-bit detector
				if algo == 'hardbit' or algo == 'hardbitNN':
					b_r = (p_b[..., 0] < 0.5) * 1
					b_rc = floatx.float_errorcorrection(b_r, p_b[..., 0])
					s_r = floatx.bit2float(b_rc)

				# 2. DNN (heteroscedastic), DNN estimator
				if algo == 'DNN' or algo == 'DNNh':
					if algo == 'DNN':
						s_r = model.predict(dnn_input(y, sigma, mode = llr))
					elif algo == 'DNNh':
						# Two outputs: mean, stddev
						s_r, s_r_var = model.predict([dnn_input(y, sigma, mode = llr), np.zeros(s.shape)]) # dummy
						s_r_var = s_r_var[:, 0]
					s_r = s_r[:, 0]
					# _, s_r = floatx.float2bitint(s_r)       # Discretization from float64 to floatx
					b_r = floatx.float2bit(s_r)
					p_b = np.concatenate((1 - b_r[:, :, np.newaxis], b_r[:, :, np.newaxis]), axis = -1)

				# 3. AE (heteroscedastic)
				if algo == 'AE' or algo == 'AEh' or algo == 'AEb' or algo == 'AEbh':
					model.layers[2].set_weights([np.array([mop.csigma(snr), mop.csigma(snr)])])
					if algo == 'AEb' or algo == 'AEbh':
						input0 = b
					else:
						input0 = s
					if algo == 'AE' or algo == 'AEb':
						s_r = model.predict(input0)
					elif algo == 'AEh' or algo == 'AEbh':
						s_r, s_r_var = model.predict([input0, np.zeros(input0.shape)]) # dummy
						s_r_var = s_r_var[:, 0]
					s_r = s_r[:, 0]
					# Discretization from float64 to floatx
					# _, s_r = floatx.float2bitint(s_r)
					b_r = floatx.float2bit(s_r)
					p_b = np.concatenate((1 - b_r[:, :, np.newaxis], b_r[:, :, np.newaxis]), axis = -1)

				# 4. MAP Sequence Estimator (Laplace Approximation), MAP detection
				if algo == 'MAPseq':
					s_r, b_r, _, _ = sequence_est(p_s, y, sigma, mod, floatx, step = seq_step, mode = 1) # home: 100, work: 1000
					p_b = np.concatenate((1 - b_r[:, :, np.newaxis], b_r[:, :, np.newaxis]), axis = -1)

				# 5. Mean Estimator (Moment Matching)
				if algo == 'Meanseq':
					_, b_r, s_r, _ = sequence_est(p_s, y, sigma, mod, floatx, step = seq_step, mode = 2) # home: 100, work: 1000
					p_b = np.concatenate((1 - b_r[:, :, np.newaxis], b_r[:, :, np.newaxis]), axis = -1)
					# Discretization from float64 to floatx
					_, s_r = floatx.float2bitint(s_r)

				# 6. Analog transmission
				if algo == 'analog':
					isfin = np.isfinite(floatx.x_poss)
					sigma_s = np.sqrt(np.sum(floatx.x_poss[isfin].astype('float64') ** 2 * p_s[isfin]))
					## Simulate floatx.N_bits channel uses
					# s_t = np.repeat(s[..., np.newaxis], floatx.N_bits, axis = -1)
					# y = com.awgn(s_t.astype('float64'), np.repeat(sigma_s * sigma, floatx.N_bits, axis = -1))
					# s_r = np.mean(y, axis = -1)
					## Short version with only one channel use but same performance
					y = com.awgn(s, sigma_s * sigma[:, 0] / np.sqrt(floatx.N_bits))
					s_r = y
					# Discretization from float64 to floatx
					# _, s_r = floatx.float2bitint(y)
					b_r = floatx.float2bit(s_r)
					p_b = np.concatenate((1 - b_r[:, :, np.newaxis], b_r[:, :, np.newaxis]), axis = -1)
				

				# Performance evaluation
				perf_meas.eval(z, p_b, mod)
				# Resolution of 64 bits necessary for MSE calculation -> otherwise overflow
				s = s.astype('float64')
				perf_meas.mse_calc(s, s_r, mode = 0)
				# Output
				perf_meas.err_print()
			
			# Save only if accuracy high enough after it_max iterations
			[print_str, sv_flag] = perf_meas.err_saveit(sim_par.SNR[ii], sim_par.EbN0[ii], sim_par.EbN0[ii] - 10 * np.log10(R_c))
			print('{}, '.format(len(sim_par.SNR) - ii) + print_str)
			if sv_flag:
				saveobj.save(pathfile, perf_meas.results()) # Save results to file



	print('Simulation ended.')
# EOF