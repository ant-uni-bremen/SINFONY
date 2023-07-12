#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of classical digital communication in the article:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Semantic Information Recovery in Wireless Networks," https://doi.org/10.48550/arXiv.2204.13366
"""

# LOADED PACKAGES
# Python packages
import os
import numpy as np
from matplotlib import pyplot as plt
import time

# Tensorflow 2 packages
import tensorflow as tf
# Keras functionality
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Add, Lambda, Concatenate, Layer, GaussianNoise
from tensorflow.keras.optimizers import SGD, Adam, Nadam
import sionna as sn


## Own packages
import sys
sys.path.append('..')	# Include parent folder, where own packages are
sys.path.append('.')	# Include current folder, where start simulation script and packages are
import mymathops as mop
from myfunc import print_time, savemodule
import mytraining as mt
import mytraining as mf	# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import myfloat as mfl

import SINFONY as sc
import huffman_coding as hc

# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'





def classiccom(s, huffman, k, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx = None, p_b = None):
	'''Classical digital bit transmission with Huffman source coding and LDPC code over AWGN channel
	s: Float values of application / bit input as integer values
	huffman: Huffman encoder/decoder object
	k: Information word length
	encoder: Channel encoder
	mapper: Modulation mapper
	channel: Communication channel
	demapper: Modulation demapper
	decoder: Channel decoder 
	snr: SNR in dB
	p_b: Probabilities of floating bits
	floatx: Floating point conversion object (optional, required for float input)
	s_r: Reconstructed float values for application / bit output as integer values
	'''
	# Check if input is integer
	int_check = np.issubdtype(s.dtype, np.integer)
	if int_check == True:
		# Integers/bits are directly fed into the communication system
		# Bit number is hard coded here
		bint = s
		inttype = bint.dtype
		if bint.dtype == 'uint8':
			N_bits = 8
		else:
			print('Not implemented!')
	else:
		# Transform source signal into floating point bits
		bint, _ = floatx.float2bitint(s)
		inttype = floatx.intx
		N_bits = floatx.N_bits
		
	# Huffman encoding
	b_huffseq, _ = huffman.encoding(bint)
	# If the number of bits required to be a code block, add random bits
	b_fill = np.random.randint(0, 2, size = k - len(b_huffseq) % k)
	b = np.concatenate((np.array(b_huffseq, dtype = 'float32'), b_fill))
	
	# Channel coding
	c = encoder(b.reshape((-1, k)))
	
	c_int = interleaver(c)
	x = mapper(c_int)
	sigma = mop.csigma(np.random.uniform(snr, snr, x.shape[0]))[..., np.newaxis].astype('float32')
	if constellation._constellation_type == 'pam':
		y = channel([x,  2* sigma ** 2])		# Complex channel with half the variance in real- and imaginary part
		llr_ch = demapper([y, 2 * sigma ** 2]) 	# Also consider here doubled variance
	else:
		y = channel([x,  sigma ** 2])
		llr_ch = demapper([y, sigma ** 2])
	llr_int = deinterleaver(llr_ch)
	c_r = decoder(llr_int) 						# Soft information of c_r cannot pass through Huffman decoding
	
	# Remove added random bits and Huffman decoding
	bint_r = np.array(huffman.decoding(c_r.numpy().flatten()[0:b.shape[0] - b_fill.shape[0]].astype(inttype).tolist()))
	# If the number of bits after Huffman decoding has changed compared to transmit signals:
	if bint.shape[0] < bint_r.shape[0]:
		# More than before: Take only as much bits
		bint_r = bint_r[0:bint.shape[0]]
	elif bint.shape[0] > bint_r.shape[0]:
		# Less than before: Add random bits
		bint_r = np.concatenate((bint_r, np.random.randint(0, 2 ** N_bits, size = bint.shape[0] - bint_r.shape[0], dtype = inttype)))
	if int_check == True:
		s_r = bint_r
	else:
		# Transform Huffman integers back to bit stream
		b_r = mop.int2bin(bint_r, N = N_bits)
		# Float error correction based on a priori probabilities
		b_rc = floatx.float_errorcorrection(b_r, p_b[..., 0][np.newaxis].repeat(b_r.shape[0], axis = 0))
		# Transform bit sequence into float value
		s_r = floatx.bit2float(b_rc)
	return s_r



def ml_com_reg_sinfony(n, m_tx, m_rx, sigma, input_shape = 1, output_shape = 1, num_layer = 1, rx_linear = False, axnorm = 0):
	'''Com. System as Autoencoder for continuous outputs/floats s like in SINFONY
	- INPUT -
	m: Layer width
	n: Number of channel uses
	sigma: stdev limits [sigma_min, sigma_max]
	input_shape: Input shape
	output_shape: Output shape
	axnorm: Axis for normalization at Tx
	- OUTPUT -
	model: AE model
	tx: Transmitter model
	rx: Receiver model
	'''
	# Transmitter design
	tx_in = Input(shape=(input_shape))
	tx_layer = tx_in
	for indl in range(0, num_layer):
		tx_layer = Dense(m_tx, activation = 'relu', kernel_initializer = 'he_uniform')(tx_layer) # for RELU
	tx_layer3 = Dense(n, activation = 'linear')(tx_layer)
	tx_out = mt.normalize_input(tx_layer3, axis = axnorm)
	tx = Model(inputs = tx_in, outputs = tx_out)

	# Receiver Design
	rx_in = Input(shape = (n, ))
	layer = rx_in
	for indl in range(0, num_layer):
		layer = Dense(m_rx, activation = 'relu', kernel_initializer = 'he_uniform')(layer)
	if rx_linear == True:
		layer = Dense(output_shape, activation = 'linear')(layer)
	rx_out = layer
	rx = Model(inputs = rx_in, outputs = rx_out)

	# Model for autoencoder training
	ae_in = Input(shape = (input_shape))
	ch_in = tx(ae_in)
	ch_out = mt.GaussianNoise2(sigma)(ch_in)
	ae_out = rx(ch_out)
	model = Model(inputs = ae_in, outputs = ae_out)

	return model, tx, rx


def sequence_prior_data_int(bint, bint_max = -1, show = False):
	'''Probabilities p(s) of bit sequences or s computed from data set
	INPUT
	bint: Bit sequence as integer
	bint_max: Highest integer of bit sequence
	show: Show data distribution
	OUTPUT
	p_s: probabilities for each possible integer value
	'''
	# Probabilities of data set discretized to integer values
	# Flatten and discretize data set to floatx precision
	b_count = np.bincount(bint)
	if bint_max == -1:
		bint_max = b_count.shape[0]
	p_s = np.zeros(bint_max)
	p_s[0:b_count.shape[0]] = b_count
	p_s = p_s / np.sum(p_s)

	## Plot data distribution before and after quantization
	if show == True:
		plt.figure()
		plt.hist(bint, bins = bint_max)
		plt.figure()
		plt.plot(np.arange(0, bint_max), p_s, 'r-o', label = 'p(s)')
	return p_s






if __name__ == '__main__':
#     my_func_main()
# def my_func_main():

	## Initialization
	tf.keras.backend.clear_session()          	# Clearing graphs
	tf.keras.backend.set_floatx('float32')		# Computation accuracy: 'float16', 'float32', or 'float64'
	mt.GPU_sel(num = -2, memory_growth = 'True')# Choose/disable GPU: (-2) default, (-1) CPU, (>=0) GPU
	np.random.seed()            				# Random seed in every run, predictable random numbers for debugging with np.random.seed(0)

	## Simulation parameters
	classic = 1									# 0: Analog AE, 1: classic, 2: classic (image transmission)
	fn_ext = '_rc25_n15360_h100'				# '_rc25_n15360_h100','_rc5_n16000_h1000', '_ntx56_NL112_Ne100_snr-4_6'
	saveobj = savemodule(form = 'npz')
	
	# Loaded dataset and SINFONY design
	dataset = 'mnist'							# mnist, cifar10, fashion_mnist
	trainx, trainy, testx, testy = sc.load_dataset(dataset)
	if classic == 2:
		filename = 'ResNet14_MNIST' 			# File for central image classification
	else:
		filename = 'ResNet14_MNIST2'			# File for distributed image classification with perfect links
	path = 'models_sem'							# Path for SINFONY model
	show_dataset = True							# Show dataset examples and model summaries

	# Parameters
	if classic == 0:
	# Analog AE parameters
		load = 0								# Load models or start new training
		algo = 'AErvec'							# AE, AErvec, AErvec_ind
		fn_ext2 = fn_ext						# '_ntx56_NL56_snr-4_6'
		path2 = 'models_classic'				# Path for AE models
		
		# Training parameters
		Ntx = 14								# Number of channel uses: 56 (AE_rvec) / 4, 16 (AE)
		# NL = 2 * Ntx							# Intermediate layer width: 2 * Ntx
		NL_tx = 14								# Intermediate Tx layer width
		NL_rx = 56								# Intermediate Rx layer width
		num_layer = 1							# Number of Tx and Rx module layers
		rx_linear = False						# Final linear Rx module layer? (default: False)
		axnorm = 0								# AE-Tx output normalization axis
		Nepoch = 20								# Number of Epochs for rvec: 100
		bs = 64									# Batch Size: 1000, 500
		optimizer = 'sgdlrs'					# sgd, adam, (sgdlrs) SGD with learning rate schedule
		lr = 1e-3								# Learning rate, SGD/Adam: 1e-3
		snr_min_train = -4 						# default: -4, 6
		snr_max_train = 6						# default: 6, 16
		## Optimizers
		if optimizer.lower() == 'sgdlrs':
			# Learning rate schedule like for SINFONY
			epoch_bound = [3, 6]				# at 32000, 48000 iterations of 64000 in total: [100, 150] for CIFAR / [3, 6] for MNIST / [2, 50] for hirise / [100] for RL CIFAR
			iter_epoch = trainx.shape[0] / bs
			boundaries = list(np.round(np.array(epoch_bound) * iter_epoch).astype('int'))
			values = [0.1, 0.01, 0.001]			# [0.1, 0.01, 0.001] for ae training
			lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
			opt = SGD(learning_rate = lr_schedule, momentum = 0.9)
		elif optimizer.lower() == 'adam':
			# Adam and its variants
			opt = Adam(learning_rate = lr)
		else:
			# Default: Stochastic Gradient Descent with momentum 0.9 as in ResNet paper
			opt = SGD(learning_rate = lr, momentum = 0.9)
		
		# Evaluation parameters
		val_rounds = 10 						# 10, rounds through validation data with different channel noise realizations
		SNR_range = [-30, 20] 					# -30, 20
		step_size = 1
	else:
	# Classic communications parameters
		ratec = 0.25							# Code rate: 0.25, 0.5, 0.75
		n = 15360								# Code length: 1000 (default), 16000 (ratec = 0.5), 15360 (ratec = 0.25), 11264 (rattec = 0.75)
		blocks = 100							# Huffman encode [blocks] feature vectors into one block + channel encoding: 100, 1000 is practical
												# Huffman code is computational bottleneck: But the smaller the blocks, the less severe errors are
		mod = 'pam'								# Modulation
		num_bits_per_symbol = 1
		float_name = 'float16'					# float16
		if classic == 1:
			algo = 'classic'
		elif classic == 2:
			algo = 'classic_image'
		
		# Evaluation parameters
		if mod == 'qam':
			M = num_bits_per_symbol / 2
		else:
			M = num_bits_per_symbol
		val_rounds = 10 						# 10, rounds through validation data with different channel noise realizations
		SNR_range = [-1, 5] + 10 * np.log10(2 * ratec * M) # [-1, 5] + 10 * np.log10(2 * ratec * M)
		step_size = 0.5



	## Evaluation script

	# Load the SINFONY model
	path0 = os.path.dirname(os.path.abspath(__file__))	# Path of script being executed
	pathfile = os.path.join(path0, path, filename)
	print('Loading model ' + filename + '...')
	model = tf.keras.models.load_model(pathfile)
	print('Model loaded.')
	if show_dataset == True:
		model.summary()

	# Preprocess Data set
	train_norm, test_norm = sc.prep_pixels(trainx, testx)
	if show_dataset == True:
		sc.summarize_dataset(trainx, trainy, testx, testy)

	if classic != 2:
		data_train = model.layers[1].predict(train_norm)
		data_val = model.layers[1].predict(test_norm)


	# Initialize classic and AE communications
	start_time = time.time()
	rng = np.random.default_rng()

	if classic == 0:
		# Analog AE Training
		filename2 = algo + fn_ext2
		pathfile2 = os.path.join(path0, path2, filename2)
		if algo == 'AErvec':
			print('AE for all agents / feature vectors rvec:')
			val_data = data_val.reshape([-1, data_val.shape[-1]])
			if load == 0:
				print('Start training...')
				# Prepare dataset for rvec
				input_shape = data_train.shape[-1]
				output_shape = input_shape
				data = data_train.reshape([-1, data_train.shape[-1]])
			else:
				print('Load model...')
		elif algo == 'AE':
			print('AE model for each entry accross all rvec entries r_i:')
			val_data = data_val.flatten()
			if load == 0:
				print('Start training...')
				# Prepare dataset for entries in rvec
				input_shape = 1
				output_shape = 1
				data = data_train.flatten()
			else:
				print('Load model...')
				
		
		if algo == 'AErvec_ind':
			# Special training procedure for AE optimized for individual rvec
			print('AE for each individual agent / feature vector rvec...')
			val_data = data_val.reshape([data_val.shape[0], -1, data_val.shape[-1]])
			Ndis = val_data.shape[1]
			if load == 0:
				print('Start training...')
				# Training
				start_time2 = time.time()
				models = []
				input_shape = data_train.shape[-1]
				output_shape = input_shape
				data = data_train.reshape([data_train.shape[0], -1, data_train.shape[-1]])
				for indae in range(0, Ndis):
					print('Start training AE' + str(indae) + '...')
					model2, _, _ = ml_com_reg_sinfony(Ntx, NL_tx, NL_rx, mop.csigma(np.array([snr_min_train, snr_max_train]))[::-1], input_shape = input_shape, output_shape = output_shape, num_layer = num_layer, rx_linear = rx_linear, axnorm = axnorm)
					model2.compile(optimizer = opt, loss = 'mean_squared_error')
					start_time = time.time()
					history = model2.fit(data[:, indae, ...], data[:, indae, ...], 
										epochs = Nepoch,
										batch_size = bs,
										validation_data = (val_data[:, indae, ...], val_data[:, indae, ...]),
										# verbose = 2,
										)
					print('Tot. time ' + 'AE' + str(indae) +': ' + print_time(time.time() - start_time))
					# Save model
					filename2 = algo + str(indae) + fn_ext2
					pathfile2 = os.path.join(path0, path2, filename2)
					print('Saving model...')
					model2.save(pathfile2)
					print('Model saved.')
					models.append(model2)
				print('Tot. time all AEs: ' + print_time(time.time() - start_time2))
			else:
				# Load model
				print('Load model...')
				models = []
				for indae in range(0, Ndis):
					filename2 = algo + str(indae) + fn_ext2
					pathfile2 = os.path.join(path0, path2, filename2)
					print('Loading AE model' + str(indae) + '...')
					model2 = tf.keras.models.load_model(pathfile2)
					print('AE model ' + str(indae) + ' loaded.')
					if show_dataset == True:
						model2.summary()
					models.append(model2)
		else:
			# Usual AE training script
			if load == 0:
				# Training
				start_time = time.time()
				model2, _, _ = ml_com_reg_sinfony(Ntx, NL_tx, NL_rx, mop.csigma(np.array([snr_min_train, snr_max_train]))[::-1], input_shape = input_shape, output_shape = output_shape, num_layer = num_layer, rx_linear = rx_linear, axnorm = axnorm)
				model2.compile(optimizer = opt, loss = 'mean_squared_error')
				history = model2.fit(data, data, 
									epochs = Nepoch,
									batch_size = bs,
									validation_data = (val_data, val_data),
									# verbose = 2,
									)
				print('Tot. time ' + 'AE: ' + print_time(time.time() - start_time))
				# Save model
				print('Saving AE model...')
				model2.save(pathfile2)
				print('AE Model saved.')
			else:
				# Load model
				print('Loading AE model...')
				model2 = tf.keras.models.load_model(pathfile2)
				print('AE Model loaded.')
				if show_dataset == True:
					model2.summary()
			models = [model2]
	elif classic == 1 or classic == 2:
		# Classic communication for whole test data set
		if classic == 2:
			# Images have 256 Bit RGB color entries
			N_bits = 8
			N_cat = 2 ** N_bits
			b_poss = np.arange(0, N_cat)
			p_s = sequence_prior_data_int(testx.flatten(), bint_max = N_cat)
		elif classic == 1:
			# Features to be transmitted are floating point values
			floatx = mfl.float_toolbox(float_name)
			# Note: Compression from, e.g., float32 to float16 possible!
			N_bits = floatx.N_bits
			b_poss = floatx.b_poss
			p_s = mfl.sequence_prior_data(floatx, data = data_val.flatten())
			p_b = mfl.compute_single_bitprob(floatx, p_s)
		huffman = hc.huffman_coder(symbols = b_poss, probs = p_s)
		# Compute total gain of the Huffman encoding
		_, _, _, _, rateh = huffman.total_gain()
		# Information word length
		k = int(n * ratec)
		# Communications components from Sionna
		encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
		interleaver = sn.fec.interleaving.RowColumnInterleaver(row_depth = num_bits_per_symbol)
		deinterleaver = sn.fec.interleaving.Deinterleaver(interleaver)
		constellation = sn.mapping.Constellation(mod, num_bits_per_symbol = num_bits_per_symbol)
		mapper = sn.mapping.Mapper(constellation = constellation)
		channel = sn.channel.AWGN()
		demapper = sn.mapping.Demapper('app', constellation = constellation)
		decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, cn_type = 'boxplus') # , hard_out = False
		# Compute total rate
		rate = rateh * ratec
		if classic == 2:
			# Number of image entries
			N_entries = np.prod(testx.shape[1:])
		else:
			# Number of features
			N_entries = data_val.shape[-1]
		# Average number of channels uses with digital communication
		mean_numchuses = N_entries * N_bits / rate
	
	# Print initialization time
	print('Init. Time: ' + print_time(time.time() - start_time))



	## Evaluation of model
	print('Evaluate model...')
	## Evaluate model for different SNRs
	SNR = np.linspace(SNR_range[0], SNR_range[1], int((SNR_range[1] - SNR_range[0]) / step_size) + 1)
	# SINFONY/RL-SINFONY evaluated with classic communication
	start_time = time.time()
	eval_meas = [[], []]
	for ii, snr in enumerate(SNR):
		sigma = mop.csigma(snr)
		sigma_test = np.array([sigma, sigma])
		# Set standard deviation weights of Noise layer in AE approach
		if classic == 0:
			for model_ae in models:
				model_ae.layers[2].set_weights([sigma_test])
		lossi = 0
		acci = 0
		for ii2 in range(0, val_rounds):
			start_time2 = time.time()
			if classic == 1:
				## Test data features enter classic communications as source s.
				s = data_val
				## Evaluate classical digital transmission
				# Huffman encode [blocks] feature vectors into one block + split across agents
				s_r = np.zeros(s.shape)
				for ind0 in range(0, int(s.shape[0] / blocks)):
					# Consider transmission of each agent separately
					for ind1 in range(0, s.shape[1]):
						for ind2 in range(0, s.shape[2]):
								s_r[ind0 * blocks:(ind0 + 1) * blocks, ind1, ind2, :] = classiccom(s[ind0 * blocks:(ind0 + 1) * blocks, ind1, ind2, :].flatten(), huffman, k, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx = floatx, p_b = p_b).reshape((blocks, -1))
				# Extract semantics based on received signal y = r_r = s_r
				cl = model.layers[-1].predict(s_r)
			elif classic == 2:
				## Test image data enters classic communications as source s.
				s = testx.reshape([testx.shape[0], -1])
				## Evaluate classical digital transmission of images
				# Huffman encode [blocks] feature vectors into one block + split across agents
				s_r = np.zeros(s.shape)
				for ind0 in range(0, int(s.shape[0] / blocks)):
					s_r[ind0 * blocks:(ind0 + 1) * blocks, :] = classiccom(s[ind0 * blocks:(ind0 + 1) * blocks, :].flatten(), huffman, k, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr).reshape((blocks, -1))
				s_r = s_r.reshape(testx.shape)
				_, s_r = sc.prep_pixels(np.array(0), s_r)
				cl = model.predict(s_r)
			elif classic == 0:
				# Test data features enter classic communications as source s.
				s = data_val
				## Evaluate AE models
				Ndis = len(models)
				s2 = val_data
				if Ndis >= 2:
					# More than one model: AE individually trained for each agent
					s_r = np.zeros((s.shape[0], Ndis, s.shape[-1]), dtype = s.dtype)
					for indae in range(0, Ndis):
						s_r[:, indae, ...] = models[indae].predict(s2[:, indae, ...])
				else:
					# One model: Feed data one-shot
					s_r = models[0].predict(s2)
				s_r = s_r.reshape(s.shape)
				# Extract semantics based on received signal y = r_r = s_r
				cl = model.layers[-1].predict(s_r)

			# Calculate average loss and accuracy
			lossii = np.mean(model.loss(testy, cl))
			accii = np.mean(np.argmax(cl, axis = -1) == np.argmax(testy, axis = -1))

			# Add current measures to total measures
			lossi = (ii2 * lossi + lossii) / (ii2 + 1)
			acci = (ii2 * acci + accii) / (ii2 + 1)
			print('Validation Round: {}/{}, CE: {:.4f}, Acc: {:.2f}'.format(ii2 + 1, val_rounds, lossi, acci) + ', Time: ' + print_time(time.time() - start_time2))
		# Append list with evaluation for each SNR value
		eval_meas[0].append(lossi)
		eval_meas[1].append(acci)
		print('Iteration: {}/{}, SNR: {}, CE: {:.4f}, Acc: {:.2f}'.format(ii + 1, len(SNR), snr, lossi, acci) + ', Time: ' + print_time(time.time() - start_time))
	acc = np.array(eval_meas[1])
	loss = np.array(eval_meas[0])
	plt.figure(1)
	plt.semilogy(SNR, 1 - acc)
	plt.figure(2)
	plt.semilogy(SNR, loss)


	# Save evaluation
	print('Save evaluation...')
	results = {
		"snr": SNR,
		"val_loss": loss,
		"val_acc": acc,
		}
	pathfile = os.path.join(path0, path, 'RES_' + algo + '_' + filename + fn_ext)
	saveobj.save(pathfile, results)
	print('Evaluation saved.')



# EOF