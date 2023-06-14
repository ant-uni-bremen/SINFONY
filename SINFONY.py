#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
"""

## LOADED PACKAGES
import os
import numpy as np
from matplotlib import pyplot as plt
import time

# Tensorflow 2 packages
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten #, Add, Concatenate, Layer, GaussianNoise #, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, Nadam


## Own packages
import sys
sys.path.append('..')	# Include parent folder, where own packages are
sys.path.append('.')	# Include current folder, where start simulation script and packages are
import mymathops as mop
from myfunc import print_time, savemodule
import mytraining as mt

# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



## ResNet for image recognition

class Residual(tf.keras.Model):
	"""The Residual block of ResNet.
	For ReLU activations weight initialization with he_uniform is better than glorot
	Preactivation version for better training
	"""
	def __init__(self, num_channels, use_1x1conv = False, strides = 1, kernel_initializer = None, kernel_regularizer = None, preactivation = True):
		super().__init__()
		self.preactivation = preactivation
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_initializer2 = tf.keras.initializers.get(kernel_initializer)
		self.kernel_initializer3 = tf.keras.initializers.get(kernel_initializer)
		self.conv1 = tf.keras.layers.Conv2D(
			num_channels, padding = 'same', kernel_size = 3, strides = strides, kernel_initializer = self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)
		self.conv2 = tf.keras.layers.Conv2D(
			num_channels, padding = 'same', kernel_size = 3, kernel_initializer = self.kernel_initializer2, kernel_regularizer = self.kernel_regularizer)
		if use_1x1conv:
			self.conv3 = tf.keras.layers.Conv2D(
				num_channels, kernel_size = 1, strides = strides, kernel_initializer = self.kernel_initializer3, kernel_regularizer = self.kernel_regularizer)
		else:
			self.conv3 = None
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.bn2 = tf.keras.layers.BatchNormalization()


	#@tf.function # different results with tf.function decorator -> not necessary since the layers/models are just defined here, it is not a function
	def call(self, X):
		if self.preactivation == True:
			# New architecture: pre-activation
			Y = self.conv1(tf.keras.activations.relu(self.bn1(X)))
			Y = self.conv2(tf.keras.activations.relu(self.bn2(Y)))
			if self.conv3 is not None:
				X = self.conv3(X)
			Y = Y + X
		else:
			# Original architecture
			Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
			Y = self.bn2(self.conv2(Y))
			if self.conv3 is not None:
				X = self.conv3(X)
			Y = Y + X
			Y = tf.keras.activations.relu(Y)
		return Y


class Residual_bottleneck(tf.keras.Model):
	"""The Residual block of ResNet in bottleneck version.
	For ReLU activations weight initialization with he_uniform is better than glorot
	Preactivation version for better training
	(Identity shortcuts are essential for efficient training with less parameters, but not used in preactivation paper...)
	"""
	def __init__(self, num_channels, use_1x1conv = False, strides = 1, kernel_initializer = None, kernel_regularizer = None, preactivation = True):
		super().__init__()
		self.preactivation = preactivation
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_initializer2 = tf.keras.initializers.get(kernel_initializer)
		self.kernel_initializer3 = tf.keras.initializers.get(kernel_initializer)
		self.kernel_initializer4 = tf.keras.initializers.get(kernel_initializer)
		self.conv1 = tf.keras.layers.Conv2D(
			num_channels, padding = 'same', kernel_size = 1, strides = strides, kernel_initializer = self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)
		self.conv2 = tf.keras.layers.Conv2D(
			num_channels, padding = 'same', kernel_size = 3, kernel_initializer = self.kernel_initializer2, kernel_regularizer = self.kernel_regularizer)
		self.conv3 = tf.keras.layers.Conv2D(
			4 * num_channels, padding = 'same', kernel_size = 1, kernel_initializer = self.kernel_initializer3, kernel_regularizer = self.kernel_regularizer)
		if use_1x1conv:
			self.conv4 = tf.keras.layers.Conv2D(
				4 * num_channels, kernel_size = 1, strides = strides, kernel_initializer = self.kernel_initializer4, kernel_regularizer = self.kernel_regularizer)
		else:
			self.conv4 = None
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.bn3 = tf.keras.layers.BatchNormalization()
	
	#@tf.function
	def call(self, X):
		if self.preactivation == True:
			# New architecture: pre-activation
			Y = self.conv1(tf.keras.activations.relu(self.bn1(X)))
			Y = self.conv2(tf.keras.activations.relu(self.bn2(Y)))
			Y = self.conv3(tf.keras.activations.relu(self.bn3(Y)))
			if self.conv4 is not None:
				X = self.conv4(X)
			Y = Y + X
		else:
			# Original architecture
			Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
			Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
			Y = self.bn3(self.conv3(Y))
			if self.conv4 is not None:
				X = self.conv4(X)
			Y = Y + X
			Y = tf.keras.activations.relu(Y)
		return Y


class ResnetBlock(tf.keras.layers.Layer):
	def __init__(self, num_channels, num_residuals, first_block = False, kernel_initializer = None, kernel_regularizer = None, preactivation = True, bottleneck = False,
				 **kwargs):
		super(ResnetBlock, self).__init__(**kwargs)
		self.residual_layers = []
		for i in range(num_residuals):
			if bottleneck == True:
				if i == 0:
					self.residual_layers.append(
						Residual_bottleneck(num_channels, use_1x1conv = True, strides = 2, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, preactivation = preactivation))
				else:
					self.residual_layers.append(Residual_bottleneck(num_channels, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, preactivation = preactivation))
			else:
				if i == 0 and not first_block:
					self.residual_layers.append(
						Residual(num_channels, use_1x1conv = True, strides = 2, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, preactivation = preactivation))					
				else:
					self.residual_layers.append(Residual(num_channels, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, preactivation = preactivation))
						
	#@tf.function
	def call(self, X):
		for layer in self.residual_layers.layers:
			X = layer(X)
		return X



def resnet(shape = (32, 32, 3), classes = 10, filters = 64, num_res = 2, num_resblocks = 4, preactivation = True, bottleneck = False):
	'''Function returns architecture of ResNet18/34/50/101/152 for ImageNet dataset
	See Table 1 in "Deep Residual Learning for Image Recognition" for ImageNet architectures
	num_res and bottleneck vary according to Table 1
	ResNet18: [2, 2, 2, 2], ResNet34: [3, 4, 6, 3], (bottleneck) ResNet50: [3, 4, 6, 3], ResNet101: [3, 4, 23, 3], ResNet152: [3, 8, 36, 3]
	------------------
	shape: of input image with x/y dimension and channel as last dimension
	classes: number of image classes
	num_resblocks: is fixed to 4 (compared to 3 for CIFAR10)
	'''
	weight_init = 'he_uniform' # default: he_uniform, he_normal in ResNet paper
	weight_decay = tf.keras.regularizers.l2(0.0001)
	# Step 1 (Setup Input Layer)
	x_input = tf.keras.layers.Input(shape)
	x = tf.keras.layers.Conv2D(filters, kernel_size = 7, strides = 2, padding = 'same', kernel_initializer = weight_init, kernel_regularizer = weight_decay)(x_input)
	if preactivation == False:
		# In original preactivation implementation no activation is used
		# But the paper says this: "For the first Residual Unit (that follows a stand-alone convolutional layer, conv1),
		# we adopt the first activation right after conv1 and before splitting into two paths"
		# No MaxPooling is mentioned
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	# MaxPooling Layer in preactivation version???
	x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	# Step 2 (ResNet Layers)
	for ind in range(0, num_resblocks):
		if ind == 0:
			x = ResnetBlock(2 ** ind * filters, num_res, first_block = True, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x) # first_block = True
		else:
			x = ResnetBlock(2 ** ind * filters, num_res, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x)
	# Step 3 (Final Layers)
	if preactivation == True:
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.GlobalAvgPool2D()(x)
	x = tf.keras.layers.Dense(units = classes, activation = 'softmax')(x)
	model = tf.keras.models.Model(inputs = x_input, outputs = x, name = 'ResNet')
	return model



def resnet_cifar(shape = (32, 32, 3), classes = 10, filters = 16, num_res = 3, num_resblocks = 3, preactivation = True, bottleneck = False):
	'''Function returns ResNet for CIFAR with [6 * num_res + 2] layers
	shape: of input image with x/y dimension and channel as last dimension
	classes: number of image classes
	num_resblocks: is fixed to 3 for CIFAR10
	num_res: with 3 we arrive at ResNet20
	'''
	weight_init = 'he_uniform' # default: he_uniform, he_normal in ResNet paper
	weight_decay = tf.keras.regularizers.l2(0.0001)
	# Step 1 (Setup Input Layer)
	x_input = tf.keras.layers.Input(shape)
	x = tf.keras.layers.Conv2D(filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = weight_init, kernel_regularizer = weight_decay)(x_input)
	if preactivation == False:
		# In original preactivation implementation no activation is used
		# But the paper says this: "For the first Residual Unit (that follows a stand-alone convolutional layer, conv1),
		# we adopt the first activation right after conv1 and before splitting into two paths"
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	# MaxPooling Layer not in CIFAR10 version
	# x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	# Step 2 (ResNet Layers)
	for ind in range(0, num_resblocks):
		if ind == 0:
			x = ResnetBlock(2 ** ind * filters, num_res, first_block = True, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x)
		else:
			x = ResnetBlock(2 ** ind * filters, num_res, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x)
	# Step 3 (Final Layers)
	if preactivation == True:
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.GlobalAvgPool2D()(x)
	x = tf.keras.layers.Dense(units = classes, activation = 'softmax')(x)
	model = tf.keras.models.Model(inputs = x_input, outputs = x, name = 'ResNet_CIFAR10')
	return model


## Functions to create Autoencoder model of SINFONY via reparametrization trick

def resnet_cifar_tx(shape = (32, 32, 3), filters = 16, num_res = 2, num_resblocks = 3, preactivation = True, bottleneck = False, axnorm = 0, n_tx = -1, num_layer = 1):
	'''SINFONY: Function returns ResNet transmitter for CIFAR with [6 * num_res + 2] layers without bottleneck structure
	axnorm: axis for normalization of tx output (otherwise power goes to infinity...)
	n_tx: Tx/Rx layer length: (-1) without, (0) same length, (>0) adjust length
	'''
	weight_init = 'he_uniform' # default: he_uniform, he_normal in ResNet paper
	weight_decay = tf.keras.regularizers.l2(0.0001)
	# Tx
	# Step 1 (Setup Input Layer)
	intx = tf.keras.layers.Input(shape)
	x = tf.keras.layers.Conv2D(filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = weight_init, kernel_regularizer = weight_decay)(intx)
	if preactivation == False:
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	# MaxPooling Layer not in CIFAR10 version
	# x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	# Step 2 (ResNet Layers)
	for ind in range(0, num_resblocks):
		if ind == 0:
			x = ResnetBlock(2 ** ind * filters, num_res, first_block = True, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x)
		else:
			x = ResnetBlock(2 ** ind * filters, num_res, kernel_initializer = weight_init, kernel_regularizer = weight_decay, preactivation = preactivation, bottleneck = bottleneck)(x)
	# Step 3 (Final Layers)
	if preactivation == True:
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.GlobalAvgPool2D()(x)
	# Channel Encoding
	if n_tx > -1:
		if n_tx == 0:
			n_tx = x.shape[-1]
		# choose number of equal layers
		for ind in range(0, num_layer):
			x = Dense(n_tx, activation = 'relu', kernel_initializer = weight_init, name = "tx_layer" + str(ind))(x)	# Enc
		x = Dense(n_tx, activation = 'linear')(x) # linear layer here, or small constant in normalization -> then no numerical instability
	outtx = mt.normalize_input(x, axis = axnorm, eps = 1e-12)
	tx = Model(inputs = intx, outputs = outtx)
	# Classification at the receiver side
	return tx


def resnet_cifar_multitx(shape = (32, 32, 3), filters = 16, num_res = 2, num_resblocks = 3, preactivation = True, bottleneck = False, axnorm = 0, n_tx = -1, num_layer = 1, image_fac = 2):
	'''SINFONY: Function returns ResNet multiple distributed transmitters for CIFAR in one model
	image_fac: divide image by image_fac to generate image_fac * image_fac equal pieces
	'''
	im_div1 = shape[0] / image_fac
	im_div2 = shape[1] / image_fac
	tx_list = [[], []]
	for ind1 in range(0, image_fac):
		for ind2 in range(0, image_fac):
			shape_tx = (int((ind1 + 1) * im_div1) - int(ind1 * im_div1), int((ind2 + 1) * im_div2) - int(ind2 * im_div2), shape[-1])
			tx_list[ind1].append(resnet_cifar_tx(shape = shape_tx, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer))

	intx = Input(shape)
	outtx_list = [[], []]
	for ind1 in range(0, image_fac):
		for ind2 in range(0, image_fac):
			outtx_list[ind1].append(tf.expand_dims(tf.expand_dims(tx_list[ind1][ind2](intx[:, int(ind1 * im_div1):int((ind1 + 1) * im_div1), int(ind2 * im_div2):int((ind2 + 1) * im_div2), :]), axis = 1), axis = 2))
	outtx_x_list = []
	for ind2 in range(0, image_fac):
		outtx_x_list.append(tf.keras.layers.Concatenate(axis = 1)(outtx_list[:][ind2]))
	outtx = tf.keras.layers.Concatenate(axis = 2)(outtx_x_list)
	tx = Model(inputs = intx, outputs = outtx)
	return tx


def resnet_cifar_rx(shape, classes = 10, n_rx = -1, num_layer = 1, rx_same = 1, rx_linear = False, image_fac = 2):
	'''SINFONY: Function returns ResNet receiver
	shape: tx.layers[-1].output_shape[1:]
	classes: number of classes
	n_rx: layer width
	num_layer: number of layers
	rx_same: [0: separate rx, 1: same rx for all tx (default), 2: one joint rx]
	image_fac: divide image by image_fac to generate image_fac * image_fac equal pieces
	'''
	# Rx
	weight_init = 'he_uniform'	# default: he_uniform, he_normal in ResNet paper
	inrx = Input(shape = shape)
	# Equalization / Channel Decoding
	rx_list = False				# False: avoid list if rx_same == 1
	if n_rx > -1:
		if n_rx == 0:
			n_rx = shape[-1]
		if rx_same == 1:
			dec_layerlist = []
			for indl in range(0, num_layer):
				# All layers the same vs. adjustable here in code
				dec_layerlist.append(Dense(n_rx, activation = "relu", kernel_initializer = weight_init, name = "rx_layer" + str(indl)))
			Dec = tf.keras.Sequential(dec_layerlist)
		if rx_list == False and rx_same == 1:
		# x = Dec(inrx) acts on each dimension
			x = Dec(inrx)
		elif rx_list == True and rx_same == 1 or rx_same == 0:
			dec_list = [[], []]
			for ind1 in range(0, image_fac):
				for ind2 in range(0, image_fac):
					# Separate decoder at Rx for each Tx
					if rx_same == 0:
						dec_layerlist = []
						for indl in range(0, num_layer):
							dec_layerlist.append(Dense(n_rx, activation = "relu", kernel_initializer = weight_init, name = "rx_layer" + str(indl)))
						Dec = tf.keras.Sequential(dec_layerlist)
					dec_list[ind1].append(tf.expand_dims(tf.expand_dims(Dec(inrx[:, ind1, ind2,:]), axis = 1), axis = 2))
			dec_x_list = []
			for ind2 in range(0, image_fac):
				dec_x_list.append(tf.keras.layers.Concatenate(axis = 1)(dec_list[:][ind2]))
			x = tf.keras.layers.Concatenate(axis = 2)(dec_x_list)
		elif rx_same == 2:
			# One joint receiver for all inputs
			x = Flatten()(inrx)
			dec_layerlist = []
			for indl in range(0, num_layer):
				# layer are image_fac ** 2 times wider since inputs are concatenated
				dec_layerlist.append(Dense(image_fac ** 2 * n_rx, activation = "relu", kernel_initializer = weight_init, name = "rx_layer" + str(indl)))
			Dec = tf.keras.Sequential(dec_layerlist)
			x = Dec(x)
	else:
		x = inrx
	if rx_linear == True:
		# This final Rx module layer improves performance for the AE approach on rvec and at low SNR for SINFONY at the cost of a higher error floor
		x = Dense(n_rx, activation = 'linear')(x)
	if image_fac >= 2 and rx_same != 2:
		x = tf.keras.layers.GlobalAvgPool2D()(x)
	outrx = tf.keras.layers.Dense(units = classes, activation = 'softmax')(x)
	rx = Model(inputs = inrx, outputs = outrx)
	return rx


def resnet_cifar_ae(shape = (32, 32, 3), classes = 10, filters = 16, num_res = 2, num_resblocks = 3, preactivation = True, bottleneck = False, axnorm = 0, n_tx = -1, n_rx = -1, rx_same = 1, rx_linear = False, num_layer = 1, sigma = np.array([0, 0]), image_fac = 2):
	'''SINFONY: Autoencoder-like model via reparametrization trick
	Function returns ResNet multi transmitter autoencoder for CIFAR with [6 * num_res + 2] layers without bottleneck structure
	'''
	# Tx
	if image_fac >= 2:
		tx = resnet_cifar_multitx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer, image_fac = image_fac)
		label = 'ResNet_CIFAR10_AE_multitx'
	else:
		tx = resnet_cifar_tx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer)
		label = 'ResNet_CIFAR10_AE'

	# Rx
	rx = resnet_cifar_rx(shape = tx.layers[-1].output_shape[1:], classes = classes, n_rx = n_rx, num_layer = num_layer, rx_same = rx_same, rx_linear = rx_linear, image_fac = image_fac)

	# Model for autoencoder training
	intx = Input(shape)
	outtx = tx(intx)
	channel = mt.GaussianNoise2(sigma)(outtx)
	outrx = rx(channel)
	model = Model(inputs = intx, outputs = outrx, name = label)
	return model, tx, rx


## Reinforcement Learning version RL-SINFONY via Stochastic Policy Gradient

class resnet_cifar_rl(Model):
	'''RL-SINFONY via Stochastic Policy Gradient
	ResNet multi transmitter reinforcement learning for CIFAR with [6 * num_res + 2] layers without bottleneck structure
	'''
	def __init__(self, shape = (32, 32, 3), classes = 10, filters = 16, num_res = 2, num_resblocks = 3, preactivation = True, bottleneck = False, axnorm = 0, n_tx = -1, n_rx = -1, rx_same = 1, rx_linear = False, num_layer = 1, image_fac = 2):
		super().__init__()

		# self._training = training
		# self._sigma = sigma
		# self.perturbation_variance = perturbation_variance
		# Tx
		if image_fac >= 2:
			self.tx = resnet_cifar_multitx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer, image_fac = image_fac)
			# label = 'ResNet_CIFAR10_AE_multitx'
		else:
			self.tx = resnet_cifar_tx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer)
			# label = 'ResNet_CIFAR10_AE'
		# Rx
		self.rx = resnet_cifar_rx(shape = self.tx.layers[-1].output_shape[1:], classes = classes, n_rx = n_rx, num_layer = num_layer, rx_same = rx_same, rx_linear = rx_linear, image_fac = image_fac)
		# model = Model(inputs = intx, outputs = outrx, name = label)

	@tf.function#(jit_compile=True)
	def call(self, s, z, sigma, perturbation_variance = tf.constant(0.0, tf.float32)):
		x = self.tx(s) * tf.sqrt(1 - perturbation_variance) # Scaling to ensure conservation of average energy
		x_p = mt.GaussianNoise3(x, tf.sqrt([perturbation_variance, perturbation_variance]))
		y = mt.GaussianNoise3(x_p, sigma)
		y = tf.stop_gradient(y)		# no gradient between Tx and Rx
		z_hat = self.rx(y)

		# Average BCE for each baseband symbol and each batch example
		cce = tf.keras.losses.categorical_crossentropy(z, z_hat)
		# The RX loss is the usual average CE
		rx_loss = tf.reduce_mean(cce)
		
		## From the TX side, the CE is seen as a feedback from the RX through which backpropagation is not possible
		cce2 = tf.stop_gradient(cce)
		x_p2 = tf.stop_gradient(x_p)
		lnpxs = - tf.reduce_sum(tf.reduce_sum(tf.reduce_sum((x_p2 - x) ** 2, axis = -1), axis = -1), axis = -1) / (2 * perturbation_variance) # - 0.5 * tf.math.log((2 * np.pi * perturbation_variance) ** n_dim) # Gradient is backpropagated through `x`
		tx_loss = tf.reduce_mean(lnpxs * cce2, axis = 0)

		acc = tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(z_hat, axis = -1), tf.argmax(z, axis = -1)), dtype = 'float32'))
		return z_hat, tx_loss, rx_loss, acc



class resnet_cifar_ae2(Model):
	'''SINFONY trained via RL-SINFONY training procedure/function
	ResNet multi transmitter autoencoder-like defined like in reinforcement learning version for CIFAR with [6 * num_res + 2] layers without bottleneck structure
	'''
	def __init__(self, shape = (32, 32, 3), classes = 10, filters = 16, num_res = 2, num_resblocks = 3, preactivation = True, bottleneck = False, axnorm = 0, n_tx = -1, n_rx = -1, rx_same = 1, rx_linear = False, num_layer = 1, image_fac = 2):
		super().__init__()
		# Tx
		if image_fac >= 2:
			self.tx = resnet_cifar_multitx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer, image_fac = image_fac)
		else:
			self.tx = resnet_cifar_tx(shape = shape, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, num_layer = num_layer)
		# Rx
		self.rx = resnet_cifar_rx(shape = self.tx.layers[-1].output_shape[1:], classes = classes, n_rx = n_rx, num_layer = num_layer, rx_same = rx_same, rx_linear = rx_linear, image_fac = image_fac)

	@tf.function#(jit_compile=True)
	def call(self, s, z, sigma, perturbation_variance = tf.constant(0.0, tf.float32)):
		# perturbation_variance not used in AE approach, but placeholder to enable integration into RL-based training function
		x = self.tx(s)
		y = mt.GaussianNoise3(x, sigma)
		z_hat = self.rx(y)

		# Average BCE loss for each baseband symbol and each batch example
		cce = tf.keras.losses.categorical_crossentropy(z, z_hat)
		# The RX loss is the usual average CE
		rx_loss = tf.reduce_mean(cce)
		# The Tx loss is the same for AE
		tx_loss = rx_loss

		# Compute classification accuracy
		acc = tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(z_hat, axis = -1), tf.argmax(z, axis = -1)), dtype = 'float32'))
		return z_hat, tx_loss, rx_loss, acc


def get_batch(inputX, inputY, batch_size):
	'''Feed batch data
	'''
	for i in range(0, len(inputX) // batch_size):
		idx = i * batch_size
		yield inputX[idx:idx+batch_size], inputY[idx:idx+batch_size]

def shuffle_data(inputX, inputY):
	'''Shuffle training data
	'''
	perm = np.random.permutation(train_norm.shape[0])
	inputX = inputX[perm, ...]
	inputY = inputY[perm, ...]
	return inputX, inputY


class pertubation_variance_schedule():
	'''Exploration/pertubation variance schedule: piecewise constant
	values: Exploration/pertubation variance values
	boundaries: Iteration after which new value is adopted
	'''
	def __init__(self, values, boundaries):
		self._it = 0
		self._itb = 0
		self.boundaries = boundaries
		self.values = values
	def __call__(self):
		if self._itb != (len(self.values) - 1):
			if self._it >= self.boundaries[self._itb]:
				self._itb = self._itb + 1
		per_var = tf.constant(self.values[self._itb], tf.float32)
		self._it = self._it + 1
		return per_var


def rl_based_training(model, trainX, trainY, opt, opt_tx = None, opt_rx2 = None, valX = None, valY = None, epochs = 10, epochs_fine = 10, tx_steps = 10, rx_steps = 10, training_batch_size = 64, sigma = np.array([0, 0]), perturbation_var = 0.15, it_print = 1, zero_epoch = False):
	'''Reinforcement-based training of the semantic communication system model
	model: model with parameters to be trained
	trainX: training data set input
	trainY: training data set output
	opt: Receiver optimizer
	opt_tx: Transmitter optimizer
	opt_rx2: Receiver finetuning optimizer
	valX: Validation data set input
	valY: Validation data set output
	epochs: Number of training epochs
	epochs_fine: Number of epochs for receiver finetuning
	tx_steps: iterations of Tx training
	rx_steps: iterations of Rx training
	training_batch_size: Training batch size
	sigma: AWGN standard deviation
	perturbation_var: stochastic policy / RL-exploration variance
	it_print: Printer after it_print iterations
	zero_epoch: Evaluation on training and validation data before first training epoch
	'''
    # Optimizers used to apply gradients
	optimizer_rx = opt 					# For training the receiver
	if opt_tx == None:
		optimizer_tx = opt 				# For training the transmitter
	else:
		optimizer_tx = opt_tx
	if opt_rx2 == None:
		optimizer_rx2 = opt
	else:
		optimizer_rx2 = opt_rx2 		# For receiver finetuning
	total_steps = tx_steps + rx_steps

    # Function that implements one transmitter training iteration using RL.
	@tf.function
	def train_tx(opt_tx, trainX, trainY, sigma, perturbation_var = tf.constant(0.0, tf.float32)):
		# Forward pass
		with tf.GradientTape() as tape:
			# Keep only the TX loss
			_, tx_loss, rx_loss, acc = model(trainX, trainY, sigma, perturbation_var())
		## Computing and applying gradients
		weights = model.tx.trainable_weights
		grads = tape.gradient(tx_loss, weights)
		opt_tx.apply_gradients(zip(grads, weights))
		return rx_loss, acc, tx_loss
    
	# Function that implements one receiver training iteration
	@tf.function
	def train_rx(opt_rx, trainX, trainY, sigma):
		# Forward pass
		with tf.GradientTape() as tape:
			# Keep only the RX loss
			_, _, rx_loss, acc = model(trainX, trainY, sigma)  # No perturbation is added
		## Computing and applying gradients
		weights = model.rx.trainable_weights
		grads = tape.gradient(rx_loss, weights)
		opt_rx.apply_gradients(zip(grads, weights))
		# The RX loss is returned to print the progress
		return rx_loss, acc

	# Function that implements one finetuning receiver training iteration
	@tf.function
	def train_rx2(opt_rx2, trainX, trainY, sigma):
		# Forward pass
		with tf.GradientTape() as tape:
			# Keep only the RX loss
			_, _, rx_loss, acc = model(trainX, trainY, sigma)  # No perturbation is added
		## Computing and applying gradients
		weights = model.rx.trainable_weights # .rx.trainable_weights
		grads = tape.gradient(rx_loss, weights)
		opt_rx2.apply_gradients(zip(grads, weights))
		# The RX loss is returned to print the progress
		return rx_loss, acc
    
	# Save performance measures / results of training in dictionary
	perf_meas =	{
  		'rx_loss': [],
  		'acc': [],
  		'tx_loss': [],
  		'rx_val_loss': [],
  		'acc_val': [],
  		'tx_val_loss': [],
	}

	# Optional initial training and validation dataset evaluation of model before first training iteration:
	# Note: Not consistent with model.fit() output of SINFONY training history.
	if zero_epoch == True:
		_, tx_loss, rx_loss, acc = model(trainX, trainY, sigma, perturbation_var())
		_, tx_val_loss, rx_val_loss, acc_val = model(trainX, trainY, sigma, perturbation_var())
		perf_meas['rx_val_loss'].append(rx_val_loss.numpy())
		perf_meas['acc_val'].append(acc_val.numpy())
		perf_meas['tx_val_loss'].append(tx_val_loss.numpy())
		perf_meas['tx_loss'].append(tx_loss.numpy())
		perf_meas['rx_loss'].append(rx_loss.numpy())
		perf_meas['acc'].append(acc.numpy())

	# Training loop
	start_time0 = time.time()
	start_time = time.time()
	Ne = len(trainX) // training_batch_size
	for i in range(epochs):
		# Receiver training is performed first to keep it ahead of the transmitter
		# as it is used for computing the losses when training the transmitter
		ii = 0
		trainX, trainY = shuffle_data(trainX, trainY)
		for batchX, batchY in get_batch(trainX, trainY, training_batch_size):
			if ii % total_steps >= rx_steps:
				# One step of transmitter training
				rx_loss, acc, tx_loss = train_tx(optimizer_tx, batchX, batchY, sigma, perturbation_var = perturbation_var)
				if (valX is None) or (valY is None):
					print_str = '[Tx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}, Acc: {:.2f}, PG: {:.4f}'.format(i + 1, epochs, ii + 1, Ne, rx_loss.numpy(), acc.numpy(), tx_loss.numpy())
				else:
					_, tx_val_loss, rx_val_loss, acc_val = model(valX, valY, sigma, perturbation_var())
					print_str = '[Tx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}/{:.4f}, Acc: {:.2f}/{:.2f}, PG: {:.4f}/{:.4f}'.format(i + 1, epochs, ii + 1, Ne, rx_loss.numpy(), rx_val_loss.numpy(), acc.numpy(), acc_val.numpy(), tx_loss.numpy(), tx_val_loss.numpy())
					perf_meas['rx_val_loss'].append(rx_val_loss.numpy())
					perf_meas['acc_val'].append(acc_val.numpy())
					perf_meas['tx_val_loss'].append(tx_val_loss.numpy())
				perf_meas['tx_loss'].append(tx_loss.numpy())
			else:
				# One step of receiver training
				rx_loss, acc = train_rx(optimizer_rx, batchX, batchY, sigma)
				if (valX is None) or (valY is None):
					print_str = '[Rx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}, Acc: {:.2f}'.format(i + 1, epochs, ii + 1, Ne, rx_loss.numpy(), acc.numpy())
				else:
					_, _, rx_val_loss, acc_val = model(valX, valY, sigma)
					print_str = '[Rx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}/{:.4f}, Acc: {:.2f}/{:.2f}'.format(i + 1, epochs, ii + 1, Ne, rx_loss.numpy(), rx_val_loss.numpy(), acc.numpy(), acc_val.numpy())
					perf_meas['rx_val_loss'].append(rx_val_loss.numpy())
					perf_meas['acc_val'].append(acc_val.numpy())
			perf_meas['rx_loss'].append(rx_loss.numpy())
			perf_meas['acc'].append(acc.numpy())
			# Printing periodically the progress
			if ii % it_print == 0:
				print(print_str + ', Time: {:04.2f}s, Tot. time: '.format(time.time() - start_time) + print_time(time.time()- start_time0)) #, end = '\r')
				start_time = time.time()
			ii = ii + 1
    

    # Once alternating training is done, the receiver is fine-tuned.
	start_time = time.time()
	print('Receiver fine-tuning... ')
	for i in range(epochs_fine):
		ii = 0
		trainX, trainY = shuffle_data(trainX, trainY)
		for batchX, batchY in get_batch(trainX, trainY, training_batch_size):
			rx_loss, acc = train_rx2(optimizer_rx2, batchX, batchY, sigma)
			if (valX is None) or (valY is None):
				print_str = '[Rx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}, Acc: {:.2f}'.format(i + 1, epochs_fine, ii + 1, Ne, rx_loss.numpy(), acc.numpy())
			else:
				_, _, rx_val_loss, acc_val = model(valX, valY, sigma)
				print_str = '[Rx] Epoch: {}/{}, Batch: {}/{}, CE: {:.4f}/{:.4f}, Acc: {:.2f}/{:.2f}'.format(i + 1, epochs_fine, ii + 1, Ne, rx_loss.numpy(), rx_val_loss.numpy(), acc.numpy(), acc_val.numpy())
			perf_meas['rx_loss'].append(rx_loss)
			perf_meas['acc'].append(acc)
			if ii % it_print == 0:
				print(print_str + ', Time: {:04.2f}s, Tot. time: '.format(time.time() - start_time) + print_time(time.time()- start_time0)) #, end = '\r')
				start_time = time.time()
			ii = ii + 1

	return perf_meas



## Dataset functions
def load_dataset(dataset = 'mnist'):
	'''Load dataset
	mnist: Handwritten digits 0-9
	cifar10: Images of animals, vehicles, ..., with 10 classes
	fasion_mnist: Like mnist but with fashion images
	hirise: Images from Martian surface with crater classes, etc. Only available after download of hirise dataset
	hirisecrater: Like hirise, but combines all classes except for craters in one class
	'''
	# Load dataset
	if dataset == 'cifar10':
		(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
	elif dataset == 'mnist':
		(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
		# Reshape dataset to have a single channel
		trainX = trainX[..., np.newaxis]
		testX = testX[..., np.newaxis]
	elif dataset == 'fashion_mnist':
		(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
		# Reshape dataset to have a single channel
		trainX = trainX[..., np.newaxis]
		testX = testX[..., np.newaxis]
	elif dataset[0:6] == 'hirise':
		# Change directory
		data_dir = os.path.dirname(os.path.abspath(__file__)) + "/ImageDatasets/hirise-map-proj-v3_2/"
		data_file = data_dir + 'labels-map-proj_v3_2.txt'
		data_file2 = data_dir + 'labels-map-proj_v3_2_train_val_test.txt'
		# Pandas required for dataset import
		import pandas as pd
		X = pd.read_csv(data_file, sep = "\s", header = None, engine = 'python')	# X.loc[:, 0]
		X2 = pd.read_csv(data_file2, sep = "\s", header = None, engine = 'python')
		# Add index of alphanumerically ordered data set to X2
		indr = np.argmax(X.sort_values(by = 0, ascending = True).loc[:, 0].to_numpy() == X2.loc[:, 0].to_numpy()[:, None], axis = -1)
		X2[3] = indr.tolist()
		labels_list = X.sort_values(by = 0, ascending = True).loc[:, 1].to_numpy()
		class_names = pd.read_csv(data_dir + 'landmarks_map-proj-v3_2_classmap.csv', sep = ',', header = None, engine = 'python')
		# Lower resolution for first trials
		if dataset[-2:] == '32':
			image_res = (32, 32)
		elif dataset[-2:] == '64':
			image_res = (64, 64)
		elif dataset[-3:] == '128':
			image_res = (128, 128)
		else:
			# Full resolution
			image_res = (227, 227)
		train_ds = tf.keras.utils.image_dataset_from_directory(
					data_dir,
					labels = labels_list.tolist(),
					label_mode = 'int',
					color_mode = 'grayscale',
					# validation_split = 0.2,
					# subset = "training", # "validation"
					shuffle = False,
					# seed = 123,
					image_size = image_res, 
					)
		# Convert into numpy format
		dataX = np.concatenate([x for x, _ in train_ds], axis = 0)
		dataY = np.concatenate([y for _, y in train_ds], axis = 0)
		# Combine all classes except for craters in one class
		if dataset[:12] == 'hirisecrater':
			dataY = (dataY == 1) * 1
		# Training data set
		ind_set = X2.loc[X2.loc[:, 2] == 'train', 3].to_numpy()
		trainX = dataX[ind_set, ...]
		trainY = dataY[ind_set, ...]
		# Validation data set
		ind_set = X2.loc[X2.loc[:, 2] == 'val', 3].to_numpy()
		testX = dataX[ind_set, ...]
		testY = dataY[ind_set, ...]
		# Actual test set, not used so far...
		# ind_set = X2.loc[X2.loc[:, 2] == 'test', 3].to_numpy()
		# testX2 = dataX[ind_set, ...]
		# testY2 = dataY[ind_set, ...]
	else:
		print('Dataset not available.')
	# one hot encode target values
	trainY = tf.keras.utils.to_categorical(trainY)
	testY = tf.keras.utils.to_categorical(testY)
	return trainX, trainY, testX, testY


def prep_pixels(train, test):
	'''Preprocessing: Scale pixels between 0 and 1
	'''
	# Convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# Normalize to range 0-1 (avoiding saturation with typical NN activation functions)
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# Return normalized images
	return train_norm, test_norm

def summarize_dataset(trainx, trainy, testx, testy):
	'''Summarize loaded training dataset (trainx, trainy)
	and validation/test dataset (testx, testy)
	'''
	print('Train: X=%s, y=%s' % (trainx.shape, trainy.shape))
	print('Test: X=%s, y=%s' % (testx.shape, testy.shape))
	# Plot first few images
	for i in range(9):
		# Define subplot
		plt.subplot(330 + 1 + i)
		# Plot raw pixel data
		plt.imshow(trainx[i], cmap = plt.get_cmap('gray'))
	# Show the figure
	plt.show()


# Custom callbacks

class BatchTracking_Callback(tf.keras.callbacks.Callback):
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
	def on_train_batch_end(self, batch, logs = None):
		self.batch_end_loss.append(logs['loss'])
		self.batch_end_acc.append(logs['accuracy'])
		# self.batch.append(batch)




if __name__ == '__main__':
#     my_func_main()
# def my_func_main():
	
	# Initialization
	tf.keras.backend.clear_session()          	# Clearing graphs
	tf.keras.backend.set_floatx('float32')		# Computation accuracy: 'float16', 'float32', or 'float64'
	mt.GPU_sel(num = -2, memory_growth = 'True')# Choose/disable GPU: (-2) default, (-1) CPU, (>=0) GPU
	np.random.seed()            				# Random seed in every run, predictable random numbers for debugging with np.random.seed(0)

	# Simulation
	load = False										# Load model and reevaluate: False (default)
	evaluation_mode = 0									# Evaluation mode: (0) default: Validation for SNR range, (1) Saving probability data for interface to application, (2) t-SNE embedding for visualization
	filename = 'ResNet14_MNIST6_Ne20_layer2_rxlinear_snr-4_6'			# ResNet20_CIFAR4_RL_sgdlr2_Ne400_snr-4_6_0, ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv2, ResNet14_MNIST4_RL_snr-4_6_test, ResNet14_MNIST4_RL_sgd_Ne3000, ResNet52_rblock5_hirise128, ResNet14_MNIST#N, ResNet20_CIFAR#N
	path = 'models_sem'									# Sub path for saved data
	path0 = os.path.dirname(os.path.abspath(__file__))	# Path of script being executed
	pathfile = os.path.join(path0, path, filename)
	pathfile2 = os.path.join(path0, path, 'RES_' + filename)
	saveobj = savemodule(form = 'npz')

	# Data set
	dataset = 'mnist'			# mnist, cifar10, fashion_mnist, hirise64, hirisecrater
	show_dataset = True			# Show first dataset examples, just for demonstration
	trainx, trainy, testx, testy = load_dataset(dataset)

	# Training
	bs = 64						# Batch size, SGD: 128/64, Adam: 500
	optimizer = 'sgdlrs'		# sgd, adam, (sgdlrs) SGD with learning rate schedule
	lr = 1e-3					# Learning rate, SGD/Adam: 1e-3, RL: 1e-4
	iter_epoch = trainx.shape[0] / bs
	Ne = 20						# Number of epochs, 200 in CIFAR original implementation, 20 for MNIST
	# Choose validation data set:
	valX = testx[:100, ...]		# None, testx[:100, ...], testx[:1000, ...]
	valY = testy[:100, ...]		# None, testy[:100, ...], testy[:1000, ...]
	# RL training
	rl = 0						# (0) default AE, (1) Reinforcement learning training, (2) AE trained with rl-based training implementation
	it_print = 10				# Print after 1 (default) iterations training progress
	rx_steps = 10				# Sequential receiver batches
	tx_steps = 10				# Sequential transmitter batches
	expl_values = [0.15]		# [0.15, 0.15 ** 2] # with higher exploration variance, the gradient estimator variance decreases at the cost of more bias...
	per_epoch_bound = []		# [2000] # only activated during tx_train
	exp_boundaries = list(np.round(np.array(per_epoch_bound) / 2 * iter_epoch).astype('int'))
	expl_var = pertubation_variance_schedule(expl_values, exp_boundaries)
	Ne_fine = 200				# Receiver finetuning for RL-based training
	# NN Com system design
	ae = 1						# (0) only image recognition, (1) with (multi) com. system inbetween
	axnorm = 0					# Power normalization axis: (0) batch dimension, (1) encode vector dimension n_tx
	n_tx = 56					# 14/16| Tx layer length: (-1) without, (0) same length as layer before Tx, (>0) adjust length
	n_rx = 56					# 56/64| Rx layer length: (-1) without, (0) same length as Tx layer, (>0) adjust length
								# For comparison/orientation:
								# 1. One ReLU layer at Tx/Rx: [36/16]/64 in "Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach"
								# 2. Two ReLU layer at Tx: (128->)256->16 / at Rx: 256->128(->128) from "Deep Learning Enabled Semantic Communication Systems"
	num_layer = 1				# Number of Tx/Rx layers: 1 (default)
	rx_same = 1					# 0: individual rx, 1: same rx (default), 2: one joint rx
	rx_linear = False			# False: No final layer for each Rx module
	image_fac = 2				# Division of picture by 2 (default) to create 4 patches
	noise = True				# Training with noise: True (default)
	snr_min_train = -4 			# default: -4, 6
	snr_max_train = 6			# default: 6, 16
	N_iter = Ne * iter_epoch

	# Some training functionality of model.fit()
	if rl == False:
		verbose = 'auto'
		#- Callbacks for early stopping and model checkpoints -
		early_stopping = [] # EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
		model_checkp = [] 	# tf.keras.callbacks.ModelCheckpoint(pathfile, monitor = 'val_loss', verbose = verbose, save_best_only = False, mode = 'auto', period = 1, save_weights_only = False, save_freq = 'epoch')
		# Track training loss and accuracy of each batch iteration
		batch_tracking = BatchTracking_Callback()

	# Optimizer
	if optimizer.lower() == 'sgdlrs':
		# Learning rate schedules
		# Original ResNet: 1/2, 3/4 of training learning rate division by 10, in total 64k iterations
		epoch_bound = [3, 6]				# at 32000, 48000 iterations of 64000 in total: [100, 150] for CIFAR / [3, 6] for MNIST / [2, 50] for hirise / [100] for RL CIFAR
		boundaries = list(np.round(np.array(epoch_bound) * iter_epoch).astype('int'))
		values = [0.1, 0.01, 0.001]			# [0.1, 0.01, 0.001] for ae training / [0.001, 0.0001, 0.00001] for adam / [1e-3, 1e-4, 1e-5] for rl training / [1e-3, 1e-4] for rl CIFAR training sgdlr2
		lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
		opt = SGD(learning_rate = lr_schedule, momentum = 0.9) #, nesterov = True) # No advantage of Nesterov momentum with DNNs (?)
		opt_tx = SGD(learning_rate = lr_schedule, momentum = 0.9)
	elif optimizer.lower() == 'adam':
		# Adam and its variants
		opt = Adam(learning_rate = lr) 		# Optimizer for rx training # Nadam() # yogi() # Generalization/validation performance expected to be bad
		opt_tx = Adam(learning_rate = lr) 	# Optimizer for tx training
	else:
		# Default: Stochastic Gradient Descent with momentum 0.9 as in ResNet paper
		opt = SGD(learning_rate = lr, momentum = 0.9)
		opt_tx = SGD(learning_rate = lr, momentum = 0.9)
	opt_rx2 = None 						# Optimizer for rx finetuning: None (default, i.e., opt)
	
	# ResNet20 model
	classes = trainy.shape[1]			# 10 classes for CIFAR10, MNIST
	num_res = 2							# Defines ResNet layer number, 3 for smallest ResNet20 for CIFAR10 (2 for MNIST)
	if dataset == 'cifar10' and num_res <= 2:
		print('Warning: Number of residual units is below minimum number for CIFAR10 dataset!')
	num_resblocks = 3					# 3 for CIFAR10, MNIST
	dataset_shape = list(trainx.shape)
	filters = int(dataset_shape[1] / 2)	# 16 for CIFAR10 (adjust for MNIST with only 1 channel instead of 3?)
	preactivation = True				# True
	bottleneck = False					# False

	# Evaluation/Validation
	snr_eval = 20				# SNR value for interface data / T-SNE embedding: -10 / 20
	SNR_range = [-30, 20]		# SNR in dB range: [-30, 20] (default)
	step_size = 1				# SNR in dB steps: 1 (default)
	val_rounds = 10 			# Rounds through validation data with different noise realizations


	### TRAINING AND EVALUATION SCRIPT

	## Preprocessing
	train_norm, testnorm = prep_pixels(trainx, testx)
	# Summarize loaded dataset
	if show_dataset == True:
		summarize_dataset(trainx, trainy, testx, testy)


	## Create/load model
	resnet_lnum = 2 * num_resblocks * num_res + 2 # Pooling layers not counted / only conv2d + dense softmax, bottleneck structure for deeper architectures -> 3 instead of 2
	print('ResNet', resnet_lnum, ' chosen')

	if load == False:
		# Create new model:
		if ae == True:
			# Training w/o noise
			if noise == False:
				# training without noise
				sigma_train = np.array([0, 0])
			else:
				# training with noise
				sigma_train = mop.csigma(np.array([snr_min_train, snr_max_train]))[::-1]
			# Select SINFONY or RL-SINFONY
			if rl == 1:
				# RL-SINFONY
				model = resnet_cifar_rl(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, n_rx = n_rx, rx_same = rx_same, rx_linear = rx_linear, num_layer = num_layer, image_fac = image_fac)
			elif rl == 2:
				# SINFONY trained via RL-SINFONY training loop
				model = resnet_cifar_ae2(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, n_rx = n_rx, rx_same = rx_same, rx_linear = rx_linear, num_layer = num_layer, image_fac = image_fac)
			else:
				# SINFONY
				model, tx, rx = resnet_cifar_ae(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, n_rx = n_rx, rx_same = rx_same, rx_linear = rx_linear, num_layer = num_layer, sigma = sigma_train, image_fac = image_fac)
		else:
			# Standard image recognition based on total image
			model = resnet_cifar(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck)
	else:
		# Load existing model:
		print('Loading model...')
		if rl >= 1 and ae == True:
			# Load RL-SINFONY via weights
			if rl == 1:
				model = resnet_cifar_rl(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, n_rx = n_rx, rx_same = rx_same, num_layer = num_layer, image_fac = image_fac)
			elif rl == 2:
				model = resnet_cifar_ae2(shape = dataset_shape[1:], classes = classes, filters = filters, num_res = num_res, num_resblocks = num_resblocks, preactivation = preactivation, bottleneck = bottleneck, axnorm = axnorm, n_tx = n_tx, n_rx = n_rx, rx_same = rx_same, num_layer = num_layer, image_fac = image_fac)
			model.load_weights(os.path.join(path0, pathfile, filename))
		else:
			# Load SINFONY model
			model = tf.keras.models.load_model(pathfile)
		print('Model loaded.')

	if rl == False:
		# Summarize AE-based SINFONY
		model.summary()
	
	## Compile and train model
	if load == False:
		if rl >= 1:
			# RL-SINFONY
			sigma_train = tf.constant(sigma_train, dtype = 'float32')
			results = rl_based_training(model, train_norm, trainy, opt, opt_tx, opt_rx2, valX = valX, valY = valY, epochs = Ne, epochs_fine = Ne_fine, tx_steps = tx_steps, rx_steps = rx_steps, training_batch_size = bs, sigma = sigma_train, perturbation_var = expl_var, it_print = it_print)
			# Save model weights:
			print('Saving model weights...')
			model.save_weights(os.path.join(path0, pathfile, filename))
			print('Model weigths saved.')
			# Save training history to avoid data loss, if validation fails
			print('Save training history...')
		else:
			# SINFONY AE-like
			# Note: For loading, compile is not necessary: optimizer, loss and metric are saved with model
			model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
			history = model.fit(train_norm, trainy, epochs = Ne, batch_size = bs, validation_data = (testnorm, testy), callbacks = [batch_tracking, model_checkp, early_stopping], verbose = verbose)
			results = history.history
			# Save Keras model:
			print('Saving model...')
			model.save(pathfile)
			print('Model saved.')
			# Save history
			print('Save training history...')
			if results != {}:
				results['rx_loss'] = batch_tracking.batch_end_loss
				results['acc'] = batch_tracking.batch_end_acc
				# Save val_loss from model.fit() history under different name since it will be overwritten otherwise
				results['val_loss_history'] = results.pop('val_loss')
		saveobj.save(pathfile2, results)
		print('Saved!')
	else:
		# Load training history to include evaluation
		print('Load training history...')
		results = saveobj.load(pathfile2)
		if results == None:
			results = dict()
		else:
			results = dict(results)
		print('Loaded!')

	##  Evaluation of model
	SNR = np.linspace(SNR_range[0], SNR_range[1], int((SNR_range[1] - SNR_range[0]) / step_size) + 1)
	if evaluation_mode == 0:
		print('Evaluate model...')
		print(filename)
		## Evaluate model for different SNRs
		if ae == True:
			# SINFONY/RL-SINFONY
			start_time = time.time()
			eval_meas = [[], []]
			for ii, snr in enumerate(SNR):
				# Evaluate for each SNR in SNR range
				sigma = mop.csigma(snr)
				sigma_test = np.array([sigma, sigma], dtype = 'float32')
				if rl == 0:
					# Set standard deviation weights of Noise layer in AE approach
					model.layers[2].set_weights([sigma_test])
				lossi = 0
				acci = 0
				for ii2 in range(0, val_rounds):
					start_time2 = time.time()
					# Evaluate for val_rounds with different noise realizations (akin to training epochs)
					if rl >= 1:
						# RL-SINFONY validation step
						_, _, lossii, accii = model(testnorm, testy, sigma = tf.constant(sigma_test, dtype = 'float32'))
					else:
						# SINFONY validation step
						lossii, accii = model.evaluate(testnorm, testy)
					
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
			# Show performance curve
			plt.figure(1)
			plt.semilogy(SNR, 1 - acc)
			plt.figure(2)
			plt.semilogy(SNR, loss)
		else:
			# Standard image recognition: Evaluate model accuracy once for test data
			if rl >= 1:
				_, _, lossi, acci = model(testnorm, testy, sigma = tf.constant([0, 0], dtype = 'float32'))
			else:
				lossi, acci = model.evaluate(testnorm, testy)
			# Independent from SNR / constant, but plotted over SNR range
			loss = np.array(lossi) * np.ones(SNR.shape)
			acc = np.array(acci) * np.ones(SNR.shape)
			print('> %.3f' % (acci * 100.0))
		
		## Save evaluation
		print('Save evaluation...')
		results['snr'] = SNR
		results['val_loss'] = loss
		results['val_acc'] = acc
		saveobj.save(pathfile2, results)
		print('Evaluation saved.')
	elif evaluation_mode == 1:
		# Give results of training and validation data to application beyond
		print('Calculate interface data...')
		sigma = mop.csigma(snr_eval)
		sigma_test = np.array([sigma, sigma], dtype = 'float32')
		if rl == 1:
			probs_val, _, _, _ = model(testnorm, testy, sigma = tf.constant(sigma_test, dtype = 'float32'))
			probs_train, _, _, _ = model(train_norm, trainy, sigma = tf.constant(sigma_test, dtype = 'float32'))
		else:
			if ae == 1:
				model.layers[2].set_weights([sigma_test])			
			probs_val = model.predict(testnorm)
			probs_train = model.predict(train_norm)
		# Save interface data
		print('Save interface data...')
		pathfile2 = os.path.join(path0, path, 'output_' + filename)
		if ae == 1:
			pathfile2 = pathfile2 + '_snr' + str(snr_eval) + 'dB'
		results = dict()
		results['probs_val'] = probs_val
		results['probs_train'] = probs_train
		results['class_val'] = testy
		results['class_train'] = trainy
		saveobj.save(pathfile2, results)
		print('Interface data saved.')
	elif evaluation_mode == 2:
		# t-SNE embedding for visualization
		if rl == 0:
			# Script so far only works with SINFONY models
			from sklearn.manifold import TSNE
			from matplotlib.lines import Line2D
			cmap = plt.cm.jet										# define the colormap
			cmaplist = [cmap(i) for i in range(cmap.N)]				# extract all colors from the .jet map
			cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)	# create the new map
			# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
			X1 = model.layers[1](testnorm)		# Output of Tx layer
			sigma = mop.csigma(snr_eval)
			sigma_test = np.array([sigma, sigma])
			model.layers[2].set_weights([sigma_test])
			X2 = model.layers[2](X1)			# Channel
			X3 = model.layers[3].layers[0](X2)	# Input layer
			X4 = model.layers[3].layers[1](X3)	# Rx layer
			X5 = model.layers[3].layers[2](X4)	# Global average pooling layer
			## t-SNE
			# Choose output to cluster and visualize
			# X = X4[:, 0, 0, :]				# Output of Rx layer
			X = X5								# Output after Global average pooling layer, just before softmax layer
			X_embedded = TSNE(n_components = 2, learning_rate = 'auto', init = 'random').fit_transform(X)
			# Plot
			plt.figure(1)
			classes = np.argmax(testy, axis = -1)
			# Estimated labels
			# classes_est = np.argmax(model.predict(testnorm), axis = -1)
			# classes = classes_est
			plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = classes, cmap = cmap)
			custom_lines = []
			for cl in np.unique(classes):
				custom_lines.append(Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = cmaplist[int(cl * (len(cmaplist) - 1) / (testy.shape[-1] - 1))]))
			if dataset == 'cifar10':
				dlabel = ['0: Airplane', '1: Automobile', '2: Bird', '3: Cat', '4: Deer', '5: Dog', '6: Frog', '7: Horse', '8: Ship', '9: Truck']
			else:
				dlabel = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			plt.legend(custom_lines, dlabel, loc = 'center left', bbox_to_anchor = (1, 0.5))

# EOF