#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 14:58:13 2024

@author: beck
SINFONY architecture build from resnet

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

# LOADED PACKAGES
import numpy as np
# Tensorflow 2 packages
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

import resnet
import my_training as mt


# Functions to create Autoencoder model of SINFONY via reparametrization trick
# TODO: Define the functions as keras models?


class EncodingConfiguration():
    '''Encoding configuration class
    Configuration of encoding/transmitter
    normalization_axis: axis for normalization of transmitter output (otherwise power goes to infinity...)
    encoding_layer_width: Tx/Rx layer length: (-1) without, (0) same length, (>0) adjust length
    image_split_factor: divide image by image_split_factor to generate image_split_factor * image_split_factor equal pieces
    weight_initialization: default is he_uniform, he_normal in ResNet paper
    '''

    def __init__(self, transmit_normalization=True, normalization_axis=0, encoding_layer_width=-1, number_encoding_layer=1, image_split_factor=1, weight_initialization='he_uniform', weight_decay=None):
        self.transmit_normalization = transmit_normalization
        self.normalization_axis = normalization_axis
        self.encoding_layer_width = encoding_layer_width
        self.number_encoding_layer = number_encoding_layer
        self.image_split_factor = image_split_factor
        self.weight_initialization = weight_initialization
        self.weight_decay = weight_decay


class NoEncodingConfiguration(EncodingConfiguration):
    '''Encoding configuration class to turn off encoding
    '''

    def __init__(self, *args, transmit_normalization=False, encoding_layer_width=-1, **kwargs):
        super().__init__(*args, transmit_normalization=transmit_normalization,
                         encoding_layer_width=encoding_layer_width, **kwargs)


class DecodingConfiguration():
    '''Decoding configuration class
    Configuration of decoding/receiver layers
    decoding_layer_width: layer width
    number_decoding_layer: number of decoding layers
    rx_joint_layers: [0: separate receiver, 1: same receiver for all transmitter (default), 2: one joint receiver]
    image_split_factor: divide image by image_split_factor to generate image_split_factor * image_split_factor equal pieces
    weight_initialization: default is he_uniform, he_normal in ResNet paper
    '''

    def __init__(self, decoding_layer_width=-1, number_decoding_layer=1, rx_joint_layers=1, rx_final_layer_linear=False, weight_initialization='he_uniform', weight_decay=None):
        self.decoding_layer_width = decoding_layer_width
        self.number_decoding_layer = number_decoding_layer
        self.rx_joint_layers = rx_joint_layers
        self.rx_final_layer_linear = rx_final_layer_linear
        self.weight_initialization = weight_initialization
        self.weight_decay = weight_decay


class CommunicationChannel():
    '''Communication channel class
    Allows for use of different channels
    '''

    def __init__(self, noise_standard_deviation_interval=np.array([0, 0])):
        self.noise_standard_deviation_interval = noise_standard_deviation_interval
        self.channel = mt.GaussianNoise2(
            self.noise_standard_deviation_interval)


class CommunicationConfiguration():
    '''Communication configuration class
    Configuration of encoding/transmitter and decoding/receiver layers
    '''

    def __init__(self, *args, encoding_config=EncodingConfiguration(), decoding_config=DecodingConfiguration(), communication_channel, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoding_config = encoding_config
        self.decoding_config = decoding_config
        self.communication_channel = communication_channel


def encoding_layers(input_shape, encoding_config=EncodingConfiguration()):
    '''Encoding layers for channel encoding of features
    '''
    transmitter_input = tf.keras.layers.Input(input_shape)
    x_tensor = transmitter_input
    encoding_layer_width = encoding_config.encoding_layer_width
    if encoding_layer_width > -1:
        if encoding_layer_width == 0:
            encoding_layer_width = x_tensor.shape[-1]
        # Choose number of equal width layers
        for layer_number in range(0, encoding_config.number_encoding_layer):
            x_tensor = Dense(encoding_layer_width, activation='relu', kernel_initializer=encoding_config.weight_initialization, kernel_regularizer=encoding_config.weight_decay,
                             name="tx_layer" + str(layer_number))(x_tensor)
        # Linear layer here, or small constant in normalization -> then no numerical instability
        x_tensor = Dense(encoding_layer_width, activation='linear',
                         kernel_regularizer=encoding_config.weight_decay)(x_tensor)
    if encoding_config.transmit_normalization is True:
        transmitter_output = mt.normalize_input(
            x_tensor, axis=encoding_config.normalization_axis, epsilon=1e-12)
    else:
        transmitter_output = x_tensor
    encoder = Model(inputs=transmitter_input, outputs=transmitter_output)
    return encoder


def resnet_transmitter(resnet_config=resnet.ResnetConfiguration(), encoding_config=EncodingConfiguration()):
    '''SINFONY: Function returns ResNet transmitter (CIFAR with [6 * number_residual_units + 2] layers without bottleneck structure)
    '''
    resnet_config.image_shape = resnet.list2first_element(
        resnet_config.image_shape)
    # Tx
    # Step 1 (Setup Input Layer)
    transmitter_input = tf.keras.layers.Input(resnet_config.image_shape)
    # Step 2 (ResNet Layers)
    feature_extractor = resnet.resnet_feature_extractor(
        resnet_config=resnet_config)
    x_tensor = feature_extractor(transmitter_input)

    # Step 3 (Channel Encoding)
    encoder = encoding_layers(
        input_shape=feature_extractor.layers[-1].output_shape[1:], encoding_config=encoding_config)
    transmitter_output = encoder(x_tensor)

    transmitter = Model(inputs=transmitter_input, outputs=transmitter_output)
    # Classification at the receiver side
    return transmitter


def resnet_multi_transmitter_imagesplit(resnet_config=resnet.ResnetConfiguration(), encoding_config=EncodingConfiguration()):
    '''SINFONY: Function returns ResNet-based multiple distributed transmitters for CIFAR in one model
    '''
    # TODO: Merge with default multi-transmitter?
    image_shape = resnet.list2first_element(
        resnet_config.image_shape)
    im_div1 = image_shape[0] / encoding_config.image_split_factor
    im_div2 = image_shape[1] / encoding_config.image_split_factor
    tx_list = [[], []]
    resnet_config_tx = resnet_config
    for index_x in range(0, encoding_config.image_split_factor):
        for index_y in range(0, encoding_config.image_split_factor):
            shape_tx = (int((index_x + 1) * im_div1) - int(index_x * im_div1),
                        int((index_y + 1) * im_div2) - int(index_y * im_div2), image_shape[-1])
            resnet_config_tx.image_shape = shape_tx
            tx_list[index_x].append(resnet_transmitter(
                resnet_config=resnet_config_tx, encoding_config=encoding_config))
    # Reset image shape
    resnet_config.image_shape = image_shape

    transmitter_input = Input(image_shape)
    transmitter_output_list = [[], []]
    for index_x in range(0, encoding_config.image_split_factor):
        for index_y in range(0, encoding_config.image_split_factor):
            transmitter_output_list[index_x].append(tf.expand_dims(tf.expand_dims(tx_list[index_x][index_y](transmitter_input[:, int(
                index_x * im_div1):int((index_x + 1) * im_div1), int(index_y * im_div2):int((index_y + 1) * im_div2), :]), axis=1), axis=2))
    transmitter_output_x_list = []
    for index_y in range(0, encoding_config.image_split_factor):
        transmitter_output_x_list.append(tf.keras.layers.Concatenate(
            axis=1)(transmitter_output_list[:][index_y]))
    transmitter_output = tf.keras.layers.Concatenate(
        axis=2)(transmitter_output_x_list)
    transmitter = Model(inputs=transmitter_input, outputs=transmitter_output)
    return transmitter


def resnet_receiver_imagesplit(received_signal_shape, number_classes=10, image_split_factor=2, decoding_config=DecodingConfiguration()):
    '''SINFONY: Function returns ResNet receiver
    received_signal_shape: transmitter.layers[-1].output_shape[1:]
    number_classes: number of classes
    '''
    decoding_layer_width = decoding_config.decoding_layer_width
    number_decoding_layer = decoding_config.number_decoding_layer
    # Rx
    receiver_input = Input(shape=received_signal_shape)
    # Equalization / Channel Decoding
    rx_list = False				# False: avoid list if rx_joint_layers == 1
    if decoding_layer_width > -1:
        if decoding_layer_width == 0:
            decoding_layer_width = received_signal_shape[-1]
        if decoding_config.rx_joint_layers == 1:
            decoder_layerlist = []
            for indl in range(0, number_decoding_layer):
                # All layers the same vs. adjustable here in code
                decoder_layerlist.append(Dense(
                    decoding_layer_width, activation="relu", kernel_initializer=decoding_config.weight_initialization, kernel_regularizer=decoding_config.weight_decay, name="rx_layer" + str(indl)))
            decoder = tf.keras.Sequential(decoder_layerlist)
        if rx_list is False and decoding_config.rx_joint_layers == 1:
            # x_tensor = decoder(receiver_input) acts on each dimension
            x_tensor = decoder(receiver_input)
        elif rx_list is True and decoding_config.rx_joint_layers == 1 or decoding_config.rx_joint_layers == 0:
            decoder_list = [[], []]
            for index_x in range(0, image_split_factor):
                for index_y in range(0, image_split_factor):
                    # Separate decoder at Rx for each Tx
                    if decoding_config.rx_joint_layers == 0:
                        decoder_layerlist = []
                        for indl in range(0, number_decoding_layer):
                            decoder_layerlist.append(Dense(
                                decoding_layer_width, activation="relu", kernel_initializer=decoding_config.weight_initialization, kernel_regularizer=decoding_config.weight_decay, name="rx_layer" + str(indl)))
                        decoder = tf.keras.Sequential(decoder_layerlist)
                    decoder_list[index_x].append(tf.expand_dims(tf.expand_dims(
                        decoder(receiver_input[:, index_x, index_y, :]), axis=1), axis=2))
            decoder_x_list = []
            for index_y in range(0, image_split_factor):
                decoder_x_list.append(tf.keras.layers.Concatenate(
                    axis=1)(decoder_list[:][index_y]))
            x_tensor = tf.keras.layers.Concatenate(axis=2)(decoder_x_list)
        elif decoding_config.rx_joint_layers == 2:
            # One joint receiver for all inputs
            decoder_layerlist = []
            for indl in range(0, number_decoding_layer):
                # layer are image_split_factor ** 2 times wider since inputs are concatenated
                decoder_layerlist.append(Dense(image_split_factor ** 2 * decoding_layer_width, activation="relu",
                                               kernel_initializer=decoding_config.weight_initialization, kernel_regularizer=decoding_config.weight_decay, name="rx_layer" + str(indl)))
            decoder = tf.keras.Sequential(decoder_layerlist)
            x_tensor = Flatten()(receiver_input)
            x_tensor = decoder(x_tensor)
    else:
        x_tensor = receiver_input
    if decoding_config.rx_final_layer_linear is True:
        # This final Rx module layer improves performance for the AE approach on rvec and at low SNR for SINFONY at the cost of a higher error floor
        x_tensor = Dense(decoding_layer_width, activation='linear')(x_tensor)
    if image_split_factor >= 2 and decoding_config.rx_joint_layers != 2:
        x_tensor = tf.keras.layers.GlobalAvgPool2D()(x_tensor)
    receiver_output = tf.keras.layers.Dense(
        units=number_classes, activation='softmax')(x_tensor)
    receiver = Model(inputs=receiver_input, outputs=receiver_output)
    return receiver


def resnet_sinfony_imagesplit(communication_config, resnet_config=resnet.ResnetConfiguration()):
    '''SINFONY: Autoencoder-like model via reparametrization trick
    Function returns ResNet multi transmitter autoencoder for CIFAR with [6 * number_residual_units + 2] layers without bottleneck structure
    '''
    # Tx
    image_split_factor = communication_config.encoding_config.image_split_factor
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        resnet_config.number_resnet_blocks, resnet_config.number_residual_units, resnet_config.bottleneck)
    if image_split_factor >= 2:
        transmitter = resnet_multi_transmitter_imagesplit(
            resnet_config=resnet_config, encoding_config=communication_config.encoding_config)
        label = f'ResNet{resnet_layer_number}_AE_multitx_{resnet_config.architecture.upper()}'
    else:
        transmitter = resnet_transmitter(resnet_config=resnet_config,
                                         encoding_config=communication_config.encoding_config)
        label = f'ResNet{resnet_layer_number}_AE_{resnet_config.architecture.upper()}'

    # Rx
    receiver = resnet_receiver_imagesplit(received_signal_shape=transmitter.layers[-1].output_shape[1:], number_classes=resnet_config.number_classes,
                                          image_split_factor=image_split_factor, decoding_config=communication_config.decoding_config)

    # Model for autoencoder training
    transmitter_input = Input(resnet_config.image_shape)
    transmitter_output = transmitter(transmitter_input)
    channel = communication_config.communication_channel.channel(
        transmitter_output)
    receiver_output = receiver(channel)
    model = Model(inputs=transmitter_input,
                  outputs=receiver_output, name=label)
    return model, transmitter, receiver


# Implementations for list of images


def resnet_multi_transmitter(resnet_config=resnet.ResnetConfiguration(), encoding_config=EncodingConfiguration(), only_features=False):
    '''SINFONY: Function returns ResNet-based multiple distributed transmitters for list of input images in one model
    only_features: Turn off encoding/power normalization and just provide features instead of transmit signals (redundent)
    '''
    image_shapes = resnet_config.image_shape
    number_filters = resnet_config.number_filters
    if not isinstance(number_filters, list):
        number_filters = [number_filters] * len(image_shapes)
    # Create image inputs and transmit signals
    images = []
    transmit_signals = []
    resnet_config_tx = resnet_config
    for shape_index, image_shape in enumerate(image_shapes):
        image = Input(image_shape)
        images.append(image)
        resnet_config_tx.image_shape = image_shape
        # Adjust the number_filters number per Tx to the input image in the list
        resnet_config_tx.number_filters = number_filters[shape_index]
        if only_features is True:
            transmit_signal = resnet.resnet_feature_extractor(
                resnet_config=resnet_config_tx)(image)
        else:
            transmit_signal = resnet_transmitter(
                resnet_config=resnet_config_tx, encoding_config=encoding_config)(image)
        transmit_signals.append(transmit_signal)
    resnet_config.image_shape = image_shapes
    # Concatenate the transmit signals
    transmit_signals = tf.keras.layers.Concatenate(
        axis=-1)(transmit_signals)
    transmitter = Model(inputs=images, outputs=transmit_signals)
    return transmitter


def resnet_receiver(received_signal_shape, number_classes=10, decoding_config=DecodingConfiguration()):
    '''SINFONY: Function returns ResNet receiver for list of input images
    received_signal_shape: transmiters.layers[-1].output_shape[1:]
    '''
    # Rx
    received_signals = Input(shape=received_signal_shape)
    x_tensor = received_signals
    decoding_layer_width = decoding_config.decoding_layer_width
    # Equalization / Channel Decoding = Fully connected ReLU layers
    if decoding_layer_width > -1:
        if decoding_layer_width == 0:
            decoding_layer_width = received_signal_shape[-1]
        for layer_number in range(0, decoding_config.number_decoding_layer):
            decoder = Dense(decoding_layer_width, activation="relu", kernel_initializer=decoding_config.weight_initialization, kernel_regularizer=decoding_config.weight_decay,
                            name="rx_layer" + str(layer_number))
            x_tensor = decoder(x_tensor)
    # Final soft classifier
    probabilityclasses = tf.keras.layers.Dense(
        units=number_classes, activation='softmax')(x_tensor)
    receiver = Model(inputs=received_signals, outputs=probabilityclasses)
    return receiver


def resnet_sinfony(communication_config, resnet_config=resnet.ResnetConfiguration()):
    '''SINFONY: Autoencoder-like model via reparametrization trick
    Function returns ResNet multi transmitter autoencoder for list of input images without bottleneck structure
    ResNet according to CIFAR implementation with [6 * number_residual_units + 2] layers
    image_shapes: List of image input shapes
    '''
    image_shapes = resnet_config.image_shape
    # Tx
    transmitters = resnet_multi_transmitter(
        resnet_config=resnet_config, encoding_config=communication_config.encoding_config)
    # Rx
    receiver = resnet_receiver(
        received_signal_shape=transmitters.layers[-1].output_shape[1:], number_classes=resnet_config.number_classes, decoding_config=communication_config.decoding_config)

    # Model for autoencoder training
    images = []
    for image_shape in image_shapes:
        image_input = Input(image_shape)
        images.append(image_input)
    transmit_signals = transmitters(images)
    channel = communication_config.communication_channel.channel(
        transmit_signals)
    probabilityclasses = receiver(channel)
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        resnet_config.number_resnet_blocks, resnet_config.number_residual_units, resnet_config.bottleneck)
    model = Model(inputs=images, outputs=probabilityclasses,
                  name=f'ResNet{resnet_layer_number}_AE_multi_transmitter_{resnet_config.architecture.upper()}')
    return model, transmitters, receiver


def resnet_multi_image(resnet_config=resnet.ResnetConfiguration(), number_combination_layer=0, combination_layer_width=0):
    '''Function returns architecture of ResNet for list of images
    architecture: Choose architecture tailored to specific dataset
    shape: of input image with x/y dimension and channel as last dimension
    number_classes: number of image classes
    weight_initialization: 'he_uniform' is default: he_uniform, he_normal in ResNet paper
    '''
    image_shapes = resnet_config.image_shape
    feature_extractors = resnet_multi_transmitter(
        resnet_config=resnet_config, encoding_config=NoEncodingConfiguration(), only_features=True)
    # Model for multi image classification training
    images = []
    for image_shape in image_shapes:
        image_input = Input(image_shape)
        images.append(image_input)
    features = feature_extractors(images)

    x_tensor = features
    # Joint preprocessing of features
    if combination_layer_width <= 0:
        combination_layer_width = feature_extractors.layers[-1].output_shape[-1]
    # number_combination_layer = 0
    if number_combination_layer > 0:
        for layer_number in range(0, number_combination_layer):
            decoder = Dense(combination_layer_width, activation="relu", kernel_initializer=resnet_config.weight_initialization, kernel_regularizer=resnet_config.weight_decay,
                            name="feature_processing_layer" + str(layer_number))
            x_tensor = decoder(x_tensor)
    # Summation of log-likelihood ratios
    probabilityclasses = tf.keras.layers.Dense(
        units=resnet_config.number_classes, activation='softmax')(x_tensor)

    resnet_layer_number = resnet.calculate_resnet_layer_number(
        resnet_config.number_resnet_blocks, resnet_config.number_residual_units, resnet_config.bottleneck)
    model = Model(inputs=images, outputs=probabilityclasses,
                  name=f'ResNet{resnet_layer_number}_multi_images_{resnet_config.architecture.upper()}')
    return model
