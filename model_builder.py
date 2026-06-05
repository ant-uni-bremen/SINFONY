#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 2026

@author: beck
Model builder module for SINFONY framework
This module provides functions to build different types of models in a modular way.
"""

import numpy as np
import keras
# import tensorflow.keras as keras

# Own packages
import sinfony_architectures.resnet as resnet
import sinfony_architectures.resnet_sinfony as resnet_sinfony
import sinfony_architectures.resnet_rl_sinfony as resnet_rl_sinfony
import utilities.my_math_operations as mop
from keras.optimizers import SGD, Adam  # , Nadam
from utilities.my_layers_keras3 import epochiterations2iterationboundaries


def create_model_from_config(params, number_classes, image_shapes):
    """
    Create a model based on the provided configuration parameters.

    Args:
        params (dict): Dictionary containing configuration parameters

    Returns:
        model: The constructed model
    """
    # Extract configuration sections
    # dataset_settings = params['dataset']
    rl = params['reinforcement_learning']['active']
    model_settings = params['model']
    transceiver_split = model_settings['communication']['transceiver_split']

    resnet_config = create_resnet(
        model_settings, number_classes, image_shapes)
    communication_config, sigma_train = create_communication_module(
        model_settings)

    model = select_model(image_shapes, resnet_config, communication_config,
                         transceiver_split=transceiver_split, reinforcement_learning=rl)
    return model, sigma_train


def create_model(settings_path, number_classes, image_shapes):
    """
    Create a model from a YAML settings file.

    Args:
        settings_path (str): Path to the YAML settings file

    Returns:
        model: The constructed model
    """
    import yaml

    # Load configuration
    with open(settings_path, 'r', encoding='UTF8') as file:
        params = yaml.safe_load(file)

    model, sigma_train = create_model_from_config(
        params, number_classes, image_shapes)

    return model, sigma_train


def select_model(image_shapes, resnet_config, communication_config, transceiver_split=0, reinforcement_learning=0):
    ''' Select model
    '''

    # Check whether multi-image input and choose model accordingly
    if len(image_shapes) == 1:
        # Create new model:
        if transceiver_split == 1:
            # Select SINFONY or RL-SINFONY
            if reinforcement_learning == 1:
                # RL-SINFONY
                model = resnet_rl_sinfony.ResnetRLSinfony(
                    resnet_config=resnet_config, communication_config=communication_config)
            elif reinforcement_learning == 2:
                # SINFONY trained via RL-SINFONY training loop
                model = resnet_rl_sinfony.ResnetAE2(
                    resnet_config=resnet_config, communication_config=communication_config)
            else:
                # SINFONY
                model, tx, rx = resnet_sinfony.resnet_sinfony_imagesplit(
                    resnet_config=resnet_config, communication_config=communication_config)
        else:
            # Standard image recognition based on total image
            model = resnet.resnet(resnet_config=resnet_config)
    else:
        # Multiple images models
        if transceiver_split == 1:
            if reinforcement_learning == 1:
                print('Not implemented yet.')
            else:
                model, tx, rx = resnet_sinfony.resnet_sinfony(
                    resnet_config=resnet_config, communication_config=communication_config)
        else:
            model = resnet_sinfony.resnet_multi_image(
                resnet_config=resnet_config)
    return model


def create_resnet(model_settings, number_classes, image_shapes):
    '''Create ResNet configuration object from params dictionary
    '''

    # Getting image data
    # number_classes = test_labels.shape[1]
    # image_shapes = [image_dataset.shape[1:] for image_dataset in test_input]

    # Extract configuration sections
    # model_settings = params['model']

    # Configure regularization
    weight_decay = keras.regularizers.l2(
        model_settings['resnet']['weight_decay']) if model_settings['resnet']['weight_decay'] != 0 else None

    # Configure ResNet, ResNet20 model
    # Defines ResNet layer number, 3 for smallest ResNet20 for CIFAR10 (2 for MNIST)
    number_residual_units = model_settings['resnet']['number_residual_units']
    # 3 for CIFAR10, MNIST
    number_resnet_blocks = model_settings['resnet']['number_resnet_blocks']

    # Handle number_filters
    number_filters = model_settings['resnet']['number_filters']
    if len(image_shapes) == 1 and isinstance(number_filters, list):
        # If only one image, then use first number of filters entry
        number_filters = number_filters[0]

    # Convert number_residual_units to a list whose length matches number_resnet_blocks if not already
    # NOTE: Then, the first element is repeated!
    number_residual_units = resnet.repeat_entry_2_list(
        number_residual_units, number_resnet_blocks)

    # Create ResNet configuration
    resnet_config = resnet.ResnetConfiguration(
        architecture=model_settings['resnet']['architecture'],
        image_shape=image_shapes,
        number_classes=number_classes,
        number_filters=number_filters,
        number_residual_units=number_residual_units,
        number_resnet_blocks=number_resnet_blocks,
        preactivation=model_settings['resnet']['preactivation'],
        bottleneck=model_settings['resnet']['bottleneck'],
        batch_normalization=model_settings['resnet']['batch_normalization'],
        weight_initialization=model_settings['resnet']['weight_initialization'],
        weight_decay=weight_decay,
        number_combination_layer=model_settings['resnet']['multi_image_layer_number'],
        combination_layer_width=model_settings['resnet']['multi_image_layer_width'],

    )

    # Create model based on configuration
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        number_resnet_blocks, number_residual_units, model_settings['resnet']['bottleneck'])
    print('ResNet', resnet_layer_number, ' chosen')

    return resnet_config


def create_communication_module(model_settings):
    '''Create communication configuration object from params dictionary
    NN Com system design
    '''

    # Extract settings
    # (0) only image recognition, (1) with (multi) com. system inbetween
    transceiver_split = model_settings['communication']['transceiver_split']

    # Configure communication components if needed
    if transceiver_split == 1:

        # Number of Tx/Rx layers: 1 (default)
        number_layer = model_settings['communication']['number_txrx_layer']

        # Configure regularization
        # TODO: l2 regularization only for ResNet feature extractor? Yes, better performance
        weight_decay_communication = keras.regularizers.l2(
            model_settings['communication']['weight_decay']) if model_settings['communication']['weight_decay'] != 0 else None

        # Create encoding configuration
        encoding_config = resnet_sinfony.EncodingConfiguration(
            transmit_normalization=True,
            normalization_axis=model_settings['communication']['power_normalization_axis'],
            encoding_layer_width=model_settings['communication']['number_channel_uses'],
            number_encoding_layer=number_layer,
            image_split_factor=model_settings['communication']['image_split_factor'],
            weight_initialization=model_settings['communication']['weight_initialization'],
            weight_decay=weight_decay_communication,
        )

        # Create decoding configuration
        decoding_config = resnet_sinfony.DecodingConfiguration(
            decoding_layer_width=model_settings['communication']['rx_layer_width'],
            number_decoding_layer=number_layer,
            rx_joint_layers=model_settings['communication']['rx_same'],
            rx_final_layer_linear=model_settings['communication']['rx_linear'],
            weight_initialization=model_settings['communication']['weight_initialization'],
            weight_decay=weight_decay_communication
        )

        # Configure noise parameters
        if model_settings['noise']['active'] is False:
            # Training w/o noise
            sigma_train = np.array([0, 0])
        else:
            # Training with noise
            sigma_train = mop.snr2standard_deviation(
                np.array([model_settings['noise']['snr_min_train'], model_settings['noise']['snr_max_train']]))[::-1]

        # Create communication channel
        communication_channel = resnet_sinfony.CommunicationChannel(
            sigma_train)
        communication_config = resnet_sinfony.CommunicationConfiguration(
            encoding_config=encoding_config,
            decoding_config=decoding_config,
            communication_channel=communication_channel,
        )
    else:
        communication_config = None
        sigma_train = None

    return communication_config, sigma_train


def create_optimizer(
    training_settings: dict,
    rl_settings: dict,
    iterations_per_epoch: int,
    learning_rate: float,
    optimizer: str,
    rl: int,
):
    """
    Creates optimizer and SPG configuration based on the training settings.

    Args:
        training_settings:      Dict with training parameters (lr_schedule, momentum, etc.)
        rl_settings:            Dict with Reinforcement Learning settings
        iterations_per_epoch:   Number of iterations per epoch
        learning_rate:          Default learning rate (overwritten if schedule is active)
        optimizer:              'adam' or 'sgd'
        rl:                     0 = no RL, otherwise RL active
        exploration_variance:   Variance schedule for exploration (only for RL)

    Returns:
        optimizer:      Keras optimizer for RX
        optimizer_tx:   Keras optimizer for TX (None if no RL)
        optimizer_rx2:  Keras optimizer for RX finetuning (None if not configured)
        spg_config:     SPG configuration (None if no RL)
    """

    # --- Exploration Variance Schedule (only for RL) ---
    if rl != 0:
        # Higher exploration variance → lower variance of gradient estimator, but more bias
        # e.g. [0.15, 0.15**2]
        exploration_values = rl_settings['exploration_variance']
        # e.g. [2000] - only active during tx_train
        per_epoch_bound = rl_settings['exploration_boundaries']

        exploration_boundaries = epochiterations2iterationboundaries(
            per_epoch_bound, iterations_per_epoch / 2)

        exploration_variance = resnet_rl_sinfony.PertubationVarianceSchedule(
            exploration_values, exploration_boundaries
        )
    else:
        exploration_variance = None

    # --- Learning Rate Schedule ---
    if training_settings['learning_rate_schedule']['active']:
        epoch_bound = training_settings['learning_rate_schedule']['epoch_bound']

        boundaries = epochiterations2iterationboundaries(
            epoch_bound, iterations_per_epoch)

        values = training_settings['learning_rate_schedule']['values']

        learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values
        )

    # --- Momentum ---
    momentum = training_settings['momentum']

    # --- Create optimizer ---
    if optimizer.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
        optimizer_tx = Adam(learning_rate=learning_rate) if rl != 0 else None

    else:  # Default: SGD with momentum 0.9 as in ResNet paper
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        optimizer_tx = SGD(learning_rate=learning_rate,
                           momentum=momentum) if rl != 0 else None

    # --- RX finetuning optimizer ---
    optimizer_rx2 = optimizer if rl_settings['own_optimizer_rxfinetuning'] else None

    # --- SPG configuration (only for RL) ---
    if rl != 0:
        spg_config = resnet_rl_sinfony.StochasticPolicyGradientConfiguration(
            rx_steps=rl_settings['rx_steps'],
            tx_steps=rl_settings['tx_steps'],
            receiver_finetuning_epochs=rl_settings['number_epochs_receiver_finetuning'],
            exploration_variance_schedule=exploration_variance,
            print_iteration=rl_settings['iteration_print']
        )
    else:
        spg_config = None

    return optimizer, optimizer_tx, optimizer_rx2, spg_config
