#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

# Tensorflow 2 packages
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam  # , Nadam

# Own packages
import datasets
import model_evaluation
import resnet
import resnet_sinfony
import resnet_rl_sinfony
import my_math_operations as mop
from my_functions import savemodule
import my_training as mt
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import my_training as mf

# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_FILE = 'cifar10/semantic_config_cifar_sinfony.yaml'
    # Avoid error messages
    # import logging
    # tf.get_logger().setLevel(logging.ERROR)
    # Load the provided configuration file or the default one
    # python SINFONY.py semantic_config.yaml
    # Workaround for interactive sessions: Only allow config file names starting 'semantic_config'
    SETTINGS_FILE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][0:15].lower(
    ) == 'semantic_config' else SETTINGS_FILE
    # Change to 'settings_saved' to reload simulations settings
    SETTINGS_FOLDER = 'settings'
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)
    with open(settings_path, 'r', encoding='UTF8') as file:
        params = yaml.safe_load(file)
    load_settings = params['load_settings']
    dataset_settings = params['dataset']
    training_settings = params['training']
    rl_settings = params['reinforcement_learning']
    model_settings = params['model']
    evaluation_settings = params['evaluation']

    # Initialization
    mt.gpu_select(number=load_settings['gpu'], memory_growth=False)
    tf.keras.backend.clear_session()          	                    # Clearing graphs
    tf.keras.backend.set_floatx(load_settings['numerical_precision'])
    # Random seed in every run, predictable random numbers for debugging with np.random.seed(0)
    np.random.seed()

    # Simulation
    # Load model and reevaluate: False (default) # params.get('load', False)
    load = load_settings['load']
    # Evaluation mode: (0) default: Validation for SNR range, (1) Saving probability data for interface to application, (2) t-SNE embedding for visualization
    evaluation_mode = evaluation_settings['mode']
    filename = load_settings['filename']
    # Sub path for saved data
    subpath_results = load_settings['path']
    # Path of script being executed
    pathfile = os.path.join(path_script, subpath_results, filename)
    pathfile2 = os.path.join(path_script, subpath_results,
                             load_settings['simulation_filename_prefix'] + filename)
    saveobj = savemodule(form=load_settings['save_format'])

    # Data set
    # mnist, cifar10, fashion_mnist, hirise64, hirisecrater, fraeser64
    dataset = dataset_settings['dataset']
    # Show first dataset examples, just for demonstration
    show_dataset = dataset_settings['show_dataset']
    train_input, train_labels, test_input, test_labels = datasets.load_dataset(
        dataset, validation_split=dataset_settings['validation_split'], image_split=dataset_settings['image_split'])

    # Training
    # Batch size, SGD: 128/64, Adam: 500
    batch_size = training_settings['batch_size']
    # sgd, adam, (sgdlrs) SGD with learning rate schedule
    optimizer = training_settings['optimizer']
    # Learning rate, SGD/Adam: 1e-3, RL: 1e-4
    learning_rate = training_settings['learning_rate']
    dataset_size = train_input[0].shape[0]
    iterations_per_epoch = dataset_size / batch_size
    # Number of epochs, 200 in CIFAR original implementation, 20 for MNIST
    number_epochs = training_settings['number_epochs']
    # Choose validation data set size: None/0, 100, 1000, test_input[0].shape[0]
    validation_dataset_size = training_settings['validation_dataset_size']
    if validation_dataset_size == 'full':
        validation_dataset_size = test_input[0].shape[0]

    # RL training
    # (0) default AE, (1) Reinforcement learning training, (2) AE trained with rl-based training implementation
    rl = rl_settings['active']
    # [0.15, 0.15 ** 2] # with higher exploration variance, the gradient estimator variance decreases at the cost of more bias...
    exploration_values = rl_settings['exploration_variance']
    # [2000] # only activated during tx_train
    per_epoch_bound = rl_settings['exploration_boundaries']
    exploration_boundaries = list(
        np.round(np.array(per_epoch_bound) / 2 * iterations_per_epoch).astype('int'))
    exploration_variance = resnet_rl_sinfony.PertubationVarianceSchedule(
        exploration_values, exploration_boundaries)

    # NN Com system design
    # (0) only image recognition, (1) with (multi) com. system inbetween
    transceiver_split = model_settings['communication']['transceiver_split']
    # Number of Tx/Rx layers: 1 (default)
    number_layer = model_settings['communication']['number_txrx_layer']
    if transceiver_split == 1:
        # Training w/o noise
        if model_settings['noise']['active'] is False:
            # training without noise
            sigma_train = np.array([0, 0])
        else:
            # training with noise
            sigma_train = mop.snr2standard_deviation(
                np.array([model_settings['noise']['snr_min_train'], model_settings['noise']['snr_max_train']]))[::-1]
    total_number_iterations = number_epochs * iterations_per_epoch
    weight_decay = tf.keras.regularizers.l2(
        model_settings['resnet']['weight_decay'])
    weight_decay_communication = tf.keras.regularizers.l2(
        model_settings['communication']['weight_decay'])
    if transceiver_split == 1:
        encoding_config = resnet_sinfony.EncodingConfiguration(transmit_normalization=True, normalization_axis=model_settings['communication']['power_normalization_axis'], encoding_layer_width=model_settings['communication'][
            'number_channel_uses'], number_encoding_layer=number_layer, image_split_factor=model_settings['communication']['image_split_factor'], weight_initialization=model_settings['communication']['weight_initialization'], weight_decay=weight_decay_communication)
        decoding_config = resnet_sinfony.DecodingConfiguration(decoding_layer_width=model_settings['communication']['rx_layer_width'], number_decoding_layer=number_layer, rx_joint_layers=model_settings['communication'][
            'rx_same'], rx_final_layer_linear=model_settings['communication']['rx_linear'], weight_initialization=model_settings['communication']['weight_initialization'], weight_decay=weight_decay_communication)
        communication_channel = resnet_sinfony.CommunicationChannel(
            sigma_train)
        communication_config = resnet_sinfony.CommunicationConfiguration(
            encoding_config=encoding_config, decoding_config=decoding_config, communication_channel=communication_channel)

    # Some training functionality of model.fit()
    if rl == 0:
        VERBOSE = 'auto'
        # - Callbacks for early stopping and model checkpoints -
        # EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
        early_stopping = []
        # tf.keras.callbacks.ModelCheckpoint(pathfile, monitor = 'val_loss', verbose = VERBOSE, save_best_only = False, mode = 'auto', period = 1, save_weights_only = False, save_freq = 'epoch')
        model_checkpoint = []
        # Track training loss and accuracy of each batch iteration
        batch_tracking = mt.BatchTrackingCallback()

    # Optimizer
    if training_settings['learning_rate_schedule']['active']:
        # Learning rate schedules
        # Original ResNet: 1/2, 3/4 of training learning rate division by 10, in total 64k iterations
        # at 32000, 48000 iterations of 64000 in total: [100, 150] for CIFAR / [3, 6] for MNIST / [2, 50] for hirise / [100] for RL CIFAR
        epoch_bound = training_settings['learning_rate_schedule']['epoch_bound']
        boundaries = list(np.round(np.array(epoch_bound)
                                   * iterations_per_epoch).astype('int'))
        # [1e-1, 1e-2, 1e-3] for ae training / [0.001, 0.0001, 0.00001] for adam / [1e-3, 1e-4, 1e-5] for rl training / [1e-3, 1e-4] for rl CIFAR training sgdlr2
        values = training_settings['learning_rate_schedule']['values']
        learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
        # , nesterov = True) # No advantage of Nesterov momentum with DNNs (?)
        learning_rate = learning_rate_schedule
    momentum = training_settings['momentum']
    if optimizer.lower() == 'adam':
        # Adam and its variants
        # Optimizer for rx training # Nadam() # yogi() # Generalization/validation performance expected to be bad
        optimizer = Adam(learning_rate=learning_rate)
        if rl != 0:
            # Optimizer for tx training
            optimizer_tx = Adam(learning_rate=learning_rate)
    else:
        # Default: Stochastic Gradient Descent with momentum 0.9 as in ResNet paper
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        if rl != 0:
            optimizer_tx = SGD(learning_rate=learning_rate, momentum=momentum)
    # Optimizer for rx finetuning: None (default, i.e., optimizer)
    if rl_settings['own_optimizer_rxfinetuning']:
        optimizer_rx2 = optimizer
    else:
        optimizer_rx2 = None
    if rl != 0:
        spg_config = resnet_rl_sinfony.StochasticPolicyGradientConfiguration(rx_steps=rl_settings['rx_steps'], tx_steps=rl_settings['tx_steps'], receiver_finetuning_epochs=rl_settings[
            'number_epochs_receiver_finetuning'], exploration_variance_schedule=exploration_variance, print_iteration=rl_settings['iteration_print'])

    # ResNet20 model
    number_classes = train_labels.shape[1]			# 10 classes for CIFAR10, MNIST
    # Defines ResNet layer number, 3 for smallest ResNet20 for CIFAR10 (2 for MNIST)
    number_residual_units = model_settings['resnet']['number_residual_units']
    if dataset.lower() == 'cifar10' and number_residual_units <= 2:
        print(
            'Warning: Number of residual units is below minimum number for CIFAR10 dataset!')
    # 3 for CIFAR10, MNIST
    number_resnet_blocks = model_settings['resnet']['number_resnet_blocks']
    image_shapes = []
    for image_dataset in test_input:
        image_shapes.append(image_dataset.shape[1:])
    number_filters = model_settings['resnet']['number_filters']
    if len(test_input) == 1 and isinstance(number_filters, list):
        # If only one image, then use first number of filters entry
        number_filters = number_filters[0]
    # Convert number_residual_units to a list whose length matches number_resnet_blocks if not already
    # NOTE: Then, the first element is repeated!
    number_residual_units = resnet.repeat_entry_2_list(
        number_residual_units, number_resnet_blocks)
    resnet_config = resnet.ResnetConfiguration(architecture=model_settings['resnet']['architecture'],
                                               image_shape=image_shapes,
                                               number_classes=number_classes,
                                               number_filters=number_filters,
                                               number_residual_units=number_residual_units,
                                               number_resnet_blocks=number_resnet_blocks,
                                               preactivation=model_settings['resnet']['preactivation'],
                                               bottleneck=model_settings['resnet']['bottleneck'],
                                               batch_normalization=model_settings['resnet']['batch_normalization'],
                                               weight_initialization=model_settings[
                                                   'resnet']['weight_initialization'],
                                               weight_decay=weight_decay)

    # Evaluation/Validation
    # SNR in dB range: [-30, 20] (default)
    snr_range = evaluation_settings['snr_range']
    # SNR in dB steps: 1 (default)
    snr_step_size = evaluation_settings['snr_step_size']
    # Rounds through validation data with different noise realizations
    validation_rounds = evaluation_settings['validation_rounds']
    # SNR value for interface data / T-SNE embedding: -10 / 20
    snr_evaluation = evaluation_settings['evaluation_snr']

    # TRAINING AND EVALUATION SCRIPT

    # Preprocessing
    train_input_normalized, test_input_normalized = datasets.preprocess_pixels(
        train_input, test_input)
    # If computational heavy (RL approach), use subset of validation set
    valY = test_labels[:validation_dataset_size, ...]
    valX = mt.create_batch(test_input_normalized, validation_dataset_size, 0)
    # Summarize loaded dataset
    if show_dataset is True:
        datasets.summarize_dataset(
            train_input, train_labels, test_input, test_labels)

    # Create/load model
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        number_resnet_blocks, number_residual_units, model_settings['resnet']['bottleneck'])
    print('ResNet', resnet_layer_number, ' chosen')

    if load is False:
        # Check whether multi-image input and choose model accordingly
        if len(image_shapes) == 1:
            # Create new model:
            if transceiver_split == 1:
                # SINFONY
                model, tx, rx = resnet_sinfony.resnet_sinfony_imagesplit(
                    resnet_config=resnet_config, communication_config=communication_config)
            else:
                # Standard image recognition based on total image
                model = resnet.resnet(resnet_config=resnet_config)
        else:
            # Multiple images models
            if transceiver_split == 1:
                model, tx, rx = resnet_sinfony.resnet_sinfony(
                    resnet_config=resnet_config, communication_config=communication_config)

    # Convert models to new Tensorflow version
    # Load existing model and extract weights:
    # MNIST: ResNet14_MNIST4_Ne20_snr-4_6, ResNet14_MNIST6_Ne20_snr-4_6
    # CIFAR10: ResNet20_CIFAR4_snr-4_6, ResNet20_CIFAR6_snr-4_6
    filename2 = 'ResNet20_CIFAR4_snr-4_6'
    print('Loading model...')
    # Load SINFONY model
    model2 = tf.keras.models.load_model(
        os.path.join(path_script, subpath_results, filename2))
    print('Model loaded.')

    model.summary()
    model2.summary()

    # Set weights to that of old model
    model.set_weights(model2.get_weights())
    pathfile_model2 = os.path.join(path_script, subpath_results,
                                   load_settings['simulation_filename_prefix'] + filename2)
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Save old Keras model for use in new Tensorflow version:
    # Note: Requires sionna conda env
    print('Saving model...')
    model.save(pathfile)
    print('Model saved.')

    # Compile and train model
    # Load training history to include evaluation
    print('Load training history...')
    results = saveobj.load(pathfile_model2)
    if results is None:
        results = {}
    else:
        results = dict(results)
    print('Loaded!')

    # Evaluation of model
    snrs = mop.snr_range2snrlist(snr_range, snr_step_size)

    print('Evaluate model...')
    print(filename)
    # Evaluate model for different SNRs
    if transceiver_split == 1:
        # SINFONY
        accuracy, loss = model_evaluation.evaluate_sinfony(model, test_input_normalized, test_labels,
                                                           snrs=snrs, validation_rounds=validation_rounds)
    # Show performance curve
    plt.figure(1)
    plt.semilogy(snrs, 1 - accuracy)
    plt.xlabel('SNR')
    plt.ylabel(
        'semantic performance measure: classification error rate')
    plt.figure(2)
    plt.semilogy(snrs, loss)
    plt.xlabel('SNR')
    plt.ylabel('crossentropy loss')

    # Save evaluation
    print('Save evaluation...')
    results['snr'] = snrs
    results['val_loss'] = loss
    results['val_acc'] = accuracy
    saveobj.save(pathfile2, results)
    print('Evaluation saved.')

# EOF
