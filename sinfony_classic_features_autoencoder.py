#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of classical digital communication in the article:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA


# LOADED PACKAGES
# Python packages
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import yaml

# Tensorflow 2 packages
import tensorflow as tf
# Keras functionality
from tensorflow.keras.models import Model
# , Conv2D, MaxPooling2D, Flatten, Add, Lambda, Concatenate, Layer, GaussianNoise
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD, Adam  # , Nadam


# Own packages
import datasets
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import my_training as mf
import my_training as mt
from my_functions import print_time, savemodule
import my_math_operations as mop


# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def classic_features_autoencoder(number_channel_uses, layer_width_transmitter, layer_width_receiver, sigma, input_shape=1, output_shape=1, number_txrx_layer=1, receiver_final_layer_linear=False, power_normalization_axis=0):
    '''Communication system as autoencoder for continuous outputs/floats source_signal like in SINFONY
    - INPUT -
    m: Layer width
    number_channel_uses: Number of channel uses
    sigma: stdev limits [sigma_min, sigma_max]
    input_shape: Input shape
    output_shape: Output shape
    power_normalization_axis: Axis for normalization at Tx
    - OUTPUT -
    model_autoencoder: Autoencoder model
    tx: Transmitter model
    rx: Receiver model
    '''
    # Transmitter design
    transmit_in = Input(shape=(input_shape))
    transmit_layer = transmit_in
    for _ in range(0, number_txrx_layer):
        transmit_layer = Dense(layer_width_transmitter, activation='relu',
                               kernel_initializer='he_uniform')(transmit_layer)  # for RELU
    transmit_layer3 = Dense(number_channel_uses,
                            activation='linear')(transmit_layer)
    transmit_out = mt.normalize_input(
        transmit_layer3, axis=power_normalization_axis)
    tx = Model(inputs=transmit_in, outputs=transmit_out)

    # Receiver Design
    receiver_in = Input(shape=(number_channel_uses, ))
    layer = receiver_in
    for _ in range(0, number_txrx_layer):
        layer = Dense(layer_width_receiver, activation='relu',
                      kernel_initializer='he_uniform')(layer)
    if receiver_final_layer_linear is True:
        layer = Dense(output_shape, activation='linear')(layer)
    receiver_out = layer
    rx = Model(inputs=receiver_in, outputs=receiver_out)

    # Model for autoencoder training
    autoencoder_in = Input(shape=(input_shape))
    channel_in = tx(autoencoder_in)
    channel_out = mt.GaussianNoise2(sigma)(channel_in)
    autoencoder_out = rx(channel_out)
    model_autoencoder = Model(inputs=autoencoder_in, outputs=autoencoder_out)

    return model_autoencoder, tx, rx


def evaluate_feature_autoencoder(models_autoencoder, source_signal, validation_data):
    '''Evaluate the feature autoencoder for source_signal and validation_data
    '''
    # Evaluate Autoencoder models
    number_distributed_agents = len(models_autoencoder)
    if number_distributed_agents >= 2:
        # More than one model: Autoencoder individually trained for each agent
        reconstructed_source = np.zeros(
            (source_signal.shape[0], number_distributed_agents, source_signal.shape[-1]), dtype=source_signal.dtype)
        for index_autoencoder in range(0, number_distributed_agents):
            reconstructed_source[:, index_autoencoder, ...] = models_autoencoder[index_autoencoder].predict(
                validation_data[:, index_autoencoder, ...])
    else:
        # One model: Feed data one-shot
        reconstructed_source = models_autoencoder[0].predict(validation_data)
    reconstructed_source = reconstructed_source.reshape(
        source_signal.shape)
    return reconstructed_source


def evaluate_feature_autoencoder_over_snr(evaluated_models_autoencoder, model_sinfony, features_validation, validation_data, test_labels, snrs=np.linspace(-30, 20, 1), validation_rounds=10):
    '''Evaluate the feature autoencoder for source_signal and validation_data over SNRs
    '''
    # SINFONY/RL-SINFONY evaluated with classic communication
    start_time = time.time()
    evaluation_measures = [[], []]
    for snr_index, snr in enumerate(snrs):
        sigma = mop.snr2standard_deviation(snr)
        sigma_test = np.array([sigma, sigma])
        # Set standard deviation weights of Noise layer in Autoencoder approach
        for model_autoencoder in evaluated_models_autoencoder:
            model_autoencoder.layers[2].set_weights([sigma_test])
        loss_i = 0
        accuracy_i = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            # Test data features enter classic communications as source_signal
            # Evaluate Autoencoder models
            reconstructed_source = evaluate_feature_autoencoder(
                evaluated_models_autoencoder, features_validation, validation_data)
            # Extract semantics based on received signal y = r_r = reconstructed_source
            number_classes = model_sinfony.layers[-1].predict(
                reconstructed_source)

            # Calculate average loss and accuracy
            loss_ii = np.mean(model_sinfony.loss(test_labels, number_classes))
            accuracy_ii = np.mean(
                np.argmax(number_classes, axis=-1) == np.argmax(test_labels, axis=-1))

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}')

        # Append list with evaluation for each SNR value
        evaluation_measures[0].append(loss_i)
        evaluation_measures[1].append(accuracy_i)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')

    accuracy = np.array(evaluation_measures[1])
    loss = np.array(evaluation_measures[0])
    return accuracy, loss


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_FILE = 'classic/config_classic_features_autoencoder.yaml'
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
    model_settings = params['model']
    evaluation_settings = params['evaluation']

    # Simulation parameters
    filename_extension = load_settings['filename_suffix']
    filename_prefix = load_settings['simulation_filename_prefix']
    save_object = savemodule(form=load_settings['save_format'])

    # Loaded dataset and SINFONY design
    # mnist, cifar10, fashion_mnist, hirise64, hirisecrater, fraeser64
    dataset = dataset_settings['dataset']
    # Show first dataset examples, just for demonstration
    show_dataset = dataset_settings['show_dataset']
    train_input, train_labels, test_input, test_labels = datasets.load_dataset(
        dataset)
    train_input = train_input[0]
    test_input = test_input[0]
    # File for distributed image classification with perfect links
    if dataset == 'mnist':
        subpath = 'mnist'
        filename = 'ResNet14_MNIST2_Ne20'
    elif dataset == 'cifar10':
        subpath = 'cifar10'
        filename = 'ResNet20_CIFAR2'
    else:
        print('Dataset not implemented into script.')
    # Path for SINFONY model
    path_sinfony = os.path.join(load_settings['path_models'], subpath)

    # Analog Autoencoder parameters
    load = load_settings['load']
    # AE, AErvec, AErvec_ind
    feature_input = model_settings['feature_input']
    filename_extension_ae_model = filename_extension    # '_ntx56_NL56_snr-4_6'
    # Path for Autoencoder models
    path_classic = load_settings['path_classic']

    # Autoencoder architecture for feature output
    number_channel_uses = model_settings['number_channel_uses']
    layer_width_tx_intermediate = model_settings['layer_width_tx_intermediate']
    layer_width_rx_intermediate = model_settings['layer_width_rx_intermediate']
    number_txrx_layer = model_settings['number_txrx_layer']
    receiver_final_layer_linear = model_settings['receiver_final_layer_linear']
    power_normalization_axis = model_settings['power_normalization_axis']

    # Training parameters
    number_epochs = training_settings['number_epochs']
    batch_size = training_settings['batch_size']
    loss = training_settings['loss']
    optimizer = training_settings['optimizer']
    learning_rate = training_settings['optimizer']
    snr_min_train = model_settings['noise']['snr_min_train']
    snr_max_train = model_settings['noise']['snr_max_train']
    dataset_size = train_input[0].shape[0]
    iterations_per_epoch = dataset_size / batch_size
    # Optimizers
    if training_settings['learning_rate_schedule']['active']:
        # Learning rate schedules
        epoch_bound = training_settings['learning_rate_schedule']['epoch_bound']
        boundaries = list(np.round(np.array(epoch_bound)
                                   * iterations_per_epoch).astype('int'))
        values = training_settings['learning_rate_schedule']['values']
        learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
        # , nesterov = True) # No advantage of Nesterov momentum with DNNs (?)
        learning_rate = learning_rate_schedule
    momentum = training_settings['momentum']
    if optimizer.lower() == 'adam':
        # Adam and its variants
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    # Evaluation parameters
    validation_rounds = evaluation_settings['validation_rounds']
    snr_range = evaluation_settings['snr_range']
    step_size = evaluation_settings['snr_step_size']

    # Evaluation script

    # Load the SINFONY model
    # Path of script being executed
    path_script = os.path.dirname(os.path.abspath(__file__))
    pathfile = os.path.join(path_script, path_sinfony, filename)
    print('Loading model ' + filename + '...')
    model_sinfony = tf.keras.models.load_model(pathfile)
    print('Model loaded.')
    if show_dataset is True:
        model_sinfony.summary()

    # Preprocess Data set
    train_input_normalized = datasets.preprocess_pixels_image(train_input)
    test_input_normalized = datasets.preprocess_pixels_image(test_input)
    if show_dataset is True:
        datasets.summarize_dataset([train_input], train_labels, [
                                   test_input], test_labels)

    features_train = model_sinfony.layers[1].predict(train_input_normalized)
    features_validation = model_sinfony.layers[1].predict(
        test_input_normalized)

    # Initialize classic and Autoencoder communications
    start_time = time.time()
    rng = np.random.default_rng()

    # Analog Autoencoder Training
    filename_model = feature_input + filename_extension_ae_model
    pathfile_model = os.path.join(path_script, path_classic, filename_model)
    if feature_input == 'AErvec':
        print('AE for all agents / feature vectors rvec:')
        validation_data = features_validation.reshape(
            [-1, features_validation.shape[-1]])
        if load is False:
            print('Start training...')
            # Prepare dataset for rvec
            input_shape = features_train.shape[-1]
            output_shape = input_shape
            data = features_train.reshape([-1, features_train.shape[-1]])
        else:
            print('Load model...')
    elif feature_input == 'AE':
        print('AE model for each entry accross all rvec entries r_i:')
        validation_data = features_validation.flatten()
        if load is False:
            print('Start training...')
            # Prepare dataset for entries in rvec
            input_shape = 1
            output_shape = 1
            data = features_train.flatten()
        else:
            print('Load model...')

    if feature_input == 'AErvec_ind':
        # Special training procedure for Autoencoder optimized for individual rvec
        print('Autoencoder for each individual agent / feature vector rvec...')
        validation_data = features_validation.reshape(
            [features_validation.shape[0], -1, features_validation.shape[-1]])
        number_distributed_agents = validation_data.shape[1]
        if load is False:
            print('Start training...')
            # Training
            start_time2 = time.time()
            models_autoencoder = []
            input_shape = features_train.shape[-1]
            output_shape = input_shape
            data = features_train.reshape(
                [features_train.shape[0], -1, features_train.shape[-1]])
            for index_autoencoder in range(0, number_distributed_agents):
                print('Start training Autoencoder' +
                      str(index_autoencoder) + '...')
                model_autoencoder, _, _ = classic_features_autoencoder(number_channel_uses, layer_width_tx_intermediate, layer_width_rx_intermediate, mop.snr2standard_deviation(np.array([snr_min_train, snr_max_train]))[
                    ::-1], input_shape=input_shape, output_shape=output_shape, number_txrx_layer=number_txrx_layer, receiver_final_layer_linear=receiver_final_layer_linear, power_normalization_axis=power_normalization_axis)
                model_autoencoder.compile(optimizer=optimizer, loss=loss)
                start_time = time.time()
                history = model_autoencoder.fit(data[:, index_autoencoder, ...], data[:, index_autoencoder, ...],
                                                epochs=number_epochs,
                                                batch_size=batch_size,
                                                validation_data=(
                    validation_data[:, index_autoencoder, ...], validation_data[:, index_autoencoder, ...]),
                    # verbose = 2,
                )
                print('Total time ' + 'Autoencoder' + str(index_autoencoder) + ': ' +
                      print_time(time.time() - start_time))
                # Save model
                filename_model = feature_input + \
                    str(index_autoencoder) + filename_extension_ae_model
                pathfile_model = os.path.join(
                    path_script, path_classic, filename_model)
                print('Saving model...')
                model_autoencoder.save(pathfile_model)
                print('Model saved.')
                models_autoencoder.append(model_autoencoder)
            print('Total time all Autoencoders: ' +
                  print_time(time.time() - start_time2))
        else:
            # Load model
            print('Load model...')
            models_autoencoder = []
            for index_autoencoder in range(0, number_distributed_agents):
                filename_model = feature_input + \
                    str(index_autoencoder) + filename_extension_ae_model
                pathfile_model = os.path.join(
                    path_script, path_classic, filename_model)
                print('Loading Autoencoder model' +
                      str(index_autoencoder) + '...')
                model_autoencoder = tf.keras.models.load_model(pathfile_model)
                print('Autoencoder model ' + str(index_autoencoder) + ' loaded.')
                if show_dataset is True:
                    model_autoencoder.summary()
                models_autoencoder.append(model_autoencoder)
    else:
        # Usual Autoencoder training script
        if load is False:
            # Training
            start_time = time.time()
            model_autoencoder, _, _ = classic_features_autoencoder(number_channel_uses, layer_width_tx_intermediate, layer_width_rx_intermediate, mop.snr2standard_deviation(np.array([snr_min_train, snr_max_train]))[
                ::-1], input_shape=input_shape, output_shape=output_shape, number_txrx_layer=number_txrx_layer, receiver_final_layer_linear=receiver_final_layer_linear, power_normalization_axis=power_normalization_axis)
            model_autoencoder.compile(optimizer=optimizer, loss=loss)
            history = model_autoencoder.fit(data, data,
                                            epochs=number_epochs,
                                            batch_size=batch_size,
                                            validation_data=(
                                                validation_data, validation_data),
                                            # verbose = 2,
                                            )
            print('Total time ' + 'Autoencoder: ' +
                  print_time(time.time() - start_time))
            # Save model
            print('Saving Autoencoder model...')
            model_autoencoder.save(pathfile_model)
            print('Autoencoder Model saved.')
        else:
            # Load model
            print('Loading Autoencoder model...')
            model_autoencoder = tf.keras.models.load_model(pathfile_model)
            print('Autoencoder Model loaded.')
            if show_dataset is True:
                model_autoencoder.summary()
        models_autoencoder = [model_autoencoder]

    # Print initialization time
    print('Initialization Time: ' + print_time(time.time() - start_time))

    # Evaluation of model
    print('Evaluate model...')
    # Evaluate model for different SNRs
    snrs = np.linspace(snr_range[0], snr_range[1], int(
        (snr_range[1] - snr_range[0]) / step_size) + 1)
    # SINFONY/RL-SINFONY evaluated with classic communication
    accuracy, loss = evaluate_feature_autoencoder_over_snr(
        models_autoencoder, model_sinfony, features_validation, validation_data, test_labels, snrs=snrs, validation_rounds=validation_rounds)

    plt.figure(1)
    plt.semilogy(snrs, 1 - accuracy)
    plt.figure(2)
    plt.semilogy(snrs, loss)

    # Save evaluation
    print('Save evaluation...')
    results = {
        "snr": snrs,
        "val_loss": loss,
        "val_acc": accuracy,
    }
    pathfile = os.path.join(path_script, path_classic, filename_prefix +
                            feature_input + '_' + filename + filename_extension)
    save_object.save(pathfile, results)
    print('Evaluation saved.')

    # Save settings when evaluation is done
    SETTINGS_SAVED_FOLDER = 'settings_saved'
    saved_settings_path = os.path.join(path_script, SETTINGS_SAVED_FOLDER)
    with open(os.path.join(saved_settings_path, filename + '.yaml'), 'w', encoding='utf8') as written_file:
        yaml.safe_dump(params, written_file, default_flow_style=False)
    print('Settings saved!')

# EOF
