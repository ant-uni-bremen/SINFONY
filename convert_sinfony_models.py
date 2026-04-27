#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

import sys
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt

# Tensorflow 2 packages
import tensorflow as tf
# import tensorflow.keras as keras
import keras
# from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam  # , Nadam


# Own packages
import datasets_keras2 as datasets
# import model_evaluation
import sinfony_architectures.resnet as resnet
import sinfony_architectures.resnet_sinfony as resnet_sinfony
import sinfony_architectures.resnet_rl_sinfony as resnet_rl_sinfony
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule
import utilities.my_training as mt
import utilities.my_training_tf1 as mt1
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import utilities.my_training as mf
import sinfony_io
import model_evaluation


# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Reinforcement Learning version RL-SINFONY via Stochastic Policy Gradient

# def resnet_cifar_tx(shape=(32, 32, 3), filters=16, num_res=2, num_resblocks=3, preactivation=True, bottleneck=False, axnorm=0, n_tx=-1, num_layer=1):
#     '''SINFONY: Function returns ResNet transmitter for CIFAR with [6 * num_res + 2] layers without bottleneck structure
#     axnorm: axis for normalization of tx output (otherwise power goes to infinity...)
#     n_tx: Tx/Rx layer length: (-1) without, (0) same length, (>0) adjust length
#     '''
#     weight_init = 'he_uniform'  # Default: he_uniform, he_normal in ResNet paper
#     weight_decay = tf.keras.regularizers.l2(0.0001)
#     # Tx
#     # Step 1 (Setup Input Layer)
#     intx = tf.keras.layers.Input(shape)
#     x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
#                                kernel_initializer=weight_init, kernel_regularizer=weight_decay)(intx)
#     if preactivation == False:
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
#     # MaxPooling Layer not in CIFAR10 version
#     # x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
#     # Step 2 (ResNet Layers)
#     for ind in range(0, num_resblocks):
#         if ind == 0:
#             x = resnet.ResnetBlock(2 ** ind * filters, num_res, first_block=True, kernel_initializer=weight_init,
#                                    kernel_regularizer=weight_decay, preactivation=preactivation, bottleneck=bottleneck)(x)
#         else:
#             x = resnet.ResnetBlock(2 ** ind * filters, num_res, kernel_initializer=weight_init,
#                                    kernel_regularizer=weight_decay, preactivation=preactivation, bottleneck=bottleneck)(x)
#     # Step 3 (Final Layers)
#     if preactivation == True:
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.GlobalAvgPool2D()(x)
#     # Channel Encoding
#     if n_tx > -1:
#         if n_tx == 0:
#             n_tx = x.shape[-1]
#         # choose number of equal layers
#         for ind in range(0, num_layer):
#             x = tf.keras.layers.Dense(n_tx, activation='relu', kernel_initializer=weight_init,
#                                       name="tx_layer" + str(ind))(x)  # Enc
#         # linear layer here, or small constant in normalization -> then no numerical instability
#         x = tf.keras.layers.Dense(n_tx, activation='linear')(x)
#     outtx = mt.normalize_input(x, axis=axnorm, eps=1e-12)
#     tx = tf.keras.Model(inputs=intx, outputs=outtx)
#     # Classification at the receiver side
#     return tx


# def resnet_cifar_multitx(shape=(32, 32, 3), filters=16, num_res=2, num_resblocks=3, preactivation=True, bottleneck=False, axnorm=0, n_tx=-1, num_layer=1, image_fac=2):
#     '''SINFONY: Function returns ResNet multiple distributed transmitters for CIFAR in one model
#     image_fac: divide image by image_fac to generate image_fac * image_fac equal pieces
#     '''
#     im_div1 = shape[0] / image_fac
#     im_div2 = shape[1] / image_fac
#     tx_list = [[], []]
#     for ind1 in range(0, image_fac):
#         for ind2 in range(0, image_fac):
#             shape_tx = (int((ind1 + 1) * im_div1) - int(ind1 * im_div1),
#                         int((ind2 + 1) * im_div2) - int(ind2 * im_div2), shape[-1])
#             tx_list[ind1].append(resnet_cifar_tx(shape=shape_tx, filters=filters, num_res=num_res, num_resblocks=num_resblocks,
#                                  preactivation=preactivation, bottleneck=bottleneck, axnorm=axnorm, n_tx=n_tx, num_layer=num_layer))

#     intx = tf.keras.layers.Input(shape)
#     outtx_list = [[], []]
#     for ind1 in range(0, image_fac):
#         for ind2 in range(0, image_fac):
#             outtx_list[ind1].append(tf.expand_dims(tf.expand_dims(tx_list[ind1][ind2](intx[:, int(
#                 ind1 * im_div1):int((ind1 + 1) * im_div1), int(ind2 * im_div2):int((ind2 + 1) * im_div2), :]), axis=1), axis=2))
#     outtx_x_list = []
#     for ind2 in range(0, image_fac):
#         outtx_x_list.append(tf.keras.layers.Concatenate(
#             axis=1)(outtx_list[:][ind2]))
#     outtx = tf.keras.layers.Concatenate(axis=2)(outtx_x_list)
#     tx = tf.keras.Model(inputs=intx, outputs=outtx)
#     return tx


# def resnet_cifar_rx(shape, classes=10, n_rx=-1, num_layer=1, rx_same=1, rx_linear=False, image_fac=2):
#     '''SINFONY: Function returns ResNet receiver
#     shape: tx.layers[-1].output_shape[1:]
#     classes: number of classes
#     n_rx: layer width
#     num_layer: number of layers
#     rx_same: [0: separate rx, 1: same rx for all tx (default), 2: one joint rx]
#     image_fac: divide image by image_fac to generate image_fac * image_fac equal pieces
#     '''
#     # Rx
#     weight_init = 'he_uniform'  # default: he_uniform, he_normal in ResNet paper
#     inrx = tf.keras.layers.Input(shape=shape)
#     # Equalization / Channel Decoding
#     rx_list = False				# False: avoid list if rx_same == 1
#     if n_rx > -1:
#         if n_rx == 0:
#             n_rx = shape[-1]
#         if rx_same == 1:
#             dec_layerlist = []
#             for indl in range(0, num_layer):
#                 # All layers the same vs. adjustable here in code
#                 dec_layerlist.append(tf.keras.layers.Dense(
#                     n_rx, activation="relu", kernel_initializer=weight_init, name="rx_layer" + str(indl)))
#             Dec = tf.keras.Sequential(dec_layerlist)
#         if rx_list == False and rx_same == 1:
#             # x = Dec(inrx) acts on each dimension
#             x = Dec(inrx)
#         elif rx_list == True and rx_same == 1 or rx_same == 0:
#             dec_list = [[], []]
#             for ind1 in range(0, image_fac):
#                 for ind2 in range(0, image_fac):
#                     # Separate decoder at Rx for each Tx
#                     if rx_same == 0:
#                         dec_layerlist = []
#                         for indl in range(0, num_layer):
#                             dec_layerlist.append(tf.keras.layers.Dense(
#                                 n_rx, activation="relu", kernel_initializer=weight_init, name="rx_layer" + str(indl)))
#                         Dec = tf.keras.Sequential(dec_layerlist)
#                     dec_list[ind1].append(tf.expand_dims(tf.expand_dims(
#                         Dec(inrx[:, ind1, ind2, :]), axis=1), axis=2))
#             dec_x_list = []
#             for ind2 in range(0, image_fac):
#                 dec_x_list.append(tf.keras.layers.Concatenate(
#                     axis=1)(dec_list[:][ind2]))
#             x = tf.keras.layers.Concatenate(axis=2)(dec_x_list)
#         elif rx_same == 2:
#             # One joint receiver for all inputs
#             x = tf.keras.layers.Flatten()(inrx)
#             dec_layerlist = []
#             for indl in range(0, num_layer):
#                 # layer are image_fac ** 2 times wider since inputs are concatenated
#                 dec_layerlist.append(tf.keras.layers.Dense(image_fac ** 2 * n_rx, activation="relu",
#                                      kernel_initializer=weight_init, name="rx_layer" + str(indl)))
#             Dec = tf.keras.Sequential(dec_layerlist)
#             x = Dec(x)
#     else:
#         x = inrx
#     if rx_linear == True:
#         # This final Rx module layer improves performance for the AE approach on rvec and at low SNR for SINFONY at the cost of a higher error floor
#         x = tf.keras.layers.Dense(n_rx, activation='linear')(x)
#     if image_fac >= 2 and rx_same != 2:
#         x = tf.keras.layers.GlobalAvgPool2D()(x)
#     outrx = tf.keras.layers.Dense(units=classes, activation='softmax')(x)
#     rx = tf.keras.Model(inputs=inrx, outputs=outrx)
#     return rx


# class resnet_cifar_rl(tf.keras.Model):
#     '''RL-SINFONY via Stochastic Policy Gradient
#     ResNet multi transmitter reinforcement learning for CIFAR with [6 * num_res + 2] layers without bottleneck structure
#     '''

#     def __init__(self, shape=(32, 32, 3), classes=10, filters=16, num_res=2, num_resblocks=3, preactivation=True, bottleneck=False, axnorm=0, n_tx=-1, n_rx=-1, rx_same=1, rx_linear=False, num_layer=1, image_fac=2):
#         super().__init__()

#         # self._training = training
#         # self._sigma = sigma
#         # self.perturbation_variance = perturbation_variance
#         # Tx
#         if image_fac >= 2:
#             self.tx = resnet_cifar_multitx(shape=shape, filters=filters, num_res=num_res, num_resblocks=num_resblocks,
#                                            preactivation=preactivation, bottleneck=bottleneck, axnorm=axnorm, n_tx=n_tx, num_layer=num_layer, image_fac=image_fac)
#             # label = 'ResNet_CIFAR10_AE_multitx'
#         else:
#             self.tx = resnet_cifar_tx(shape=shape, filters=filters, num_res=num_res, num_resblocks=num_resblocks,
#                                       preactivation=preactivation, bottleneck=bottleneck, axnorm=axnorm, n_tx=n_tx, num_layer=num_layer)
#             # label = 'ResNet_CIFAR10_AE'
#         # Rx
#         self.rx = resnet_cifar_rx(shape=self.tx.layers[-1].output_shape[1:], classes=classes,
#                                   n_rx=n_rx, num_layer=num_layer, rx_same=rx_same, rx_linear=rx_linear, image_fac=image_fac)
#         # model = Model(inputs = intx, outputs = outrx, name = label)

#     @tf.function  # (jit_compile=True)
#     def call(self, s, z, sigma, perturbation_variance=tf.constant(0.0, tf.float32)):
#         # Scaling to ensure conservation of average energy
#         x = self.tx(s) * tf.sqrt(tf.constant(1.0,
#                                              dtype=tf.float32) - perturbation_variance)
#         # x_p = GaussianNoise3(x, tf.sqrt(
#         #     [perturbation_variance, perturbation_variance]))
#         # y = GaussianNoise3(x_p, sigma)
#         x_p = mt.gaussian_noise3(x, tf.sqrt(
#             [perturbation_variance, perturbation_variance]))
#         y = mt.gaussian_noise3(x_p, sigma)
#         y = tf.stop_gradient(y)		# no gradient between Tx and Rx
#         z_hat = self.rx(y)

#         # Average BCE for each baseband symbol and each batch example
#         cce = tf.keras.losses.categorical_crossentropy(z, z_hat)
#         # The RX loss is the usual average CE
#         rx_loss = tf.reduce_mean(cce)

#         # From the TX side, the CE is seen as a feedback from the RX through which backpropagation is not possible
#         cce2 = tf.stop_gradient(cce)
#         x_p2 = tf.stop_gradient(x_p)
#         # - 0.5 * tf.math.log((2 * np.pi * perturbation_variance) ** n_dim) # Gradient is backpropagated through `x`
#         lnpxs = - tf.reduce_sum(tf.reduce_sum(tf.reduce_sum((x_p2 - x) **
#                                 2, axis=-1), axis=-1), axis=-1) / (2 * perturbation_variance)
#         tx_loss = tf.reduce_mean(lnpxs * cce2, axis=0)

#         acc = tf.reduce_mean(tf.cast(tf.math.equal(
#             tf.argmax(z_hat, axis=-1), tf.argmax(z, axis=-1)), dtype='float32'))
#         return z_hat, tx_loss, rx_loss, acc


def compare_model_weights(m1, m2, rtol=1e-6, atol=1e-8, verbose=True):
    w1 = list(m1.weights)
    w2 = list(m2.weights)

    if len(w1) != len(w2):
        raise ValueError(
            f"Different number of weights: {len(w1)} vs {len(w2)}")

    same = True
    for i, (v1, v2) in enumerate(zip(w1, w2)):
        a = np.asarray(v1.numpy())
        b = np.asarray(v2.numpy())

        name1 = getattr(v1, "path", v1.name)
        name2 = getattr(v2, "path", v2.name)

        if a.shape != b.shape:
            if verbose:
                print(
                    f"[{i}] SHAPE differs: {name1} {a.shape}  vs  {name2} {b.shape}")
            same = False
            continue

        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            if verbose:
                max_abs = float(np.max(np.abs(a - b)))
                print(f"[{i}] VALUE differs: {name1}  (max abs diff={max_abs})")
                if name1 != name2:
                    print(f"     other name: {name2}")
            same = False

    return same


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_FILE = 'mnist/semantic_config_mnist_rlsinfony_load.yaml'
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
    # keras.backend.clear_session()          	                    # Clearing graphs
    # keras.backend.set_floatx(load_settings['numerical_precision'])
    # # Random seed in every run, predictable random numbers for debugging with np.random.seed(0)
    # np.random.seed()

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
    valX = mt1.create_batch(test_input_normalized, validation_dataset_size, 0)
    # Summarize loaded dataset
    if show_dataset is True:
        datasets.summarize_dataset(
            train_input, train_labels, test_input, test_labels)

    # Create/load model
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        number_resnet_blocks, number_residual_units, model_settings['resnet']['bottleneck'])
    print('ResNet', resnet_layer_number, ' chosen')

    # if load is False:
    # Check whether multi-image input and choose model accordingly
    if len(image_shapes) == 1:
        # image_shapes = image_shapes[0]
        # number_filters = number_filters[0]
        # Create new model:
        if transceiver_split == 1:
            # Select SINFONY or RL-SINFONY
            if rl == 1:
                # RL-SINFONY
                model = resnet_rl_sinfony.ResnetRLSinfony(
                    resnet_config=resnet_config, communication_config=communication_config)
                # model2 = resnet_cifar_rl(shape=resnet_config.image_shape, classes=resnet_config.number_classes, filters=resnet_config.number_filters, num_res=resnet_config.number_residual_units[0], num_resblocks=resnet_config.number_resnet_blocks, preactivation=resnet_config.preactivation,
                #                          bottleneck=resnet_config.bottleneck, axnorm=communication_config.encoding_config.normalization_axis, n_tx=communication_config.encoding_config.encoding_layer_width, n_rx=communication_config.decoding_config.decoding_layer_width, rx_same=communication_config.decoding_config.rx_joint_layers, rx_linear=communication_config.decoding_config.rx_final_layer_linear, num_layer=communication_config.decoding_config.number_decoding_layer, image_fac=communication_config.encoding_config.image_split_factor)
            elif rl == 2:
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
            if rl == 1:
                print('Not implemented yet.')
            else:
                model, tx, rx = resnet_sinfony.resnet_sinfony(
                    resnet_config=resnet_config, communication_config=communication_config)
        else:
            model = resnet_sinfony.resnet_multi_image(resnet_config=resnet_config, number_combination_layer=model_settings[
                'resnet']['multi_image_layer_number'], combination_layer_width=model_settings['resnet']['multi_image_layer_width'])

    # Convert models to new Tensorflow version
    # Load existing model and extract weights:
    # MNIST: ResNet14_MNIST4_Ne20_snr-4_6, ResNet14_MNIST6_Ne20_snr-4_6
    # CIFAR10: ResNet20_CIFAR4_snr-4_6, ResNet20_CIFAR6_snr-4_6
    save_only_weights = True
    load_from_weights = True
    # filename2 = 'ResNet14_MNIST4_Ne20_snr-4_6'

    # model.summary()
    pathfile_model2_results = os.path.join(path_script, subpath_results,
                                           load_settings['simulation_filename_prefix'] + filename)
    layer_found = sinfony_io.find_layer_by_name(model, 'rx_layer0')
    layer_found.get_weights()

    # Set weights to that of old model
    if load_from_weights is True:
        print('Loading model from weights...')
        pathfile_weights = os.path.join(
            path_script, subpath_results, filename + '_weights.h5')
        # pathfile_weights2 = os.path.join(
        #     path_script, subpath_results, filename, filename)
        model(test_input, test_labels, np.array(
            [0.0, 0.0], dtype='float32'), np.array(0.0, dtype='float32'))
        model.load_weights(pathfile_weights, skip_mismatch=False)
        # model2.load_weights(pathfile_weights2, skip_mismatch=False)
        # model = model2
        print('Weights loaded.')
        layer_found = sinfony_io.find_layer_by_name(model, 'rx_layer0')
        layer_found.get_weights()
    else:
        print('Loading model...')
        # Load SINFONY model
        pathfile_model2 = os.path.join(path_script, subpath_results, filename)
        model2 = keras.models.load_model(pathfile_model2)
        # model2 = tf.keras.layers.TFSMLayer(
        #     pathfile_model2, call_endpoint="serving_default")
        print('Model loaded.')
        model2.summary()
        model = model2
        # model.set_weights(model2.get_weights())
        # model.save_weights(pathfile_model2 + '.h5')
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: Unify with model.save before
    # Save old Keras model for use in new Tensorflow version:
    # Note: Requires sionna conda env
    if load is False:
        print('Saving model...')
        pathfile_weights = os.path.join(
            path_script, subpath_results, 'weights_' + filename + '.weights.h5')
        if save_only_weights is True:
            model.save_weights(pathfile_weights)
        else:
            model.save(pathfile + '.keras')
        print('Model saved.')
    else:
        # Compile and train model
        # Load training history to include evaluation
        print('Load training history...')
        results = saveobj.load(pathfile_model2_results)
        if results is None:
            results = {}
        else:
            results = dict(results)
        print('Loaded!')

    # Evaluation of model
    snrs = mop.snr_range2snrlist(snr_range, snr_step_size)
    if evaluation_mode == 0:
        print('Evaluate model...')
        print(filename)
        # Evaluate model for different SNRs
        if transceiver_split == 1:
            # SINFONY/RL-SINFONY
            if rl >= 1:
                accuracy, loss = model_evaluation.evaluate_rlsinfony(model, test_input, test_labels,
                                                                     snrs=snrs, validation_rounds=validation_rounds)
            else:
                accuracy, loss = model_evaluation.evaluate_sinfony(model, test_input, test_labels,
                                                                   snrs=snrs, validation_rounds=validation_rounds)
        else:
            # Standard image recognition: Evaluate model accuracy once for test data
            if rl >= 1:
                _, _, loss_i, accuracy_i = model(
                    test_input, test_labels, sigma=tf.convert_to_tensor([0, 0], dtype='float32'))
            else:
                accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                    model, test_input, test_labels)
            # Independent from SNR / constant, but plotted over SNR range
            loss = np.array(loss_i) * np.ones(snrs.shape)
            accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
            print(f'Validation Accuracy: {accuracy_i * 100.0:.3f}%')
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
