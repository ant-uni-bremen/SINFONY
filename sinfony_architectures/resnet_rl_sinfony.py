#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 14:58:13 2024

@author: beck
RL-SINFONY architecture build from resnet

Belongs to simulation framework for numerical results of the article:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

# LOADED PACKAGES
import os
import time
import numpy as np

# Tensorflow 2 packages
import keras

import sinfony_architectures.resnet as resnet
import sinfony_architectures.resnet_sinfony as rs
from utilities.my_functions import print_time
import utilities.my_layers_keras3 as mt
from utilities.my_dataset_handling import shuffle_dataset, get_batch_dataset


# Reinforcement Learning version RL-SINFONY via Stochastic Policy Gradient
# TODO: Replace gaussian_noise3 function by model

class ResnetRLSinfony(keras.Model):
    '''RL-SINFONY via Stochastic Policy Gradient
    ResNet multi transmitter reinforcement learning for CIFAR with [6 * number_residual_units + 2] layers without bottleneck structure
    '''

    def __init__(self, communication_config, resnet_config=resnet.ResnetConfiguration()):
        super().__init__()
        self.image_split_factor = communication_config.encoding_config.image_split_factor
        if self.image_split_factor >= 2:
            self.transmitter = rs.resnet_multi_transmitter_imagesplit(
                resnet_config=resnet_config, encoding_config=communication_config.encoding_config)
        else:
            self.transmitter = rs.resnet_transmitter(
                resnet_config=resnet_config, encoding_config=communication_config.encoding_config)
        self.receiver = rs.resnet_receiver_imagesplit(
            received_signal_shape=(communication_config.encoding_config.image_split_factor,
                                   communication_config.encoding_config.image_split_factor, communication_config.encoding_config.encoding_layer_width),
            number_classes=resnet_config.number_classes,
            image_split_factor=self.image_split_factor,
            decoding_config=communication_config.decoding_config)

    def call(self, observation, true_labels, sigma, perturbation_variance=None):
        '''Compute model outputs and loss/accuracy
        '''
        if perturbation_variance is None:
            perturbation_variance = keras.ops.zeros(())  # scalar 0.0

        # Scaling to ensure conservation of average energy
        transmit_signal = self.transmitter(observation) * \
            keras.ops.sqrt(1 - perturbation_variance)

        exploration_signal = mt.gaussian_noise3(transmit_signal, keras.ops.sqrt(
            keras.ops.stack([perturbation_variance, perturbation_variance])))
        received_signal = mt.gaussian_noise3(exploration_signal, sigma)
        received_signal = keras.ops.stop_gradient(received_signal)

        estimated_labels = self.receiver(received_signal)

        # Average BCE for each baseband symbol and each batch example
        cross_entropy_empirical = keras.losses.categorical_crossentropy(
            true_labels, estimated_labels)
        # The RX loss is the usual average CE
        receiver_loss = keras.ops.mean(cross_entropy_empirical)

        # From the TX side, the CE is seen as a feedback from the RX through which backpropagation is not possible
        cross_entropy_empirical2 = keras.ops.stop_gradient(
            cross_entropy_empirical)
        exploration_signal2 = keras.ops.stop_gradient(exploration_signal)

        # Unified reduce_sum over all non-batch dims (backend-agnostic)
        lnpxs = - keras.ops.sum(
            keras.ops.reshape((exploration_signal2 - transmit_signal) ** 2,
                              [keras.ops.shape(transmit_signal)[0], -1]),
            axis=-1
        ) / (2 * perturbation_variance)

        transmitter_loss = keras.ops.mean(
            lnpxs * cross_entropy_empirical2, axis=0)

        accuracy = keras.ops.mean(keras.ops.cast(
            keras.ops.equal(
                keras.ops.argmax(estimated_labels, axis=-1),
                keras.ops.argmax(true_labels, axis=-1)),
            dtype='float32'))

        return estimated_labels, transmitter_loss, receiver_loss, accuracy


class ResnetAE2(ResnetRLSinfony):
    '''SINFONY trained via RL-SINFONY training procedure/function
    ResNet multi transmitter autoencoder-like defined like in reinforcement learning version for CIFAR with [6 * number_residual_units + 2] layers without bottleneck structure
    '''

    def call(self, observation, true_labels, sigma, perturbation_variance=None):
        '''Compute model outputs and loss/accuracy
        perturbation_variance not used in AE approach, but placeholder to enable integration into RL-based training function
        '''
        transmit_signal = self.transmitter(observation)
        received_signal = mt.gaussian_noise3(transmit_signal, sigma)
        estimated_labels = self.receiver(received_signal)

        # Average BCE loss for each baseband symbol and each batch example
        cross_entropy_empirical = keras.losses.categorical_crossentropy(
            true_labels, estimated_labels)
        # The RX loss is the usual average CE
        rx_loss = keras.ops.mean(cross_entropy_empirical)
        # The Tx loss is the same for AE
        tx_loss = rx_loss

        # Compute classification accuracy
        accuracy = keras.ops.mean(keras.ops.cast(keras.ops.equal(
            keras.ops.argmax(estimated_labels, axis=-1),
            keras.ops.argmax(true_labels, axis=-1)), dtype='float32'))

        return estimated_labels, tx_loss, rx_loss, accuracy


class PertubationVarianceSchedule():
    '''Exploration/pertubation variance schedule: piecewise constant
    values: Exploration/pertubation variance values
    boundaries: Iteration after which new value is adopted
    '''

    def __init__(self, values=[0.15], boundaries=[]):
        self._iteration = 0
        self._iteration_boundary = 0
        self.boundaries = boundaries
        self.values = values

    def __call__(self):
        if self._iteration_boundary != (len(self.values) - 1):
            if self._iteration >= self.boundaries[self._iteration_boundary]:
                self._iteration_boundary = self._iteration_boundary + 1
        pertubation_variance = keras.ops.convert_to_tensor(
            self.values[self._iteration_boundary], 'float32')
        self._iteration = self._iteration + 1
        return pertubation_variance


class StochasticPolicyGradientConfiguration():
    '''Stochastic Policy Gradient configuration class
    '''

    def __init__(self, rx_steps=10, tx_steps=10, receiver_finetuning_epochs=200, exploration_variance_schedule=PertubationVarianceSchedule(), print_iteration=1):
        self.rx_steps = rx_steps
        self.tx_steps = tx_steps
        self.receiver_finetuning_epochs = receiver_finetuning_epochs
        self.exploration_variance_schedule = exploration_variance_schedule
        self.print_iteration = print_iteration
        # self.optimizer = optimizer
        # self.optimizer_transmitter = optimizer_transmitter
        # self.optimizer_receiver_finetuning = optimizer_receiver_finetuning


def create_training_status_string(epoch, total_epochs, batch, total_batches, rx_loss, accuracy, acc_val=None, rx_val_loss=None, tx_loss=None, tx_val_loss=None):
    """
    Helper function to create training status strings for consistent formatting

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        batch: Current batch number
        total_batches: Total number of batches
        rx_loss: Receiver loss value
        accuracy: Accuracy value
        acc_val: Validation accuracy (optional)
        rx_val_loss: Validation receiver loss (optional)
        tx_loss: Transmitter loss value (needed for Tx training)
        tx_val_loss: Validation transmitter loss (optional)

    Returns:
        Formatted string representing training status
    """
    epoch_label = f'Epoch: {epoch + 1}/{total_epochs}'

    if tx_val_loss is None:
        tx_val_loss = np.NaN
    else:
        tx_val_loss = keras.ops.convert_to_numpy(tx_val_loss)
    if rx_val_loss is None:
        rx_val_loss = np.NaN
    else:
        rx_val_loss = keras.ops.convert_to_numpy(rx_val_loss)
    if acc_val is None:
        acc_val = np.NaN
    else:
        acc_val = keras.ops.convert_to_numpy(acc_val)

    if tx_loss is None:
        tx_extension = ''
        transceiver_label = '[Rx]'
    else:
        tx_extension = f', PG: {keras.ops.convert_to_numpy(tx_loss):.4f}/{tx_val_loss:.4f}'
        transceiver_label = '[Tx]'

    training_status = f'{transceiver_label} {epoch_label}, Batch: {batch + 1}/{total_batches}, CE: {keras.ops.convert_to_numpy(rx_loss):.4f}/{rx_val_loss:.4f}, Acc: {keras.ops.convert_to_numpy(accuracy):.2f}/{acc_val:.2f}' + tx_extension

    return training_status


BACKEND = os.environ.get('KERAS_BACKEND', 'tensorflow')

if BACKEND == 'tensorflow':
    import tensorflow as tf

    @tf.function
    def train_tx(model, opt_tx, train_input, train_labels, sigma, exploration_variance_schedule=keras.ops.convert_to_tensor(0.0, 'float32')):
        '''Function that implements one transmitter training iteration using RL
        '''
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the TX loss
            _, tx_loss, rx_loss, accuracy = model(
                train_input, train_labels, sigma, exploration_variance_schedule)
        # Computing and applying gradients
        weights = model.transmitter.trainable_weights
        gradients = tape.gradient(tx_loss, weights)
        opt_tx.apply_gradients(zip(gradients, weights))
        return rx_loss, accuracy, tx_loss

    @tf.function
    def train_rx(model, opt_rx, train_input, train_labels, sigma):
        '''Function that implements one receiver training iteration
        Reused for receiver finetuning
        '''
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the RX loss
            # No perturbation is added
            _, _, rx_loss, accuracy = model(train_input, train_labels, sigma)
        # Computing and applying gradients
        weights = model.receiver.trainable_weights
        gradients = tape.gradient(rx_loss, weights)
        opt_rx.apply_gradients(zip(gradients, weights))
        # The RX loss is returned to print the progress
        return rx_loss, accuracy

elif BACKEND == 'torch':
    # Not optimized for pytorch, but compatible
    import torch

    def prepare_model_for_pytorch_training(model):
        for w in model.trainable_weights:
            w.value.requires_grad_(True)

    def train_tx(model, opt_tx, train_input, train_labels, sigma, exploration_variance_schedule=keras.ops.convert_to_tensor(0.0, 'float32')):
        # Zero gradients on all trainable weights
        for w in model.transmitter.trainable_weights:
            w.value.grad = None

        _, tx_loss, rx_loss, accuracy = model(
            train_input, train_labels, sigma, exploration_variance_schedule)

        tx_loss.backward()

        # Apply gradients via Keras optimizer
        grads = [w.value.grad for w in model.transmitter.trainable_weights]
        opt_tx.apply(grads, model.transmitter.trainable_weights)
        return rx_loss, accuracy, tx_loss

    def train_rx(model, opt_rx, train_input, train_labels, sigma):
        for w in model.receiver.trainable_weights:
            w.value.grad = None

        _, _, rx_loss, accuracy = model(train_input, train_labels, sigma)

        rx_loss.backward()

        grads = [w.value.grad for w in model.receiver.trainable_weights]
        opt_rx.apply(grads, model.receiver.trainable_weights)

        return rx_loss, accuracy


def rl_based_training(model, train_input, train_labels, opt, opt_tx=None, opt_rx2=None, validation_input=None, validation_labels=None, epochs=10, training_batch_size=64, sigma=np.array([0, 0]), stochastic_policy_gradient_config=StochasticPolicyGradientConfiguration(), zero_epoch=False):
    '''Reinforcement-based training of the semantic communication system model
    model: model with parameters to be trained
    train_input: training data set input
    train_labels: training data set output
    opt: Receiver optimizer
    opt_tx: Transmitter optimizer
    opt_rx2: Receiver finetuning optimizer
    validation_input: Validation data set input
    validation_labels: Validation data set output
    epochs: Number of training epochs
    receiver_finetuning_epochs: Number of epochs for receiver finetuning
    tx_steps: iterations of Tx training
    rx_steps: iterations of Rx training
    training_batch_size: Training batch size
    sigma: AWGN standard deviation
    exploration_variance_schedule: stochastic policy / RL-exploration variance
    print_iteration: Printer after print_iteration iterations
    zero_epoch: Evaluation on training and validation data before first training epoch
    '''
    tx_steps = stochastic_policy_gradient_config.tx_steps
    rx_steps = stochastic_policy_gradient_config.rx_steps
    receiver_finetuning_epochs = stochastic_policy_gradient_config.receiver_finetuning_epochs
    exploration_variance_schedule = stochastic_policy_gradient_config.exploration_variance_schedule
    print_iteration = stochastic_policy_gradient_config.print_iteration
    # Optimizers used to apply gradients
    optimizer_rx = opt 					# For training the receiver
    if opt_tx is None:
        optimizer_tx = opt 				# For training the transmitter
    else:
        optimizer_tx = opt_tx
    if opt_rx2 is None:
        optimizer_rx2 = opt
    else:
        optimizer_rx2 = opt_rx2 		# For receiver finetuning
    total_steps = tx_steps + rx_steps

    # Save performance measures / results of training in dictionary
    perf_meas = {
        'rx_loss': [],
        'acc': [],
        'tx_loss': [],
        'rx_val_loss': [],
        'acc_val': [],
        'tx_val_loss': [],
    }

    # Optional initial training and validation dataset evaluation of model before first
    # training iteration:
    # Note: Not consistent with model.fit() output of SINFONY training history.
    if zero_epoch is True:
        pv = exploration_variance_schedule()
        _, tx_loss, rx_loss, accuracy = model(
            train_input, train_labels, sigma, pv)
        if (validation_input is not None) and (validation_labels is not None):
            _, tx_val_loss, rx_val_loss, acc_val = model(
                validation_input, validation_labels, sigma, pv)
            perf_meas['rx_val_loss'].append(keras.ops.convert_to_numpy(rx_val_loss))
            perf_meas['acc_val'].append(keras.ops.convert_to_numpy(acc_val))
            perf_meas['tx_val_loss'].append(keras.ops.convert_to_numpy(tx_val_loss))
        perf_meas['tx_loss'].append(keras.ops.convert_to_numpy(tx_loss))
        perf_meas['rx_loss'].append(keras.ops.convert_to_numpy(rx_loss))
        perf_meas['acc'].append(keras.ops.convert_to_numpy(accuracy))

    # Training loop
    if BACKEND == 'torch':
        prepare_model_for_pytorch_training(model)
    start_time0 = time.time()
    start_time = time.time()
    number_batches = len(train_input[0]) // training_batch_size
    for index_epoch in range(epochs):
        # Receiver training is performed first to keep it ahead of the transmitter
        # as it is used for computing the losses when training the transmitter
        batch_count = 0
        train_input, train_labels = shuffle_dataset(
            train_input, train_labels)
        for batch_input, batch_labels in get_batch_dataset(train_input, train_labels, training_batch_size):
            if batch_count % total_steps >= rx_steps:
                # One step of transmitter training
                pv = exploration_variance_schedule()
                rx_loss, accuracy, tx_loss = train_tx(
                    model, optimizer_tx, batch_input, batch_labels, sigma, exploration_variance_schedule=pv)
                if (validation_input is None) or (validation_labels is None):
                    training_status_string = create_training_status_string(
                        epoch=index_epoch, total_epochs=epochs, batch=batch_count, total_batches=number_batches,
                        rx_loss=rx_loss, accuracy=accuracy, tx_loss=tx_loss)
                else:
                    _, tx_val_loss, rx_val_loss, acc_val = model(
                        validation_input, validation_labels, sigma, pv)
                    training_status_string = create_training_status_string(
                        epoch=index_epoch, total_epochs=epochs, batch=batch_count, total_batches=number_batches,
                        rx_loss=rx_loss, accuracy=accuracy, acc_val=acc_val, rx_val_loss=rx_val_loss, tx_loss=tx_loss, tx_val_loss=tx_val_loss)
                    perf_meas['rx_val_loss'].append(keras.ops.convert_to_numpy(rx_val_loss))
                    perf_meas['acc_val'].append(keras.ops.convert_to_numpy(acc_val))
                    perf_meas['tx_val_loss'].append(keras.ops.convert_to_numpy(tx_val_loss))
                perf_meas['tx_loss'].append(keras.ops.convert_to_numpy(tx_loss))
            else:
                # One step of receiver training
                rx_loss, accuracy = train_rx(
                    model, optimizer_rx, batch_input, batch_labels, sigma)
                if (validation_input is None) or (validation_labels is None):
                    training_status_string = create_training_status_string(
                        epoch=index_epoch, total_epochs=epochs, batch=batch_count, total_batches=number_batches,
                        rx_loss=rx_loss, accuracy=accuracy)
                else:
                    _, _, rx_val_loss, acc_val = model(
                        validation_input, validation_labels, sigma)
                    training_status_string = create_training_status_string(
                        epoch=index_epoch, total_epochs=epochs, batch=batch_count, total_batches=number_batches,
                        rx_loss=rx_loss, accuracy=accuracy, acc_val=acc_val, rx_val_loss=rx_val_loss)
                    perf_meas['rx_val_loss'].append(keras.ops.convert_to_numpy(rx_val_loss))
                    perf_meas['acc_val'].append(keras.ops.convert_to_numpy(acc_val))
            perf_meas['rx_loss'].append(keras.ops.convert_to_numpy(rx_loss))
            perf_meas['acc'].append(keras.ops.convert_to_numpy(accuracy))
            # Printing periodically the progress
            if batch_count % print_iteration == 0:
                print_training_status(training_status_string, start_time, start_time0, print_time)
                start_time = time.time()
            batch_count += 1

    # Once alternating training is done, the receiver is fine-tuned.
    start_time = time.time()
    print('Receiver fine-tuning... ')
    for index_epoch in range(receiver_finetuning_epochs):
        batch_count = 0
        train_input, train_labels = shuffle_dataset(
            train_input, train_labels)
        for batch_input, batch_labels in get_batch_dataset(train_input, train_labels, training_batch_size):
            rx_loss, accuracy = train_rx(
                model, optimizer_rx2, batch_input, batch_labels, sigma)
            if (validation_input is None) or (validation_labels is None):
                training_status_string = create_training_status_string(
                    epoch=index_epoch, total_epochs=receiver_finetuning_epochs, batch=batch_count, total_batches=number_batches,
                    rx_loss=rx_loss, accuracy=accuracy)
            else:
                _, _, rx_val_loss, acc_val = model(
                    validation_input, validation_labels, sigma)
                training_status_string = create_training_status_string(
                    epoch=index_epoch, total_epochs=receiver_finetuning_epochs, batch=batch_count, total_batches=number_batches,
                    rx_loss=rx_loss, accuracy=accuracy, acc_val=acc_val, rx_val_loss=rx_val_loss)
                perf_meas['rx_val_loss'].append(keras.ops.convert_to_numpy(rx_val_loss))
                perf_meas['acc_val'].append(keras.ops.convert_to_numpy(acc_val))
            perf_meas['rx_loss'].append(keras.ops.convert_to_numpy(rx_loss))
            perf_meas['acc'].append(keras.ops.convert_to_numpy(accuracy))
            if batch_count % print_iteration == 0:
                print_training_status(training_status_string, start_time, start_time0, print_time)
                start_time = time.time()
            batch_count += 1

    return perf_meas


def print_training_status(training_status_string, start_time, start_time0, print_time):
    """
    Helper function to print training status with consistent formatting

    Args:
        training_status_string: Formatted training status string from create_training_status_string
        start_time: Timestamp for last print iteration
        start_time0: Timestamp for overall training start
        print_time: Function to format elapsed time
    """
    print(
        f"{training_status_string}, Time: {time.time() - start_time:.2f}s, Tot. time: {print_time(time.time() - start_time0)}")
