#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:42:43 2024

@author: beck
Module for semantic communication model evaluation

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# import os
import gc

# LOADED PACKAGES
import time
import numpy as np

# Tensorflow 2 packages
import tensorflow as tf

# Own packages
from utilities.my_functions import print_time, get_ram
import utilities.my_math_operations as mop
import gcm as gcm


def find_layer_by_name(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, 'layers'):  # If the layer has sublayers (e.g., nested model)
            found = find_layer_by_name(layer, name)
            if found:
                return found
    return None  # Not found


def gcm_train(exemplars, exemplar_labels, test_exemplars, test_exemplars_labels, optimizer, training_epochs, batch_size):
    ''' GCM training for subset of memory
    '''
    gcm_model = gcm.GeneralizedContextModel(
        exemplars, exemplar_labels, similarity_gradient=1.0)
    gcm_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
    gcm_model.fit(exemplars, exemplar_labels, validation_data=(
        test_exemplars, test_exemplars_labels), batch_size=batch_size, epochs=training_epochs)
    return gcm_model


def evaluate_gcm_memory(gcm_input, sinfony, transceiver_split, train_input, train_labels, snr_training, test_input, test_labels, snr_test, training_epochs=20, batch_size=64, validation_batch_size=None, opt_class=tf.keras.optimizers.SGD, opt_config={"learning_rate": 1e-3, "momentum": 0.9}, validation_rounds=10, gcm_decision_policy=False, fixed_dataset_per_memory_size=True, memory_sizes=None):
    '''Evaluate GCM for different memory sizes for multiple validation rounds (= dataset epochs)
    fixed_dataset_per_memory_size: Use the same dataset per memory size for simulation speedup
    '''
    start_time = time.time()
    eval_meas = [[], [], []]

    if not validation_batch_size:
        validation_batch_size = batch_size

    if not memory_sizes:
        memory_sizes = gcm.logarithmic_scale_inclusive(
            train_input[0].shape[0])

    exemplar_labels = train_labels
    test_exemplars_labels = test_labels
    if gcm_input == 2:
        exemplars = gcm.compute_gcm_input_data(
            train_input, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
        test_exemplars = gcm.compute_gcm_input_data(
            test_input, gcm_input, sinfony, transceiver_split, snr_test, batch_size=validation_batch_size)
    else:
        exemplars = 0
        test_exemplars = 0

    for idx_memory, memory_size in enumerate(memory_sizes):
        loss_i = 0
        accuracy_i = 0
        accuracy_i_2 = 0
        if fixed_dataset_per_memory_size is True and gcm_input != 2:
            exemplars = gcm.compute_gcm_input_data(
                train_input, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
            test_exemplars = gcm.compute_gcm_input_data(
                test_input, gcm_input, sinfony, transceiver_split, snr_test, batch_size=validation_batch_size)
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            if fixed_dataset_per_memory_size is False and gcm_input != 2:
                exemplars = gcm.compute_gcm_input_data(
                    train_input, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
                test_exemplars = gcm.compute_gcm_input_data(
                    test_input, gcm_input, sinfony, transceiver_split, snr_test, batch_size=validation_batch_size)
            # Random subset of memory of given size
            indices_memory = np.random.choice(
                exemplars.shape[0], size=memory_size, replace=False)
            # GCM training for subset of memory
            optimizer = gcm.new_optimizer(
                opt_class=opt_class, opt_config=opt_config)
            gcm_model = gcm_train(exemplars[indices_memory, ...], exemplar_labels[indices_memory, ...],
                                  test_exemplars, test_exemplars_labels, optimizer, training_epochs, batch_size)
            if gcm_decision_policy is True:
                # GCM + SINFONY validation step
                # GCM suboptimal random decision policy
                accuracy_ii, accuracy_ii_2, loss_ii = evaluate_gcm(
                    gcm_model, test_exemplars, test_exemplars_labels, batch_size=validation_batch_size)
                accuracy_i_2 = (validation_round * accuracy_i_2 +
                                accuracy_ii_2) / (validation_round + 1)
            else:
                # SINFONY validation step
                # Raw empirical accuracy with optimal decision policy
                loss_ii, accuracy_ii = gcm_model.evaluate(
                    test_exemplars, test_exemplars_labels, batch_size=validation_batch_size)

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            # Free RAM, otherwise it accumulates
            gc.collect()
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}, RAM: {get_ram():.2f} GB')
        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        eval_meas[2].append(accuracy_i_2)
        print(
            f'Iteration: {idx_memory + 1}/{len(memory_sizes)}, Memory Size: {memory_size}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
        if gcm_decision_policy is True:
            accuracy = [np.array(eval_meas[1]), np.array(eval_meas[2])]
        else:
            accuracy = np.array(eval_meas[1])
        loss = np.array(eval_meas[0])
    return accuracy, loss, memory_sizes


def evaluate_gcm_working_memory(evaluated_model, test_input, test_labels, validation_rounds=10, batch_size=32, gcm_decision_policy=False, working_memory_sizes=None):
    '''Evaluate GCM for different working memory capacity sizes for multiple validation rounds
    evaluated_model: Keras model to be evaluated
    '''
    start_time = time.time()
    eval_meas = [[], [], []]
    raw_attention_weights = [
        v for v in evaluated_model.weights if "raw_weights" in v.name][0]
    raw_attention_weights0 = raw_attention_weights.numpy()
    size_attention_weights = raw_attention_weights0.shape[-1]

    if not working_memory_sizes:
        working_memory_sizes = range(-size_attention_weights,  # -size_attention_weights+1
                                     size_attention_weights+1)

    for idx_memory, working_memory_size in enumerate(working_memory_sizes):
        # Take GCM's k largest or smallest weights to model working memory capacity
        if working_memory_size < 0:
            top_k_indices = np.argsort(raw_attention_weights0)[
                :, :, 0:-working_memory_size]
        else:
            top_k_indices = np.argsort(raw_attention_weights0)[:, :, ::-1][
                :, :, 0:working_memory_size]
        weights_k_top = np.zeros_like(raw_attention_weights0)
        weights_k_top[:, :, top_k_indices] = raw_attention_weights0[
            :, :, top_k_indices]
        raw_attention_weights.assign(weights_k_top)

        # Evaluate GCM with current working memory capacity for multiple rounds
        loss_i = 0
        accuracy_i = 0
        accuracy_i_2 = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            if gcm_decision_policy is True:
                # GCM + SINFONY validation step
                # GCM suboptimal random decision policy
                accuracy_ii, accuracy_ii_2, loss_ii = evaluate_gcm(
                    evaluated_model, test_input, test_labels, batch_size=batch_size)
                accuracy_i_2 = (validation_round * accuracy_i_2 +
                                accuracy_ii_2) / (validation_round + 1)
            else:
                # SINFONY validation step
                # Raw empirical accuracy with optimal decision policy
                loss_ii, accuracy_ii = evaluated_model.evaluate(
                    test_input, test_labels, batch_size=batch_size)
            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            # Free RAM, otherwise it accumulates
            gc.collect()
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}, RAM: {get_ram():.2f} GB')
        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        eval_meas[2].append(accuracy_i_2)
        print(f'Iteration: {idx_memory + 1}/{len(working_memory_sizes)}, Working Memory Size: {working_memory_size}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
    if gcm_decision_policy is True:
        accuracy = [np.array(eval_meas[1]), np.array(eval_meas[2])]
    else:
        accuracy = np.array(eval_meas[1])
    loss = np.array(eval_meas[0])
    return accuracy, loss, working_memory_sizes


def evaluate_sinfony(evaluated_model, test_input, test_labels, snrs=np.linspace(-30, 20, 1), validation_rounds=10, batch_size=32, gcm_decision_policy=False):
    '''Evaluate SINFONY over noisy SNR range for multiple validation rounds (= dataset epochs)
    evaluated_model: Keras model to be evaluated
    '''
    # SINFONY AE
    start_time = time.time()
    eval_meas = [[], [], []]
    noise_layer = find_layer_by_name(evaluated_model, 'gaussian_noise2')
    for snr_index, snr in enumerate(snrs):
        # Evaluate for each SNR in SNR range
        sigma = mop.snr2standard_deviation(snr)
        sigma_test = np.array([sigma, sigma], dtype='float32')
        # Set standard deviation weights of Noise layer in AE approach
        # evaluated_model.get_layer('gaussian_noise2').set_weights([sigma_test])
        noise_layer.set_weights([sigma_test])
        # evaluated_model.layers[-2].set_weights([sigma_test])
        loss_i = 0
        accuracy_i = 0
        accuracy_i_2 = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            # Evaluate for validation_rounds with different noise realizations (akin to training epochs)
            if gcm_decision_policy is True:
                # GCM + SINFONY validation step
                # GCM suboptimal random decision policy
                accuracy_ii, accuracy_ii_2, loss_ii = evaluate_gcm(
                    evaluated_model, test_input, test_labels, batch_size=batch_size)
                accuracy_i_2 = (validation_round * accuracy_i_2 +
                                accuracy_ii_2) / (validation_round + 1)
            else:
                # SINFONY validation step
                # Raw empirical accuracy with optimal decision policy
                loss_ii, accuracy_ii = evaluated_model.evaluate(
                    test_input, test_labels, batch_size=batch_size)

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            # Free RAM, otherwise it accumulates
            gc.collect()
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}, RAM: {get_ram():.2f} GB')
        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        eval_meas[2].append(accuracy_i_2)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
    if gcm_decision_policy is True:
        accuracy = [np.array(eval_meas[1]), np.array(eval_meas[2])]
    else:
        accuracy = np.array(eval_meas[1])
    loss = np.array(eval_meas[0])
    return accuracy, loss


def evaluate_gcm(evaluated_model, test_input, test_labels, batch_size=32, cce=tf.keras.losses.CategoricalCrossentropy()):
    '''Evaluate GCM suboptimal random decision policy
    '''
    prediction_probs = evaluated_model.predict(
        test_input, batch_size=batch_size)
    # evaluate also captures regularization losses
    # which only dependend on weights for SINFONY
    # and hence constant during testing time
    reg_loss = np.sum(
        evaluated_model.losses) if evaluated_model.losses else 0.0
    loss_ii = cce(test_labels, prediction_probs) + reg_loss
    # GCM decision policy
    accuracy_ii = np.mean(
        np.sum(prediction_probs * test_labels, axis=-1))
    # Optimal decision policy
    accuracy_ii_opt = np.mean(
        np.argmax(prediction_probs, axis=-1) == np.argmax(test_labels, axis=-1))
    return accuracy_ii, accuracy_ii_opt, loss_ii


def evaluate_rlsinfony(evaluated_model, test_input, test_labels, snrs=np.linspace(-30, 20, 1), validation_rounds=10):
    '''Evaluate RL-SINFONY over noisy SNR range for multiple validation rounds (= dataset epochs)
    evaluated_model: RL-SINFONY model to be evaluated
    '''
    # RL-SINFONY
    start_time = time.time()
    eval_meas = [[], []]
    for snr_index, snr in enumerate(snrs):
        # Evaluate for each SNR in SNR range
        sigma = mop.snr2standard_deviation(snr)
        sigma_test = np.array([sigma, sigma], dtype='float32')
        loss_i = 0
        accuracy_i = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            # Evaluate for validation_rounds with different noise realizations (akin to training epochs)
            # RL-SINFONY validation step
            _, _, loss_ii, accuracy_ii = evaluated_model(
                test_input, test_labels, sigma=tf.constant(sigma_test, dtype='float32'))

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            # Free RAM, otherwise it accumulates
            gc.collect()
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}, RAM: {get_ram():.2f} GB')

        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
    accuracy = np.array(eval_meas[1])
    loss = np.array(eval_meas[0])
    return accuracy, loss


def evaluate_image_classifier(evaluated_model, test_input, test_labels, batch_size=32, with_gcm=False):
    '''Evaluate image classifier Keras model on validation/test set
    '''
    if with_gcm is True:
        # GCM + SINFONY validation step
        # GCM suboptimal random decision policy
        accuracy_i, accuracy_i_2, loss_i = evaluate_gcm(
            evaluated_model, test_input, test_labels, batch_size=batch_size)
        accuracy = [accuracy_i, accuracy_i_2]
    else:
        # Standard image recognition: Evaluate model accuracy once for test data
        loss_i, accuracy = evaluated_model.evaluate(
            test_input, test_labels, batch_size=batch_size)
        # Independent from SNR / constant, but plotted over SNR range
    return accuracy, loss_i
