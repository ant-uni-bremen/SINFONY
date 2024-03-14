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

# LOADED PACKAGES
import time
import numpy as np

# Tensorflow 2 packages
import tensorflow as tf

# Own packages
from my_functions import print_time
import my_math_operations as mop


def evaluate_sinfony(evaluated_model, test_input, test_labels, snrs=np.linspace(-30, 20, 1), validation_rounds=10):
    '''Evaluate SINFONY over noisy SNR range for multiple validation rounds (= dataset epochs)
    evaluated_model: Keras model to be evaluated
    '''
    # SINFONY AE
    start_time = time.time()
    eval_meas = [[], []]
    for snr_index, snr in enumerate(snrs):
        # Evaluate for each SNR in SNR range
        sigma = mop.snr2standard_deviation(snr)
        sigma_test = np.array([sigma, sigma], dtype='float32')
        # Set standard deviation weights of Noise layer in AE approach
        evaluated_model.get_layer('gaussian_noise2').set_weights([sigma_test])
        # evaluated_model.layers[-2].set_weights([sigma_test])
        loss_i = 0
        accuracy_i = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            # Evaluate for validation_rounds with different noise realizations (akin to training epochs)
            # SINFONY validation step
            loss_ii, accuracy_ii = evaluated_model.evaluate(
                test_input, test_labels)

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}')

        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
    accuracy = np.array(eval_meas[1])
    loss = np.array(eval_meas[0])
    return accuracy, loss


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
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}')

        # Append list with evaluation for each SNR value
        eval_meas[0].append(loss_i)
        eval_meas[1].append(accuracy_i)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')
    accuracy = np.array(eval_meas[1])
    loss = np.array(eval_meas[0])
    return accuracy, loss


def evaluate_image_classifier(evaluated_model, test_input, test_labels):
    '''Evaluate image classifier Keras model on validation/test set
    '''
    # Standard image recognition: Evaluate model accuracy once for test data
    loss_i, accuracy_i = evaluated_model.evaluate(test_input, test_labels)
    # Independent from SNR / constant, but plotted over SNR range
    return accuracy_i, loss_i
