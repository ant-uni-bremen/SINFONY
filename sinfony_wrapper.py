#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 16:06:40 2024

@author: beck
Wrapper for easy use of SINFONY models in Human on Mars Initiative and its seed project Human-integrated Swarm Exploration (HiSE)

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022).
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
from matplotlib import pyplot as plt

# Tensorflow 2 packages
import tensorflow as tf


# Own packages
import my_math_operations as mop
import datasets
import model_evaluation


class ResNetWrapper():
    """Wrapper for ResNet classifier (Keras, Tensorflow 2) for interface within HiSE and Human Rover Journal
    """

    def __init__(self, path='models/fraeser', filename='ResNet4_fraeser64_test', last_layer_input=False):
        '''Initialize object by loading the SINFONY Keras model
        INPUT
        path: Path where the model file lies
        filename: Name of the model file
        last_layer_input: Output the input of the last layer, i.e., the extracted feature vector
        '''
        pathfile = os.path.join(path, filename)
        self.model = tf.keras.models.load_model(pathfile)
        if last_layer_input is True:
            # Extract models last layer input features: Features = Input to last softmax layer
            # Note: only required for psychologist experiments
            self.model = self.model.layers[-2]

    def __call__(self, image_data):
        '''Execute ResNet classifier given image data image_data
        INPUT
        image_data: Image data as RGB values
        OUTPUT
        semantic_probs: Semantic class predictions with probabilities (array of floats) or features with last_layer_input==True
        '''
        semantic_probs = self.model.predict(image_data)
        return semantic_probs


class SinfonyWrapper():
    """Wrapper for SINFONY (Keras, Tensorflow 2) for interface within HiSE and Human Rover Journal
    TODO: Only for AE approach, no reinforcement learning included so far
    """

    def __init__(self, path='models/fraeser', filename='sinfony4_fraeser64_test', last_layer_input=False):
        '''Initialize object by loading the SINFONY Keras model
        INPUT
        path: Path where the model file lies
        filename: Name of the model file
        last_layer_input: Output the input of the last layer, i.e., the extracted feature vector
        '''
        pathfile = os.path.join(path, filename)
        self.model = tf.keras.models.load_model(pathfile)
        if last_layer_input is True:
            # Extract models last layer input features: Features = Input to last softmax layer
            # Note: only required for psychologist experiments
            output_model1 = self.model.get_layer('gaussian_noise2').input
            output_model_noise = self.model.get_layer(
                'gaussian_noise2')(output_model1)
            output_model3 = self.model.layers[-1].layers[-2](
                output_model_noise)
            self.model = tf.keras.Model(
                inputs=self.model.input, outputs=output_model3)

    def __call__(self, image_data, snr=6):
        '''Execute SINFONY given image data image_data for SNR value snr
        INPUT
        image_data: Image data as RGB values
        snr: Signal-to-Noise-Ratio - scalar snr or interval [snr_min, snr_max] (SINFONY is trained within -4 to 6 but usable outside this interval)
        dist: TODO - Distances from robots/rovers for calculation of simple line of sight path loss
        OUTPUT
        semantic_probs: Semantic class predictions with probabilities (array of floats) or features with last_layer_input==True
        '''
        # Channel included in SINFONY model:
        # Set standard deviation weights of the noise in the channel layer in AE approach
        sigma = mop.snr2standard_deviation(np.array(snr))
        if sigma.shape in (1, ()):
            sigma_test = np.array([sigma, sigma], dtype='float32')
        else:
            sigma_test = sigma
        if (self.model.get_layer('gaussian_noise2').weights[0].numpy() != [sigma_test]).any():
            self.model.get_layer(
                'gaussian_noise2').set_weights([sigma_test])
        # Execution of SINFONY
        semantic_probs = self.model.predict(image_data)
        return semantic_probs


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load sinfony wrapper
    path_script = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path_script, 'models', 'fraeser')
    subpath_results = os.path.join(path_script, 'models_output')

    last_layer_input = False
    transceiver_split = 1       # image recognition: 0, sinfony: 1

    if transceiver_split == 1:
        # Models for testing so far:
        # SINFONY approach with communication channel: sinfony4_fraeser64_test, sinfony_hise64, sinfony_hise256_imagenet
        filename = 'sinfony4_fraeser64_test'
        sinfony = SinfonyWrapper(path=path, filename=filename,
                                 last_layer_input=last_layer_input)
    else:
        # Only image recognition: ResNet4_fraeser64_test, ResNet20_hise64, ResNet18_hise256_imagenet
        filename = 'ResNet4_fraeser64_test'
        sinfony = ResNetWrapper(path=path, filename=filename,
                                last_layer_input=last_layer_input)

    # Possible data sets:
    # mnist, cifar10, fraeser, fraeser64, hise, hise64, hise256
    # Number after dataset name is the resolution of the images
    dataset_name = 'fraeser64'
    train_input, train_labels, test_input, test_labels = datasets.load_dataset(
        dataset_name)
    train_input_norm, test_input_norm = datasets.preprocess_pixels(
        train_input, test_input)
    datasets.summarize_dataset(
        train_input, train_labels, test_input, test_labels)

    # Evaluation of model
    evaluation_mode = 1         # Interface data: 0, SNR evaluation: 1

    if evaluation_mode == 0:
        # Evaluation parameters
        snr_evaluation = 6

        # Provided functionality
        # Give results of training and validation data to application beyond
        print('Calculate interface data...')
        if transceiver_split == 1:
            sinfony_output_validation = sinfony(
                test_input_norm, snr=snr_evaluation)
            sinfony_output_training = sinfony(
                train_input_norm, snr=snr_evaluation)
        else:
            sinfony_output_validation = sinfony.model.predict(
                test_input_norm)
            sinfony_output_training = sinfony.model.predict(
                train_input_norm)

        # Save interface data
        print('Save interface data...')
        pathfile2 = os.path.join(
            path_script, subpath_results, 'output_' + filename)
        if transceiver_split == 1:
            # Add evaluated SNR for SINFONY
            pathfile2 = pathfile2 + '_snr' + str(snr_evaluation) + 'dB'
        results = {}
        if last_layer_input is True:
            results['last_layer_features_validation'] = sinfony_output_validation
            results['last_layer_features_training'] = sinfony_output_training
        else:
            results['estimated_probabilities_validation'] = sinfony_output_validation
            results['estimated_probabilities_training'] = sinfony_output_training
        results['class_validation'] = test_labels
        results['class_training'] = train_labels
        np.savez(pathfile2, results)
        print('Interface data saved.')

    elif evaluation_mode == 1 and last_layer_input is False:
        # For demonstration: Evaluation of SINFONY over SNR range
        # Evaluation parameters
        snr_range = [-30, 20]
        snr_step_size = 1
        validation_rounds = 10
        snrs = mop.snr_range2snrlist(snr_range, snr_step_size)

        # Evaluate model for different SNRs
        print('Evaluate model...')
        if transceiver_split == 1:
            accuracy, loss = model_evaluation.evaluate_sinfony(sinfony.model, test_input_norm, test_labels,
                                                               snrs=snrs, validation_rounds=validation_rounds)
        else:
            # Classifier only evaluated once
            accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                sinfony.model, test_input_norm, test_labels)
            loss = np.array(loss_i) * np.ones(snrs.shape)
            accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
            print(f'> {accuracy_i * 100.0:.3f}')
        # Show performance curve
        plt.figure(1)
        plt.semilogy(snrs, 1 - accuracy)
        plt.xlabel('SNR')
        plt.ylabel('semantic performance measure: classification error rate')
        plt.figure(2)
        plt.semilogy(snrs, loss)
        plt.xlabel('SNR')
        plt.ylabel('crossentropy loss')
