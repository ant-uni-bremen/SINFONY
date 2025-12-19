#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 16:06:40 2024

@author: beck
Wrapper for easy use of SINFONY models in Human on Mars Initiative and its seed project Human-integrated Swarm Exploration (HiSE)
The TensorFlow models were generated in version 2.15.0 for better compatibility.

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022).
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
3. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. doi: 10.48550/arXiv.2412.05103.
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
import utilities.my_math_operations as mop
import datasets
import model_evaluation
from sinfony import try_load_model
import sinfony_architectures.resnet as resnet


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
        self.model = try_load_model(pathfile, custom_objects={
            "Residual": resnet.Residual,
            "ResNetBlock": resnet.ResnetBlock,
            "ResidualBottleneck": resnet.ResidualBottleneck,
        })
        self.model_full = None
        if last_layer_input is True:
            # Extract models last layer input features: Features = Input to last softmax layer
            # Note: only required for psychologist experiments
            self.model_full = self.model
            self.model = self.model.layers[-2]

    def __call__(self, image_data, batch_size=32, preprocess=False):
        '''Execute ResNet classifier given image data image_data
        INPUT
        image_data: Image data as RGB values
        batch_size: Batch size for evaluation
        preprocess: Include image preprocessing in wrapper
        OUTPUT
        semantic_probs: Semantic class predictions with probabilities (array of floats) or features with last_layer_input==True
        '''
        if preprocess:
            image_data = datasets.preprocess_pixels_image(image_data)
        semantic_probs = self.model.predict(image_data, batch_size=batch_size)
        return semantic_probs


def clone_model_inputs(model):
    '''Clones the model inputs to rebuild and extend the model
    '''
    inputs = model.inputs
    if isinstance(inputs, list):
        # Multiple inputs
        new_inputs = [
            tf.keras.Input(shape=inp.shape[1:], dtype=inp.dtype) for inp in inputs
        ]
    else:
        # Single input
        new_inputs = tf.keras.Input(shape=inputs.shape[1:], dtype=inputs.dtype)
    return new_inputs


def remove_last_inner_layer(model: tf.keras.Model) -> tf.keras.Model:
    """
    Given a Functional Keras model whose last layer is itself a Model (nested),
    this function removes the last layer of the nested model, re-wraps it,
    and returns a new model with the trimmed nested model still in place.

    Assumes:
    - model is Functional (not Sequential).
    - model.layers[-1] is a Model or Sequential.
    """
    if not isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
        raise ValueError("Expected a Functional Keras model.")

    outer_input = clone_model_inputs(model)
    # outer_input = model.input
    x = outer_input

    # Apply all layers except the final nested model
    for layer in model.layers[:-1]:
        x = layer(x)

    last_layer = model.layers[-1]
    if not isinstance(last_layer, tf.keras.Model):
        raise ValueError(
            "Last layer must be a nested model (e.g., Sequential or Functional).")

    # Rebuild the nested model without its last layer
    nested_input = tf.keras.Input(shape=last_layer.input_shape[1:])
    # nested_input = last_layer.input
    nested_x = nested_input
    for nested_layer in last_layer.layers[:-1]:
        nested_x = nested_layer(nested_x)

    trimmed_nested_model = tf.keras.Model(
        inputs=nested_input, outputs=nested_x, name=last_layer.name + "_trimmed")

    # Plug the trimmed nested model back into the graph
    x = trimmed_nested_model(x)

    # Reconstruct the full model
    new_model = tf.keras.Model(
        inputs=outer_input, outputs=x, name=model.name + "_trimmed")

    return new_model


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
        self.model = try_load_model(pathfile, custom_objects={
            "Residual": resnet.Residual,
            "ResNetBlock": resnet.ResnetBlock,
            "ResidualBottleneck": resnet.ResidualBottleneck,
        })
        self.model_full = None
        if last_layer_input is True:
            # Extract models last layer input features: Features = Input to last softmax layer
            # Note: only required for psychologist experiments
            self.model_full = self.model
            self.model = remove_last_inner_layer(self.model_full)
        self.set_snr(snr=6)
        self.get_snr()

    def get_snr(self):
        '''Read SNR from model
        '''
        noise_layer = model_evaluation.find_layer_by_name(
            self.model, 'gaussian_noise2')
        self.snr = 10*np.log10(1 / noise_layer.get_weights()[0] ** 2)

    def set_snr(self, snr=6):
        '''Set SNR in SINFONY model
        Channel included in SINFONY model:
        Set standard deviation weights of the noise in the channel layer in AE approach
        '''
        sigma = mop.snr2standard_deviation(np.array(snr))
        if sigma.shape in (1, ()):
            sigma_test = np.array([sigma, sigma], dtype='float32')
        else:
            # Reverse SNR list ordering
            if sigma[0] > sigma[1]:
                sigma_test = sigma[::-1]
            else:
                sigma_test = sigma
        noise_layer = model_evaluation.find_layer_by_name(
            self.model, 'gaussian_noise2')
        if (noise_layer.get_weights()[0] != [sigma_test]).any():
            noise_layer.set_weights([sigma_test])
        self.get_snr()

    def __call__(self, image_data, snr=6, batch_size=32, preprocess=False):
        '''Execute SINFONY given image data image_data for SNR value snr
        INPUT
        image_data: Image data as RGB values
        batch_size: Batch size for evaluation
        preprocess: Include image preprocessing in wrapper
        snr: Signal-to-Noise-Ratio - scalar snr or interval [snr_min, snr_max] (SINFONY is trained within -4 to 6 but usable outside this interval)
        dist: TODO - Distances from robots/rovers for calculation of simple line of sight path loss
        OUTPUT
        semantic_probs: Semantic class predictions with probabilities (array of floats) or features with last_layer_input==True
        '''
        if preprocess:
            image_data = datasets.preprocess_pixels_image(image_data)
        self.set_snr(snr)
        # Execution of SINFONY
        semantic_probs = self.model.predict(image_data, batch_size=batch_size)
        return semantic_probs


def template_models(wrapper):
    '''Load template files for given wrapper
    wrapper: Wrapper to be loaded
    template_files: Template SINFONY model names
    dataset_name: Name of the respective dataset
    '''
    if wrapper == 'hise':
        # HiSE template files
        # SINFONY versions: sinfony_hise256_v4_imagenet, sinfony_hise256_v4_imagenet_ntx64_2
        # Only image recognition: ResNet18_hise256_v4_imagenet
        # Only image recognition based on one image view: ResNet18_hise256_v4_imagenet_oneimage
        template_files = [['sinfony_hise256_v4_imagenet', 'sinfony_hise256_v4_imagenet_ntx64_2'],
                          'ResNet18_hise256_v4_imagenet', 'ResNet18_hise256_v4_imagenet_oneimage']
        dataset_name = 'hise256'
    elif wrapper == 'fraeser':
        # Human Rover template files
        # SINFONY versions: sinfony18_fraeser_lr1e-3_3, sinfony18_fraeser_ntx128
        # Only image recognition: ResNet18_fraeser
        template_files = [['sinfony18_fraeser_lr1e-3_3',
                           'sinfony18_fraeser_ntx128'], 'ResNet18_fraeser', '']
        dataset_name = 'fraeser'
    elif wrapper == 'mnist':
        # MNIST template files
        # SINFONY versions: sinfony14_MNIST_ntx56_Ne20_snr-4_6_human, sinfony14_MNIST_ntx14_Ne20_snr-4_6_human
        # Only image recognition: ResNet14_MNIST_Ne20_human
        template_files = [
            ['sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human'], 'ResNet14_MNIST_Ne20_human', '']
        dataset_name = 'mnist'
    elif wrapper == 'cifar10':
        # CIFAR10 template files
        # SINFONY versions: sinfony20_CIFAR_ntx64_snr-4_6_human, sinfony20_CIFAR_ntx16_snr-4_6_human
        # Only image recognition: ResNet20_CIFAR_human
        template_files = [
            ['sinfony20_CIFAR_ntx64_snr-4_6_human', 'sinfony20_CIFAR_ntx16_snr-4_6_human'], 'ResNet20_CIFAR_human', '']
        dataset_name = 'cifar10'
    elif wrapper == 'hise64':
        # HiSE template files for low 64 resolution
        # SINFONY versions: sinfony_hise64_v4, sinfony_hise64_v4_ntx16
        # Only image recognition: ResNet20_hise64_v4_2
        # Only image recognition based on one image view: ResNet20_hise64_v4_oneimage
        template_files = [['sinfony_hise64_v4', 'sinfony_hise64_v4_ntx16'],
                          'ResNet20_hise64_v4_2', 'ResNet20_hise64_v4_oneimage']
        dataset_name = 'hise64'
    elif wrapper == 'fraeser64':
        # Human Rover template files for low 64 resolution
        # SINFONY versions: sinfony6_fraeser64, sinfony6_fraeser64_ntx16
        # Only image recognition: ResNet6_fraeser64_test
        template_files = [
            ['sinfony6_fraeser64', 'sinfony6_fraeser64_ntx16'], 'ResNet6_fraeser64_test', '']
        dataset_name = 'fraeser64'
    elif wrapper == 'speechcommands':
        # Speech command template files: audio dataset from tensorflow
        template_files = [['sinfony_speechcommands_imagenet', 'sinfony_speechcommands_imagenet_ntx64'],
                          'ResNet18_speechcommands_imagenet', '']
        dataset_name = 'speech_commands'
    elif wrapper == 'urbansound8k':
        # Urbansound8k template files: audio dataset
        template_files = [['sinfony_urbansound8k_imagenet', 'sinfony_urbansound8k_imagenet_ntx64'],
                          'ResNet18_urbansound8k_imagenet', '']
        dataset_name = 'urbansound8k'
    else:
        template_files = None
        dataset_name = None
    return template_files, dataset_name


def select_sinfony(path, template_files, image_split=True, transceiver_split=1, sinfony_version=0, last_layer_input=False):
    '''Select and load the SINFONY model
    '''
    if image_split is True:
        if transceiver_split == 1:
            # SINFONY approach with communication channel:
            filename2 = template_files[0][sinfony_version]
            sinfony = SinfonyWrapper(path=path, filename=filename2,
                                     last_layer_input=last_layer_input)
        else:
            # Only image recognition:
            filename2 = template_files[1]
            sinfony = ResNetWrapper(path=path, filename=filename2,
                                    last_layer_input=last_layer_input)
    else:
        if transceiver_split == 0:
            # Only image recognition based on one image view:
            filename2 = template_files[2]
            sinfony = ResNetWrapper(path=path, filename=filename2,
                                    last_layer_input=last_layer_input)
        else:
            sinfony = None
    return sinfony, filename2


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # import my_training as mt
    # mt.gpu_select(number=3, memory_growth=False)

    # Choose project/dataset
    # Possible data sets: mnist, cifar10, fraeser (human rover), fraeser64, hise, hise64, hise256, speechcommands, urbansound8k
    # Number after dataset name is the resolution of the images
    wrapper = 'mnist'

    last_layer_input2 = False
    transceiver_split = 1       # image recognition: 0, sinfony: 1
    sinfony_version = 0         # high bandwidth: 0, low bandwidth: 1
    # image_split: One image (False) or multi-image classification based on different views (true)
    image_split = True
    evaluation_mode = 1         # Interface data: 0, SNR evaluation: 1

    # Load sinfony wrapper
    path_script = os.path.dirname(os.path.abspath(__file__))
    path2 = os.path.join(path_script, 'models', wrapper)
    subpath_results = os.path.join(path_script, 'models_output')

    template_files, dataset_name = template_models(wrapper)
    sinfony, filename2 = select_sinfony(path=path2, template_files=template_files, image_split=image_split,
                                        transceiver_split=transceiver_split, sinfony_version=sinfony_version, last_layer_input=last_layer_input2)

    train_input_norm, train_labels, test_input_norm, test_labels = datasets.load_dataset(
        dataset_name, image_split=image_split, preprocess=True)
    # train_input_norm, test_input_norm = datasets.preprocess_pixels(
    #     train_input, test_input)
    datasets.summarize_dataset(
        train_input_norm, train_labels, test_input_norm, test_labels)

    # Evaluation of model

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
            path_script, subpath_results, 'output_' + filename2)
        if transceiver_split == 1:
            # Add evaluated SNR for SINFONY
            pathfile2 = pathfile2 + '_snr' + str(snr_evaluation) + 'dB'
        results = {}
        if last_layer_input2 is True:
            results['last_layer_features_validation'] = sinfony_output_validation
            results['last_layer_features_training'] = sinfony_output_training
        else:
            results['estimated_probabilities_validation'] = sinfony_output_validation
            results['estimated_probabilities_training'] = sinfony_output_training
        results['class_validation'] = test_labels
        results['class_training'] = train_labels
        np.savez(pathfile2, results)
        print('Interface data saved.')

    elif evaluation_mode == 1 and last_layer_input2 is False:
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
