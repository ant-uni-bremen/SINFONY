#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40 2025

@author: beck
GCM implemented in tensorflow

Belongs to simulation framework for numerical results of the articles:
1. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. doi: 10.48550/arXiv.2412.05103.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

import os
import gc
# import tensorflow as tf
# import tensorflow.keras as keras
import keras
from keras.optimizers import SGD, Adam
import numpy as np
from matplotlib import pyplot as plt
import time

import datasets
from utilities.my_training import gpu_select, epoch2iterationboundaries, new_optimizer
import sinfony_wrapper as sw
import model_evaluation
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule, print_time, get_ram
import sinfony_io
import gcm as gcm_model


if __name__ == '__main__':

    gpu_select(number=-2, memory_growth=True, cpus=64)
    use_weights = True

    # Choose project/dataset
    # Possible data sets: mnist, cifar10, fraeser, hise, speechcommands, urbansound8k
    wrapper = 'mnist'
    simulation = 'snr'          # snr, memory, working_memory

    load = True
    filename_gcm = 'GCM_Ne1_Na20'
    gcm_input = 0               # 0: sinfony output, 1: last layer input, 2: image input
    # Number of last layer input features used by GCM [only active for gcm_input==1]
    number_features = -1        # -1: all features are used
    # Probabilistic GCM decisions + Optimal policy: True, Optimal policy: False
    gcm_decision_policy = True

    # SINFONY version
    transceiver_split = 1       # image recognition: 0, sinfony: 1
    sinfony_version = 0         # high bandwidth: 0, low bandwidth: 1
    # image_split: One image (False) or multi-image classification based on different views (true)
    image_split = True

    # Training parameters
    snr_training = [-4, 6]      # Default training SNR [-4, 6]
    training_batch_size = 64    # Default: 64
    training_epochs = 1         # Default: 1
    # GCM learning rate schedule
    epoch_bound = [2, 10]
    values = [1.0e-0, 1.0e-1, 1.0e-2]
    learning_rate_schedule = 1.0e-0  # 1.0e-0, None
    opt_class = SGD             # SGD, Adam

    # Joint training
    joint_training = False
    # Parameters updated in alternation or jointly, default: True
    alternating = True
    differentiable_memory = False
    joint_gcm_training_epochs = 1
    joint_sinfony_training_epochs = 1
    alternating_training_iterations = 20
    epoch_bound2 = []
    values2 = [1.0e-3]
    learning_rate_schedule2 = 1.0e-3  # 1.0e-3, None
    opt_class2 = SGD             # SGD, Adam
    # optimizer2 = Adam()

    # Evaluation settings
    validation_rounds = 10      # Default: 10 (100 for fraeser)
    validation_batch_size = 64  # Default: 64
    # SNR Simulation
    snr_range = [10, 20]       # Default: [-30,20]
    snr_step_size = 1           # 1
    # Memory simulation
    # SNR for memory_size simulation, evaluate for training SNR?
    snr_evaluation = [20, 20]   # Default: [20, 20]
    fixed_dataset_per_memory_size = True
    memory_sizes = []           # Automatically logarithmic spacing
    # Working memory simulation
    working_memory_sizes = []
    print('Simulation starts...')

    # Load dataset
    dataset_name = wrapper
    if dataset_name == 'urbansound8k':
        # validation_split needs to be set manually for urbansound8k
        train_input_norm, train_labels, test_input_norm, test_labels = datasets.load_dataset(
            dataset_name, image_split=image_split, validation_split=0.90)
    else:
        train_input_norm, train_labels, test_input_norm, test_labels = datasets.load_dataset(
            dataset_name, image_split=image_split)
    datasets.summarize_dataset(
        train_input_norm, train_labels, test_input_norm, test_labels)
    number_classes, image_shapes = datasets.get_data_properties(
        test_labels, test_input_norm)

    # --------------- TRAINING + EVALUATION -------------------

    if gcm_input == 1:
        last_layer_input = True
    elif gcm_input == 0:
        last_layer_input = False
    else:
        last_layer_input = True
    # Load sinfony wrapper
    path_script = os.path.dirname(os.path.abspath(__file__))
    subpath_results = os.path.join(path_script, 'models', wrapper)
    if gcm_input != 2:
        template_files, _ = sw.template_models(wrapper)
        filename_sinfony = sw.select_sinfony(template_files=template_files, image_split=image_split,
                                             transceiver_split=transceiver_split, sinfony_version=sinfony_version)
        pathfile_sinfony = os.path.join(subpath_results, filename_sinfony)
        sinfony = sw.select_wrapper(transceiver_split, number_classes,
                                    image_shapes, path=pathfile_sinfony, last_layer_input=last_layer_input)
        sinfony.set_snr(snr_training)
        if gcm_input == 1 and number_features != -1:
            # Select k most important features and set rest to zero
            weights_last_layer = sinfony.model_full.get_weights()[-2]
            feature_importance = np.sum(np.abs(weights_last_layer), axis=1)
            top_k_indices = np.argpartition(
                feature_importance, -number_features)[-number_features:]
            # weights_k_important = np.zeros(
            #     weights_last_layer.shape[0], dtype='float32')
            # weights_k_important[top_k_indices] = 1
            sinfony.model = gcm_model.wrap_with_output_masking(
                sinfony.model, top_k_indices)
            # sinfony_weights = sinfony.model_full.get_weights()
            # sinfony_weights[-2] = weights_k_important
            # sinfony.model.set_weights(sinfony_weights)
    else:
        sinfony = []
        filename_sinfony = ''

    # Calculate iteration boundaries for learning rate schedule
    if not learning_rate_schedule:
        boundaries = epoch2iterationboundaries(
            epoch_bound, train_input_norm[0].shape[0], training_batch_size)
        learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
    if not learning_rate_schedule2:
        boundaries2 = epoch2iterationboundaries(
            epoch_bound2, train_input_norm[0].shape[0], training_batch_size)
        learning_rate_schedule2 = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries2, values2)
    opt_config = {"learning_rate": learning_rate_schedule, "momentum": 0.9}
    optimizer = new_optimizer(opt_class=opt_class, opt_config=opt_config)
    optimizer2 = opt_class2(
        learning_rate=learning_rate_schedule2, momentum=0.9)

    # Create filenames
    if gcm_input != 2 and differentiable_memory is True:
        filename_gcm = filename_gcm + '_differentiable'
    if gcm_input == 0:
        filename = filename_gcm + '+' + filename_sinfony
    elif gcm_input == 1:
        filename = filename_gcm + '_Nfeatures' + \
            str(number_features) + '+' + filename_sinfony
    else:
        filename = filename_gcm + '_' + dataset_name
    if joint_training is True and gcm_input != 2:
        filename = filename + '+joint_training'
    pathfile = os.path.join(path_script, subpath_results, filename)

    results = {}
    saveobj = savemodule(form='npz')
    if simulation == 'memory':
        fn_simulation = '_memory'
    elif simulation == 'working_memory':
        fn_simulation = '_working_memory'
    else:
        fn_simulation = ''
    filename_prefix = ''
    filename_suffix = '_results'
    pathfile_results = os.path.join(path_script, subpath_results,
                                    filename_prefix + filename + filename_suffix + fn_simulation)
    pathfile_results_opt = pathfile_results + '_opt'

    # Main simulation
    accuracy_opt = 0
    if simulation == 'memory':
        print('Memory Size Simulation')

        accuracy, loss, memory_sizes = model_evaluation.evaluate_gcm_memory(gcm_input, sinfony, transceiver_split, train_input_norm, train_labels, snr_training, test_input_norm, test_labels, snr_evaluation, training_epochs=training_epochs, batch_size=training_batch_size,
                                                                            validation_batch_size=validation_batch_size, opt_class=opt_class, opt_config=opt_config, validation_rounds=validation_rounds, gcm_decision_policy=gcm_decision_policy, fixed_dataset_per_memory_size=fixed_dataset_per_memory_size, memory_sizes=memory_sizes)
        if gcm_decision_policy is True:
            accuracy_opt = accuracy[1]
            accuracy = accuracy[0]

    elif simulation == 'snr' or simulation == 'working_memory':
        if simulation == 'snr':
            print('SNR Simulation')
        else:
            print('Working Memory Capacity Simulation')

        # GCM input computation
        exemplars = gcm_model.compute_gcm_input_data(
            train_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
        exemplar_labels = train_labels
        if load is False or gcm_input == 2:
            test_exemplars = gcm_model.compute_gcm_input_data(
                test_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
            test_exemplars_labels = test_labels
        else:
            test_exemplars = 0
            test_exemplars_labels = 0
        # Easily OOM if not enough RAM! Lines are for debugging
        # number_exemplars = 10000  # 60000
        # exemplars = exemplars[0:number_exemplars, ...]
        # exemplar_labels = exemplar_labels[0:number_exemplars, ...]

        # GCM model and training - Only GCM training first
        gcm = gcm_model.GeneralizedContextModel(
            exemplars, exemplar_labels, similarity_gradient=1.0)
        # tf.debugging.check_numerics(
        #     gcm.attention_weights, message="input contains NaNs or infs")
        gcm.compile(optimizer=optimizer,
                    loss='categorical_crossentropy', metrics=['accuracy'])

        if load is False:
            # GCM training
            print('GCM training starts...')
            gcm.fit(exemplars, exemplar_labels, validation_data=(
                test_exemplars, test_exemplars_labels), batch_size=training_batch_size, epochs=training_epochs)
            print('GCM training complete!')
            # Save the GCM model
            print('Saving model...')
            sinfony_io.try_save(gcm, pathfile, to_weights=use_weights)
            print('Model saved.')

        if load is True and joint_training is False:
            # Load existing GCM model:
            print('Loading model...')
            gcm = sinfony_io.try_load(gcm, pathfile, from_weights=use_weights)
            print('Model loaded.')

        if gcm_input != 2:
            # Combine SINFONY + GCM for evaluation or joint training
            # Compose them into one model
            if differentiable_memory is True and joint_training is True:
                # GCM version with differentiable memory -> trained end-to-end
                # NOTE: Too complex for gradient computation with weight sharing
                sinfony.model.trainable = True
                gcm_diff = gcm_model.GeneralizedContextModelDifferentiableMemory(
                    exemplars.shape[1], exemplar_labels, similarity_gradient=1.0)
                gcm_diff.set_weights(
                    gcm.get_weights()[1:][:-2] + gcm.get_weights()[1:][-2:])
                input_image = sw.clone_model_inputs(sinfony.model)
                features = sinfony.model(input_image)

                image_memory = []
                for train_input_norm_item in train_input_norm:
                    image_memory.append(
                        keras.ops.convert_to_tensor(train_input_norm_item))
                features_memory = sinfony.model(image_memory)
                gcm_output = gcm_diff(features, features_memory)
                # Final model
                sinfony_gcm = keras.Model(
                    inputs=input_image, outputs=gcm_output)
            else:
                input_image = sw.clone_model_inputs(sinfony.model)
                # input_image = sinfony.model.input
                features = sinfony.model(input_image)
                # Input reshaper
                # features_flat = keras.layers.Flatten()(features)
                gcm_output = gcm(features)
                # Final model
                sinfony_gcm = keras.Model(
                    inputs=input_image, outputs=gcm_output)
        else:
            sinfony_gcm = gcm

        if load is True and joint_training is True:
            # Load existing model:
            print('Loading model...')
            # sinfony_gcm = keras.models.load_model(pathfile)
            sinfony_gcm = sinfony_io.try_load(
                sinfony_gcm, pathfile, from_weights=use_weights)
            print('Model loaded.')

        sinfony_gcm.compile(optimizer=optimizer2,
                            loss='categorical_crossentropy', metrics=['accuracy'])

        if load is False and joint_training is True:
            if differentiable_memory is True:
                print('Joint training with differentiable memory starts...')
                history_sinfony = sinfony_gcm.fit(train_input_norm, exemplar_labels, validation_data=(
                    test_input_norm, test_exemplars_labels), batch_size=training_batch_size, epochs=alternating_training_iterations)
            else:
                print('Joint alternating training starts...')
                start_time = time.time()
                for idx_alternate in range(0, alternating_training_iterations):
                    # SINFONY + GCM Training
                    # Note: Keeps optimizer state between iterations
                    # Trainable sinfony model weights
                    sinfony.model.trainable = True
                    # Freeze GCM since memory is fixed
                    if alternating is True:
                        gcm.trainable = False
                    sinfony_gcm.compile(optimizer=optimizer2,
                                        loss='categorical_crossentropy', metrics=['accuracy'])
                    history_sinfony = sinfony_gcm.fit(train_input_norm, exemplar_labels, validation_data=(
                        test_input_norm, test_exemplars_labels), batch_size=training_batch_size, epochs=joint_gcm_training_epochs)
                    # Calculate new GCM exemplars for memory
                    exemplars = gcm_model.compute_gcm_input_data(
                        train_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
                    test_exemplars = gcm_model.compute_gcm_input_data(
                        test_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
                    exemplar_var = [
                        v for v in gcm.weights if "exemplars_memory" in v.name][0]
                    exemplar_var.assign(exemplars)
                    # gcm.get_layer('exemplars_labels').set_weights(
                    #     [exemplar_labels])
                    # Freeze the sinfony model weights
                    sinfony.model.trainable = False
                    # Train GCM based on new exemplars
                    if alternating is True:
                        gcm.trainable = True
                    gcm.compile(optimizer=optimizer,
                                loss='categorical_crossentropy', metrics=['accuracy'])
                    history_gcm = gcm.fit(exemplars, exemplar_labels, validation_data=(
                        test_exemplars, test_exemplars_labels), batch_size=training_batch_size, epochs=joint_sinfony_training_epochs)
                    # Free RAM, otherwise it accumulates
                    gc.collect()
                    print(
                        f"Iteration: {idx_alternate + 1}/{alternating_training_iterations}, CE: {history_gcm.history['val_loss'][-1]:.4f}, Acc: {history_gcm.history['val_accuracy'][-1]:.2f}, Time: {print_time(time.time() - start_time)}, RAM: {get_ram():.2f} GB")

            # Save the model
            print('Saving model...')
            sinfony_io.try_save(sinfony_gcm, pathfile, to_weights=use_weights)
            sinfony_gcm.save(pathfile)
            print('Model saved.')

        if simulation == 'working_memory':

            if gcm_input == 2:
                accuracy, loss, working_memory_sizes = model_evaluation.evaluate_gcm_working_memory(gcm, test_exemplars, test_exemplars_labels, validation_rounds=validation_rounds,
                                                                                                    batch_size=validation_batch_size, gcm_decision_policy=gcm_decision_policy, working_memory_sizes=working_memory_sizes)
            else:
                sinfony.set_snr(snr_evaluation)
                accuracy, loss, working_memory_sizes = model_evaluation.evaluate_gcm_working_memory(sinfony_gcm, test_input_norm, test_labels, validation_rounds=validation_rounds,
                                                                                                    batch_size=validation_batch_size, gcm_decision_policy=gcm_decision_policy, working_memory_sizes=working_memory_sizes)
            if gcm_decision_policy is True:
                accuracy_opt = accuracy[1]
                accuracy = accuracy[0]

        else:
            # Evaluation of model
            snrs = mop.snr_range2snrlist(snr_range, snr_step_size)
            print('Evaluate model...')
            print(filename)
            # Evaluate model for different SNRs
            if transceiver_split == 1:
                # SINFONY/RL-SINFONY
                accuracy, loss = model_evaluation.evaluate_sinfony(sinfony_gcm, test_input_norm, test_labels,
                                                                   snrs=snrs, validation_rounds=validation_rounds, gcm_decision_policy=gcm_decision_policy)
                if gcm_decision_policy is True:
                    accuracy_opt = accuracy[1]
                    accuracy = accuracy[0]
            else:
                if gcm_input == 2:
                    # GCM image recognition
                    accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                        gcm, test_exemplars, test_exemplars_labels, with_gcm=gcm_decision_policy)
                else:
                    # Standard image recognition: Evaluate model accuracy once for test data
                    accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                        sinfony_gcm, test_input_norm, test_labels, with_gcm=gcm_decision_policy)
                # Independent from SNR / constant, but plotted over SNR range
                loss = np.array(loss_i) * np.ones(snrs.shape)
                if gcm_decision_policy is True:
                    accuracy = np.array(accuracy_i[0]) * np.ones(snrs.shape)
                    accuracy_opt = np.array(
                        accuracy_i[1]) * np.ones(snrs.shape)
                else:
                    accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
                print(f'> {accuracy_i * 100.0:.3f}')
    else:
        print('Simulation not implemented')
        loss = 0
        accuracy = 0

    # Show performance curve
    if simulation == 'memory':
        x_axis = memory_sizes
    elif simulation == 'working_memory':
        x_axis = working_memory_sizes
    else:
        x_axis = snrs

    if simulation == 'memory':
        xlabel = 'Memory Size'
    elif simulation == 'working_memory':
        xlabel = 'Working Memory Size'
    else:
        xlabel = 'SNR'
    plt.figure(1)
    plt.semilogy(x_axis, 1 - accuracy)
    if gcm_decision_policy is True:
        plt.semilogy(x_axis, 1 - accuracy_opt)
    plt.xlabel(xlabel)
    plt.ylabel(
        'semantic performance measure: classification error rate')
    plt.figure(2)
    plt.semilogy(x_axis, loss)
    plt.xlabel(xlabel)
    plt.ylabel('Crossentropy Loss')

    # Save evaluation
    print('Save evaluation...')
    results['val_loss'] = loss
    results['val_acc'] = accuracy
    if simulation == 'memory':
        results['memory_size'] = memory_sizes
    elif simulation == 'working_memory':
        results['working_memory_size'] = working_memory_sizes
    else:
        results['snr'] = snrs
    saveobj.save(pathfile_results, results)
    if gcm_decision_policy is True:
        results['val_acc'] = accuracy_opt
        saveobj.save(pathfile_results_opt, results)
    print('Evaluation saved.')
