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
import shutil
import tensorflow as tf
# import tensorflow.keras as keras
import keras
from keras.optimizers import SGD, Adam
import numpy as np

import datasets
from utilities.my_training import gpu_select, new_optimizer
import sinfony_wrapper as sw
import model_evaluation
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule
import sinfony_io
import gcm as gcm_model
from sinfony_test import compare_model_accuracies


def extract_parameters_from_filename(filename, template_files):
    """
    Extracts parameters from filename to reconstruct the original parameters.

    Args:
        filename: The filename to parse
        template_files: List of two lists containing template names

    Returns:
        tuple: (number_features, sinfony_version, gcm_input, joint_training, differentiable_memory, basename, transceiver_split) 
    """
    # Initialize defaults
    number_features = 20  # default value
    sinfony_version = 0   # default value
    gcm_input = 1         # default value
    joint_training = True  # default value
    differentiable_memory = False  # default value
    transceiver_split = 0  # default value

    # Extract basename (everything up to _Nfeatures or +)
    basename = filename
    if '_Nfeatures' in filename:
        # Find the position of _Nfeatures and extract everything before it
        nfeatures_pos = filename.find('_Nfeatures')
        basename = filename[:nfeatures_pos]
        # Extract number_features
        nfeatures_start = filename.find('_Nfeatures') + len('_Nfeatures')
        nfeatures_end = filename.find('+', nfeatures_start)
        if nfeatures_end == -1:  # If no + found, it might be the end
            nfeatures_end = len(filename)
        try:
            number_features = int(filename[nfeatures_start:nfeatures_end])
        except ValueError:
            print(ValueError)  # Keep default if parsing fails
        # If _Nfeatures is present, gcm_input should be 1 (last layer input)
        gcm_input = 1
    elif '+' in filename:
        # If there's a + but no _Nfeatures, gcm_input should be 0 (sinfony output)
        gcm_input = 0
        # Extract basename up to the first +
        plus_pos = filename.find('+')
        basename = filename[:plus_pos]
    else:
        # If neither _Nfeatures nor + is present, gcm_input should be 2 (image input)
        gcm_input = 2
        # Full filename is the basename
        basename = filename

    # Check for joint_training (indicated by '+joint_training')
    if '+joint_training' in filename:
        joint_training = True
    elif '+joint_training' not in filename:
        joint_training = False

    # Check for differentiable_memory (indicated by '_differentiable_' in filename)
    if '_differentiable' in filename:
        differentiable_memory = True

    # Analyze template_files to determine sinfony_version and transceiver_split
    # template_files is a list of two lists
    if template_files and len(template_files) >= 2:
        # Find the position of the filename in template_files
        # We need to look at the part of the filename after the first + and potentially before +joint_training
        name_part = filename
        if '+' in filename:
            # Extract everything after the first +
            first_plus_pos = filename.find('+')
            name_part = filename[first_plus_pos + 1:]
            # Remove +joint_training if present
            if '+joint_training' in name_part:
                name_part = name_part.split('+joint_training')[0]

        # Check if name_part exists in either of the template lists
        for i, template_list in enumerate(template_files):
            if name_part in template_list:
                # Found in list i
                # sinfony_version corresponds to the position of name_part in the template_list
                sinfony_version = template_list.index(name_part)
                transceiver_split = 1 if i == 0 else 0  # 1 for first list, 0 for second
                break

    return number_features, sinfony_version, gcm_input, joint_training, differentiable_memory, basename, transceiver_split


def swap_adjacent_pairs(lst):
    """
    Swaps adjacent pairs in a list.

    Args:
        lst: List to swap adjacent pairs in

    Returns:
        List with adjacent pairs swapped
    """
    # Create a copy to avoid modifying the original
    result = lst.copy()

    # Iterate through the list in increments of 2 and swap adjacent elements
    for i in range(0, len(result) - 1, 2):
        # Swap the elements at positions i and i+1
        result[i], result[i+1] = result[i+1], result[i]

    return result


def convert_model_weights(sinfony_gcm, tfsm_model, reverse_remaining_weights=True):
    """
    Compares two models and creates a weight mapping for synchronization.

    This function compares the weights of two models (SINFONY-GCM and tfsm_model) 
    and creates a mapping between their weights based on name matching. Remaining 
    unmatched weights are reordered and swapped to facilitate proper weight transfer.

    Args:
        sinfony_gcm: The SINFONY-GCM model
        tfsm_model: The TensorFlow model to compare against
        reverse_remaining_weights: If True, reverses and swaps adjacent pairs 
                                   of remaining weights for proper ordering

    Returns:
        list: A list of weights in the correct order for synchronization
    """

    # Retrieve all weights along with their names and indices
    sinfony_gcm_weight_names = [w.path.replace(
        '/', '_') for w in sinfony_gcm.weights]
    sinfony_gcm_weight_names_short = [
        w.name for w in sinfony_gcm.weights]
    tfsm_weight_names = [w.name[:-2]
                         for w in tfsm_model.weights]

    # Create a mapping based on the names
    weight_mapping = {}
    weight_not_mapping = {}
    mapped_indices = set()  # Save which tfsm_model weights were already matched

    for i, sinfony_name in enumerate(sinfony_gcm_weight_names):
        # Search for matching weight in tfsm_model
        matched = False
        for j, tfsm_name in enumerate(tfsm_weight_names):
            if sinfony_name == tfsm_name:
                weight_mapping[i] = tfsm_model.weights[j]
                mapped_indices.add(j)
                matched = True
                break
            elif sinfony_gcm_weight_names_short[i] == tfsm_name or tfsm_name in sinfony_name:
                weight_mapping[i] = tfsm_model.weights[j]
                mapped_indices.add(j)
                matched = True
                break

        if not matched:
            # Track weights not matched
            if i < len(tfsm_model.weights):
                weight_not_mapping[i] = sinfony_gcm_weight_names[i]

    # Create a list of all tfsm_model weights that have not been assigned
    unmapped_tfsm_weights = []
    for j, tfsm_weight in enumerate(tfsm_model.weights):
        if j not in mapped_indices:
            unmapped_tfsm_weights.append(tfsm_weight)

    # Reverse the list and swap adjacent pairs
    if reverse_remaining_weights:
        unmapped_tfsm_weights = swap_adjacent_pairs(
            unmapped_tfsm_weights[::-1])

    index_unmapped = 0
    for _, i in enumerate(weight_not_mapping):
        weight_mapping[i] = unmapped_tfsm_weights[index_unmapped]
        index_unmapped = index_unmapped + 1

    sorted_weights = [weight_mapping[key]
                      for key in sorted(weight_mapping.keys())]

    return sorted_weights


def set_deterministic(seed=42):
    """
    Forces deterministic behavior across different machines.

    Args:
        seed: Random seed to use
    """
    import random
    import platform
    # Seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Deterministische TF Operationen
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # oneDNN deaktivieren
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Threads fixieren
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print(f"Python:     {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras:      {keras.__version__}")
    print(f"NumPy:      {np.__version__}")
    print(f"CPU:        {platform.processor()}")


if __name__ == '__main__':

    # set_deterministic(seed=42)

    gpu_select(number=-2, memory_growth=True, cpus=64)
    use_weights = True

    # Choose project/dataset
    # Possible data sets: mnist, cifar10, fraeser, hise, speechcommands, urbansound8k
    wrapper = 'speechcommands'
    simulation = 'snr'          # snr, memory, working_memory

    # Load sinfony wrapper
    path_script = os.path.dirname(os.path.abspath(__file__))
    subpath_results = os.path.join(path_script, 'models', wrapper)
    entries = os.listdir(subpath_results)
    folders = [entry for entry in entries if os.path.isdir(
        os.path.join(subpath_results, entry))]
    template_files, _ = sw.template_models(wrapper)

    from_tfsm_model = False
    move_old_model = True

    load = True
    filename_gcm = 'GCM_Ne1_Na20'
    gcm_input = 1               # 0: sinfony output, 1: last layer input, 2: image input
    # Number of last layer input features used by GCM [only active for gcm_input==1]
    number_features = 20        # -1: all features are used
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
    joint_training = True
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
    snr_range = [20, 20]        # Default: [-30,20]
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

    for folder in folders:
        if from_tfsm_model is True:
            print('Trying conversion of: ' + folder)
        else:
            print('Testing: ' + folder)
        try:
            number_features, sinfony_version, gcm_input, joint_training, differentiable_memory, filename_gcm, transceiver_split = extract_parameters_from_filename(
                folder, template_files)

            # --------------- TRAINING + EVALUATION -------------------

            if gcm_input == 1:
                last_layer_input = True
            elif gcm_input == 0:
                last_layer_input = False
            else:
                last_layer_input = False
            if gcm_input != 2:
                template_files, _ = sw.template_models(wrapper)
                filename_sinfony = sw.select_sinfony(template_files=template_files, image_split=image_split,
                                                     transceiver_split=transceiver_split, sinfony_version=sinfony_version)
                pathfile_sinfony = os.path.join(
                    subpath_results, filename_sinfony)
                sinfony = sw.select_wrapper(transceiver_split, number_classes,
                                            image_shapes, path=pathfile_sinfony, last_layer_input=last_layer_input)
                sinfony.set_snr(snr_training)
                if gcm_input == 1 and number_features != -1:
                    # Select k most important features and set rest to zero
                    weights_last_layer = sinfony.model_full.get_weights()[-2]
                    feature_importance = np.sum(
                        np.abs(weights_last_layer), axis=1)
                    top_k_indices = np.argpartition(
                        feature_importance, -number_features)[-number_features:]
                    sinfony.model = gcm_model.wrap_with_output_masking(
                        sinfony.model, top_k_indices)
            else:
                sinfony = []
                filename_sinfony = ''

            # Calculate iteration boundaries for learning rate schedule
            if not learning_rate_schedule:
                boundaries = gcm_model.epoch2iterationboundaries(
                    epoch_bound, train_input_norm[0].shape[0], training_batch_size)
                learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries, values)
            if not learning_rate_schedule2:
                boundaries2 = gcm_model.epoch2iterationboundaries(
                    epoch_bound2, train_input_norm[0].shape[0], training_batch_size)
                learning_rate_schedule2 = keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries2, values2)
            opt_config = {
                "learning_rate": learning_rate_schedule, "momentum": 0.9}
            optimizer = new_optimizer(
                opt_class=opt_class, opt_config=opt_config)
            optimizer2 = opt_class2(
                learning_rate=learning_rate_schedule2, momentum=0.9)

            # Create filenames
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

            if differentiable_memory is False:
                try:
                    print('Checking whether GCM with differentiable memory was used...')
                    tfsm_model2 = tf.saved_model.load(pathfile)
                    sig = tfsm_model2.signatures['serving_default']
                    differentiable_memory = any('differentiable_memory' in name for name in sig.structured_outputs.keys()
                                                )
                    print(differentiable_memory)
                except Exception as e:
                    print(f"Error: {e}")

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

                # GCM model and training - Only GCM training first
                gcm = gcm_model.GeneralizedContextModel(
                    exemplars, exemplar_labels, similarity_gradient=1.0)
                gcm.compile(optimizer=optimizer,
                            loss='categorical_crossentropy', metrics=['accuracy'])

                if load is False:
                    # GCM training
                    print('GCM training starts...')
                    gcm.fit(exemplars, exemplar_labels, validation_data=(
                        test_exemplars, test_exemplars_labels), batch_size=training_batch_size, epochs=training_epochs)
                    print('GCM training complete!')
                if joint_training is False:
                    if load is True:
                        if from_tfsm_model:
                            tfsm_model = sinfony_io.try_load_model(pathfile)
                            try:
                                gcm.set_weights(tfsm_model.get_weights())
                            except Exception as e:
                                print(
                                    f"Fallback after: {e}")
                                gcm.set_weights(tfsm_model.get_weights(
                                )[-2:] + tfsm_model.get_weights()[:-2])
                        else:
                            # Load existing GCM model:
                            print('Loading model...')
                            try:
                                gcm = sinfony_io.try_load(gcm, pathfile)
                                print('Model loaded.')
                            except Exception as e:
                                print(
                                    f"Failed to load model, jump to next file: {e}")
                                raise
                    # else:
                    #     # Save the GCM model
                    #     print('Saving model...')
                    #     sinfony_io.try_save(gcm, pathfile, to_weights=use_weights)
                    #     print('Model saved.')

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
                                tf.constant(train_input_norm_item))
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

                if load is True and joint_training is True and from_tfsm_model is False:
                    # Load existing model:
                    print('Loading model...')
                    try:
                        # sinfony_gcm = keras.models.load_model(pathfile)
                        sinfony_gcm = sinfony_io.try_load(
                            sinfony_gcm, pathfile)
                        print('Model loaded.')
                    except Exception as e:
                        print(
                            f"Failed to load model; jump to next file: {e}")
                        raise

                sinfony_gcm.compile(optimizer=optimizer2,
                                    loss='categorical_crossentropy', metrics=['accuracy'])

                if from_tfsm_model is True and joint_training is True:

                    tfsm_model = sinfony_io.try_load_model(pathfile)

                    # Calculate test result
                    test_size = 100
                    if dataset_name != 'fraeser':
                        for var in tfsm_model.weights:
                            if 'stddev' in var.name:
                                print(f"Found: {var.name} = {var.numpy()}")
                                new_stddev = tf.constant(
                                    [1/100000, 1/100000], dtype=tf.float32)
                                var.assign(tf.cast(new_stddev, var.dtype))
                        test_result = tfsm_model(
                            test_input_norm[0][:test_size, ...])
                    else:
                        tfsm_model2 = tf.saved_model.load(pathfile)
                        infer = tfsm_model2.signatures['serving_default']
                        # Variable finden und neu setzen
                        for var in tfsm_model2.variables:
                            if 'stddev' in var.name:
                                print(f"Found: {var.name} = {var.numpy()}")
                                new_stddev = tf.constant(
                                    [1/100000, 1/100000], dtype=tf.float32)
                                var.assign(tf.cast(new_stddev, var.dtype))
                        if gcm_input == 1:
                            if number_features == -1:
                                test_result = infer(
                                    input_4=test_input_norm[0][:test_size, ...],
                                    input_5=test_input_norm[1][:test_size, ...]
                                )
                            else:
                                test_result = infer(
                                    input_6=test_input_norm[0][:test_size, ...],
                                    input_7=test_input_norm[1][:test_size, ...]
                                )
                        else:
                            test_result = infer(
                                input_1=test_input_norm[0][:test_size, ...],
                                input_2=test_input_norm[1][:test_size, ...]
                            )

                    test_result = list(test_result.values())[0]

                    if dataset_name != 'fraeser':
                        reverse_remaining_weights = True
                    else:
                        reverse_remaining_weights = False
                    sorted_weights = convert_model_weights(
                        sinfony_gcm, tfsm_model, reverse_remaining_weights=reverse_remaining_weights)

                    sinfony_gcm.set_weights(sorted_weights)

                    if dataset_name != 'fraeser':
                        test_result2 = sinfony_gcm(
                            test_input_norm[0][:test_size, ...])
                    else:
                        test_result2 = sinfony_gcm(
                            [test_input_norm[0][:test_size, ...], test_input_norm[1][:test_size, ...]])
                    # model_deviation = np.sqrt(np.mean(
                    #     np.abs(test_result - test_result2) ** 2))
                    model_deviation = np.mean(
                        np.abs(test_result - test_result2))
                    model_deviation_equal = model_deviation <= 1e-3
                    if model_deviation_equal:
                        print(
                            f'✅ Models have same output! (model_deviation = {model_deviation:.2e})')
                    else:
                        print(
                            f'❌ Models differ! (model_deviation = {model_deviation:.2e})')

                    sinfony_io.try_save(sinfony_gcm, pathfile)
                else:
                    print(
                        'No joint training: Already converted models are tested based on the weights files.')
                    model_deviation_equal = False
                    if load and joint_training is False and from_tfsm_model:
                        sinfony_io.try_save(gcm, pathfile)

            #       if load is False and joint_training is True:
            #           [Saving after joint training]

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
                            accuracy = np.array(
                                accuracy_i[0]) * np.ones(snrs.shape)
                            accuracy_opt = np.array(
                                accuracy_i[1]) * np.ones(snrs.shape)
                        else:
                            accuracy = np.array(
                                accuracy_i) * np.ones(snrs.shape)
                        print(f'> {accuracy_i * 100.0:.3f}')
            else:
                print('Simulation not implemented')
                loss = 0
                accuracy = 0

            # # Save evaluation
            print('Set up evaluation...')
            results['val_loss'] = loss
            results['val_acc'] = accuracy
            if simulation == 'memory':
                results['memory_size'] = memory_sizes
            elif simulation == 'working_memory':
                results['working_memory_size'] = working_memory_sizes
            else:
                results['snr'] = snrs
            # saveobj.save(pathfile_results, results)
            if gcm_decision_policy is True:
                results['val_acc'] = accuracy_opt
                # saveobj.save(pathfile_results_opt, results)
            print('Evaluation set up.')

            try:
                results2 = saveobj.load(pathfile_results)
            except Exception as e:
                print(
                    f"Simulation results file not found, jump to next file: {e}")
                raise

            accuracy_load = results2['val_acc'][np.isin(results2['snr'], snrs)]

            accuracy_equal = compare_model_accuracies(
                accuracy, accuracy_load, tolerance=2e-2)

            # Backup the folder
            if (accuracy_equal or model_deviation_equal) and move_old_model:
                try:
                    backup_path = os.path.join(
                        os.path.dirname(pathfile), 'backup')
                    os.makedirs(backup_path, exist_ok=True)
                    shutil.move(pathfile, backup_path)
                    print(f"Successfully moved {pathfile} to {backup_path}")
                except Exception as move_error:
                    print(f"Failed to move {pathfile} to backup: {move_error}")
        except Exception as e:
            print(f"Error: {e}")
            # raise
            # pass  # Keep default if parsing fails
        finally:
            keras.backend.clear_session()
