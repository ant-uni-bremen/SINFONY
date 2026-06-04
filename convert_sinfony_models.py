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


# Own packages
# import datasets_keras2 as datasets
import datasets
import model_evaluation
import sinfony_architectures.resnet_rl_sinfony as resnet_rl_sinfony
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule
import utilities.my_training as mt
import utilities.my_training_tf1 as mt1
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import utilities.my_training as mf
from sinfony_io import try_load, try_save, try_load_model, try_load_weights, find_layer_by_name
import model_builder
import h5py
import re


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

# resnet_block (global 2-25, local 0-23):
# Old:  k0,b1, k2,b3, g4,b5, g6,b7, k8,b9, k10,b11, g12,b13, g14,b15, mm16,mv17, mm18,mv19, mm20,mv21, mm22,mv23
# New:  k,b, k,b, g,b,mm,mv, g,b,mm,mv,   k,b, k,b, g,b,mm,mv, g,b,mm,mv
# Reorder: 0,1, 2,3, 4,5,16,17, 6,7,18,19,  8,9, 10,11, 12,13,20,21, 14,15,22,23


def inspect_h5_weights(weights_path):
    with h5py.File(weights_path, 'r') as f:
        def print_weight(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'{name} {obj.shape} {obj.dtype}')
        f.visititems(print_weight)


def inspect_model_weights(model):
    for w in model.weights:
        print(w.path, w.shape)


def load_weights_remapped(model, weights_path):
    saved_weights = {}

    def collect_weights(name, obj):
        if isinstance(obj, h5py.Dataset):
            match = re.search(r'_(\d+)_(\d+)$', name)
            if match:
                global_idx = int(match.group(1))
                saved_weights[global_idx] = obj[:]

    with h5py.File(weights_path + '_weights.h5', 'r') as f:
        f.visititems(collect_weights)

    sw = [saved_weights[k] for k in sorted(saved_weights.keys())]

    # Remapping: old global index -> new position
    # conv2d: 0,1 -> same
    # resnet_block (2-25): reorder within block
    # resnet_block_1 (26-51): same pattern
    # resnet_block_2 (52-77): same pattern
    # batch_normalization_12 (78-81): same
    # dense (82-83): same

    def remap_resnet_block(sw, start):
        # old local: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
        # new local: 0,1,2,3,4,5,16,17,6,7,18,19,8,9,10,11,12,13,20,21,14,15,22,23
        old_order = [0, 1, 2, 3, 4, 5, 16, 17, 6, 7, 18, 19,
                     8, 9, 10, 11, 12, 13, 20, 21, 14, 15, 22, 23]
        return [sw[start + i] for i in old_order]

    def remap_resnet_block_1_2(sw, start):
        # Old order (global indices relative to start):
        # 0:k, 1:b, 2:k, 3:b, 4:k, 5:b,          <- kernels/biases unit1 + skip
        # 6:g, 7:b, 8:g, 9:b,                      <- BN unit0 gamma/beta
        # 10:k, 11:b, 12:k, 13:b,                  <- kernels/biases unit2
        # 14:g, 15:b, 16:g, 17:b,                  <- BN unit1+2 gamma/beta
        # 18:mm, 19:mv, 20:mm, 21:mv,              <- moving stats unit0+1
        # 22:mm, 23:mv, 24:mm, 25:mv               <- moving stats unit2+3

        # New model order:
        # unit1: k,b, k,b, k(skip),b, g,b,mm,mv, g,b,mm,mv
        # unit2: k,b, k,b,            g,b,mm,mv, g,b,mm,mv
        old_order = [0, 1, 2, 3, 4, 5,
                     6, 7, 18, 19, 8, 9, 20, 21,
                     10, 11, 12, 13,
                     14, 15, 22, 23, 16, 17, 24, 25]
        return [sw[start + i] for i in old_order]

    reordered = (
        [sw[0], sw[1]] +                    # conv2d
        remap_resnet_block(sw, 2) +         # resnet_block (24 weights)
        remap_resnet_block_1_2(sw, 26) +    # resnet_block_1 (26 weights)
        remap_resnet_block_1_2(sw, 52) +    # resnet_block_2 (26 weights)
        [sw[78], sw[79], sw[80], sw[81]] +  # batch_normalization_12
        [sw[82], sw[83]]                    # dense
    )

    print(f'Saved weights: {len(reordered)}')
    print(f'Model weights: {len(model.weights)}')

    all_match = True
    for i, (model_w, saved_w) in enumerate(zip(model.weights, reordered)):
        if model_w.shape != tuple(saved_w.shape):
            print(f'❌ {i}: {model_w.name} {model_w.shape} != {saved_w.shape}')
            all_match = False
        else:
            print(f'✅ {i}: {model_w.name} {model_w.shape}')

    if all_match:
        model.set_weights(reordered)
        print('✅ Weights loaded successfully!')
        model.save_weights(weights_path + '.weights.h5')
    else:
        print('❌ Shape mismatches found')

    return model


def remap_resnet_block_plain(sw, start, n_units=3):
    # Old order: (kb,gb) per unit, then all moving stats
    # unit0: k(0),b(1),k(2),b(3), g(4),b(5),g(6),b(7)
    # unit1: k(8),b(9),k(10),b(11), g(12),b(13),g(14),b(15)
    # unit2: k(16),b(17),k(18),b(19), g(20),b(21),g(22),b(23)
    # moving: mm(24),mv(25),mm(26),mv(27),mm(28),mv(29),mm(30),mv(31),mm(32),mv(33),mm(34),mv(35)
    n = n_units
    old_order = []
    for u in range(n):
        kb = 8 * u        # k,b,k,b
        gb = 8 * u + 4    # g,b,g,b
        mm = 8 * n + 4 * u  # mm,mv,mm,mv
        old_order += [kb, kb + 1, kb + 2, kb + 3,    # k,b,k,b
                      gb, gb + 1, mm, mm + 1,     # g,b,mm,mv
                      gb + 2, gb + 3, mm + 2, mm + 3]     # g,b,mm,mv
    return [sw[start + i] for i in old_order]


def remap_resnet_block_with_skip(sw, start, n_units=3):
    # Old order: (kb,gb) per unit, then all moving stats
    # unit0: k(0),b(1),k(2),b(3),k_skip(4),b_skip(5), g(6),b(7),g(8),b(9)
    # unit1: k(10),b(11),k(12),b(13), g(14),b(15),g(16),b(17)
    # unit2: k(18),b(19),k(20),b(21), g(22),b(23),g(24),b(25)
    # moving: mm(26),mv(27),...,mm(37),mv(37)
    n = n_units
    total_kb_gb_first = 10   # 6 kb + 4 gb
    total_kb_gb_rest = 8    # 4 kb + 4 gb
    moving_start = total_kb_gb_first + total_kb_gb_rest * (n - 1)  # 26

    old_order = []
    # Unit 0
    old_order += [0, 1, 2, 3, 4, 5,                    # k,b,k,b,k_skip,b_skip
                  6, 7, moving_start + 0, moving_start + 1,   # g,b,mm,mv
                  8, 9, moving_start + 2, moving_start + 3]   # g,b,mm,mv
    # Units 1..n-1
    for u in range(1, n):
        kb = total_kb_gb_first + total_kb_gb_rest * (u - 1)
        gb = kb + 4
        mm = moving_start + 4 * u
        old_order += [kb, kb + 1, kb + 2, kb + 3,   # k,b,k,b
                      gb, gb + 1, mm, mm + 1,    # g,b,mm,mv
                      gb + 2, gb + 3, mm + 2, mm + 3]    # g,b,mm,mv
    return [sw[start + i] for i in old_order]


def load_weights_remapped_20(model, weights_path):
    import re
    import h5py

    saved_weights = {}

    def collect_weights(name, obj):
        if isinstance(obj, h5py.Dataset):
            match = re.search(r'_(\d+)_(\d+)$', name)
            if match:
                global_idx = int(match.group(1))
                saved_weights[global_idx] = obj[:]

    with h5py.File(weights_path, 'r') as f:
        f.visititems(collect_weights)

    sw = [saved_weights[k] for k in sorted(saved_weights.keys())]

    reordered = (
        [sw[0], sw[1]] +
        remap_resnet_block_plain(sw, 2, n_units=3) +      # global 2-37  (36 weights)
        remap_resnet_block_with_skip(sw, 38, n_units=3) +  # global 38-75 (38 weights)
        remap_resnet_block_with_skip(sw, 76, n_units=3) +  # global 76-113 (38 weights)
        [sw[114], sw[115], sw[116], sw[117]] +
        [sw[118], sw[119]]
    )

    print(f'Saved weights: {len(reordered)}')
    print(f'Model weights: {len(model.weights)}')

    all_match = True
    for i, (model_w, saved_w) in enumerate(zip(model.weights, reordered)):
        if model_w.shape != tuple(saved_w.shape):
            print(f'❌ {i}: {model_w.name} {model_w.shape} != {saved_w.shape}')
            all_match = False
        else:
            print(f'✅ {i}: {model_w.name} {model_w.shape}')

    if all_match:
        model.set_weights(reordered)
        print('✅ Weights loaded successfully!')
    else:
        print('❌ Shape mismatches found')

    return model


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # SETTINGS_FILE = 'urbansound8k/semantic_config_urbansound8k_central.yaml'
    SETTINGS_FILE = 'cifar10/ResNet20_CIFAR_test2.yaml'
    # Change from 'settings' to 'models' to reload simulations settings
    SETTINGS_FOLDER = 'models'
    # Load the provided configuration file or the default one
    # python SINFONY.py semantic_config.yaml
    # Workaround for interactive sessions: Only allow config file names starting 'semantic_config'
    SETTINGS_FILE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][0:15].lower(
    ) == 'semantic_config' else SETTINGS_FILE
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)
    with open(settings_path, 'r', encoding='UTF8') as file:
        params = yaml.safe_load(file)
    load_settings = params['load_settings']
    dataset_settings = params['dataset']
    training_settings = params['training']
    rl_settings = params['reinforcement_learning']
    model_settings = params['model']
    transceiver_split = model_settings['communication']['transceiver_split']

    # Initialization
    # mt.gpu_select(number=load_settings.get('gpu', -2), memory_growth=False)
    # keras.backend.set_floatx(load_settings['numerical_precision'])

    # Simulation
    # Load model and reevaluate: False (default) # params.get('load', False)
    load = load_settings.get('load', False)
    use_weights = load_settings.get('use_weights', True)
    filename = load_settings['filename']
    # Sub path for saved data
    subpath_results = load_settings['path']
    # Path of script being executed
    pathfile_model = os.path.join(path_script, subpath_results, filename)
    pathfile_results = os.path.join(path_script, subpath_results,
                                    load_settings.get('simulation_filename_prefix', '') + filename + load_settings.get('simulation_filename_suffix', ''))
    saveobj = savemodule(form=load_settings.get('save_format', 'npz'))

    # Data set
    # mnist, cifar10, fashion_mnist, hirise64, hirisecrater, fraeser64
    dataset = dataset_settings['dataset']
    train_input, train_labels, test_input, test_labels = datasets.load_dataset(
        dataset, validation_split=dataset_settings['validation_split'], image_split=dataset_settings['image_split'], preprocess=True)
    # Summarize loaded dataset
    if dataset_settings['show_dataset'] is True:
        datasets.summarize_dataset(
            train_input, train_labels, test_input, test_labels)

    # Training
    # Batch size, SGD: 128/64, Adam: 500
    batch_size = training_settings['batch_size']
    # Number of epochs, 200 in CIFAR original implementation, 20 for MNIST
    number_epochs = training_settings['number_epochs']

    dataset_size = train_input[0].shape[0]
    iterations_per_epoch = dataset_size / batch_size
    total_number_iterations = number_epochs * iterations_per_epoch
    # Choose validation data set size: None/0, 100, 1000, test_input[0].shape[0]
    validation_dataset_size = training_settings['validation_dataset_size']
    if validation_dataset_size == 'full':
        validation_dataset_size = test_input[0].shape[0]
    # If computational heavy (RL approach), use subset of validation set
    valY = test_labels[:validation_dataset_size, ...]
    valX = mt1.create_batch(test_input, validation_dataset_size, 0)

    # RL training
    # (0) default AE, (1) Reinforcement learning training, (2) AE trained with rl-based training implementation
    rl = rl_settings['active']

    # Some training functionality of model.fit()
    if rl == 0:
        VERBOSE = 'auto'
        # - Callbacks for early stopping and model checkpoints -
        # EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
        early_stopping = []
        # keras.callbacks.ModelCheckpoint(pathfile_model, monitor = 'val_loss', verbose = VERBOSE, save_best_only = False, mode = 'auto', period = 1, save_weights_only = False, save_freq = 'epoch')
        model_checkpoint = []
        # Track training loss and accuracy of each batch iteration
        batch_tracking = mt.BatchTrackingCallback()
    else:
        VERBOSE = 'auto'
        batch_tracking = None
        model_checkpoint = None
        early_stopping = None

    # Optimizer
    optimizer, optimizer_tx, optimizer_rx2, spg_config = model_builder.create_optimizer(
        training_settings=training_settings,
        rl_settings=rl_settings,
        iterations_per_epoch=iterations_per_epoch,
        learning_rate=training_settings['learning_rate'],
        optimizer=training_settings['optimizer'],
        rl=rl,
    )

    # TRAINING AND EVALUATION SCRIPT

    # ResNet20 model
    if dataset.lower() == 'cifar10' and model_settings['resnet']['number_residual_units'] <= 2:
        print(
            'Warning: Number of residual units is below minimum number for CIFAR10 dataset!')
    number_classes, image_shapes = datasets.get_data_properties(
        test_labels, test_input)
    model, sigma_train = model_builder.create_model_from_config(
        params, number_classes, image_shapes)

    # if load is True:
    #     # Load existing model:
    #     print('Loading model...')
    #     # Load SINFONY via weights or model save
    #     model = try_load(model, pathfile_model, from_weights=use_weights)
    #     print('Model loaded.')
    #     # For loading, compile is necessary
    #     model.compile(optimizer=optimizer,
    #                   loss='categorical_crossentropy', metrics=['accuracy'])

    # if model.name != 'tfsm_layer':
    #     # Summarize AE-based SINFONY
    #     model.summary()

    # Convert models to new Tensorflow version
    # Load existing model and extract weights:
    # MNIST: ResNet14_MNIST4_Ne20_snr-4_6, ResNet14_MNIST6_Ne20_snr-4_6
    # CIFAR10: ResNet20_CIFAR4_snr-4_6, ResNet20_CIFAR6_snr-4_6
    save_only_weights = True
    load_from_weights = True
    # filename2 = 'ResNet14_MNIST4_Ne20_snr-4_6'

    pathfile_model2_results = os.path.join(path_script, subpath_results,
                                           load_settings['simulation_filename_prefix'] + filename)

    # Set weights to that of old model
    if load_from_weights is True:
        print('Loading model from weights...')
        # pathfile_weights = os.path.join(
        #     path_script, subpath_results, filename)
        # model(test_input, test_labels, np.array(
        #     [0.0, 0.0], dtype='float32'), np.array(0.0, dtype='float32'))
        if model.name == 'ResNet14_CIFAR10':
            model = load_weights_remapped(model, pathfile_model + '_weights.h5')
        elif model.name == 'ResNet20_CIFAR10':
            model = load_weights_remapped_20(model, pathfile_model + '_weights.h5')
        else:
            model = try_load_weights(model, pathfile_model)
        # pathfile_weights2 = os.path.join(
        #     path_script, subpath_results, filename, filename)
        # model2.load_weights(pathfile_weights2, skip_mismatch=False)
        # model = model2
        print('Weights loaded.')
    else:
        print('Loading model...')
        # Load SINFONY model
        pathfile_model2 = os.path.join(path_script, subpath_results, filename)
        model2 = try_load_model(pathfile_model2)
        # model2 = tf.keras.layers.TFSMLayer(
        #     pathfile_model2, call_endpoint="serving_default")
        print('Model loaded.')
        model2.summary()
        model = model2
        # model.set_weights(model2.get_weights())
        # model.save_weights(pathfile_model2 + '.h5')
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # if load is False:
    #     # Save model weights:
    #     try_save(model, pathfile_model, to_weights=save_only_weights)
    # else:
    #     # Compile and train model
    #     # Load training history to include evaluation
    #     print('Load training history...')
    #     results = saveobj.load(pathfile_model2_results)
    #     if results is None:
    #         results = {}
    #     else:
    #         results = dict(results)
    #     print('Loaded!')

    # # Evaluation of model
    # # Evaluation/Validation of model
    # evaluation_settings = params['evaluation']
    # snrs = mop.snr_range2snrlist(
    #     evaluation_settings['snr_range'], evaluation_settings['snr_step_size'])

    # print('Evaluate model...')
    # print(filename)
    # # Evaluate model for different SNRs
    # if transceiver_split == 1:
    #     # SINFONY/RL-SINFONY
    #     if rl >= 1:
    #         accuracy, loss = model_evaluation.evaluate_rlsinfony(model, test_input, test_labels,
    #                                                              snrs=snrs, validation_rounds=evaluation_settings['validation_rounds'])
    #     else:
    #         accuracy, loss = model_evaluation.evaluate_sinfony(model, test_input, test_labels,
    #                                                            snrs=snrs, validation_rounds=evaluation_settings['validation_rounds'])
    # else:
    #     # Standard image recognition: Evaluate model accuracy once for test data
    #     if rl >= 1:
    #         _, _, loss_i, accuracy_i = model(
    #             test_input, test_labels, sigma=tf.convert_to_tensor([0, 0], dtype='float32'))
    #     else:
    #         accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
    #             model, test_input, test_labels)
    #     # Independent from SNR / constant, but plotted over SNR range
    #     loss = np.array(loss_i) * np.ones(snrs.shape)
    #     accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
    #     print(f'Validation Accuracy: {accuracy_i * 100.0:.3f}%')
    # # Show performance curve
    # plt.figure(1)
    # plt.semilogy(snrs, 1 - accuracy)
    # plt.xlabel('SNR')
    # plt.ylabel(
    #     'semantic performance measure: classification error rate')
    # plt.figure(2)
    # plt.semilogy(snrs, loss)
    # plt.xlabel('SNR')
    # plt.ylabel('crossentropy loss')

    # # Save evaluation
    # print('Save evaluation...')
    # results['snr'] = snrs
    # results['val_loss'] = loss
    # results['val_acc'] = accuracy
    # saveobj.save(pathfile_results, results)
    # print('Evaluation saved.')

# EOF
