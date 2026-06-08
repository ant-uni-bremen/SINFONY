#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022).
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
3. E. Beck, H.- Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. doi: 10.48550/arXiv.2412.05103.
"""


import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from utilities.gpu_select_tf import gpu_select
    backend_tf = True
else:
    backend_tf = False
import keras

# Own packages
import datasets
import model_evaluation
import sinfony_architectures.resnet_rl_sinfony as resnet_rl_sinfony
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule
import utilities.my_layers_keras3 as mt
import utilities.my_dataset_handling as mdh
from sinfony_io import try_load, try_save, find_unique_results_path
import model_builder


def simulation_sinfony(settings_path, test_mode=False, load=None, snrs=None):
    '''SINFONY simulation
    '''
    with open(settings_path + '.yaml', 'r', encoding='UTF8') as file:
        params = yaml.safe_load(file)
    load_settings = params['load_settings']
    dataset_settings = params['dataset']
    training_settings = params['training']
    rl_settings = params['reinforcement_learning']
    model_settings = params['model']
    transceiver_split = model_settings['communication']['transceiver_split']

    # Initialization
    if backend_tf and not test_mode:
        gpu_select(number=load_settings.get('gpu', -2), memory_growth=False)
    keras.backend.set_floatx(load_settings['numerical_precision'])

    # Simulation
    # Load model and reevaluate: False (default) # params.get('load', False)
    if load is None:
        load = load_settings.get('load', False)
    use_weights = load_settings.get('use_weights', True)
    filename = load_settings['filename']
    # Sub path for saved data
    subpath_results = load_settings['path']
    # Path of script being executed
    path_script = os.path.dirname(os.path.abspath(__file__))
    pathfile_model = os.path.join(path_script, subpath_results, filename)
    pathfile_results = os.path.join(path_script, subpath_results,
                                    load_settings.get('simulation_filename_prefix', '') + filename + load_settings.get('simulation_filename_suffix', ''))
    save_file_ending = load_settings.get('save_format', 'npz')
    save_object = savemodule(form=save_file_ending)

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
    valX = mdh.create_batch(test_input, validation_dataset_size, 0)

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

    if load is True:
        # Load existing model:
        print('Loading model...')
        # Load SINFONY via weights or model save
        model = try_load(model, pathfile_model, from_weights=use_weights)
        print('Model loaded.')

    if model.name != 'tfsm_layer':
        # Summarize AE-based SINFONY
        model.summary()

    if rl == 0:
        # For loading, compile is necessary
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # Compile and train model
    if load is False:
        if rl >= 1:
            # RL-SINFONY
            sigma_train = keras.ops.convert_to_tensor(
                sigma_train, dtype='float32')
            results = resnet_rl_sinfony.rl_based_training(model, train_input, train_labels, optimizer, optimizer_tx, optimizer_rx2, validation_input=valX,
                                                          validation_labels=valY, epochs=number_epochs, training_batch_size=batch_size, sigma=sigma_train, stochastic_policy_gradient_config=spg_config)
        else:
            # SINFONY AE-like
            history = model.fit(train_input, train_labels, epochs=number_epochs, batch_size=batch_size, validation_data=(
                valX, valY), callbacks=[batch_tracking, model_checkpoint, early_stopping], verbose=VERBOSE)
            results = history.history
            # Save history
            if results != {}:
                results['rx_loss'] = batch_tracking.batch_end_loss
                results['acc'] = batch_tracking.batch_end_acc
                # Save val_loss from model.fit() history under different name since it will be overwritten otherwise
                results['val_loss_history'] = results.pop('val_loss')
        if not test_mode:
            # Save model weights:
            try_save(model, pathfile_model, to_weights=use_weights)
            # Save training history to avoid data loss, if validation fails
            print('Save training history...')
            save_object.save(pathfile_results, results)
            print('Saved!')
        else:
            print('No model save in test mode.')
    else:
        # Load training history to include evaluation
        print('Load training history...')
        results = save_object.load(pathfile_results)
        if results is None:
            results = {}
        else:
            results = dict(results)
        print('Loaded!')

    # Save settings when training is done
    if not test_mode:
        SETTINGS_SAVED_FOLDER = os.path.join(path_script, subpath_results)  # 'settings_saved'
        saved_settings_path = os.path.join(path_script, SETTINGS_SAVED_FOLDER)
        with open(os.path.join(saved_settings_path, filename + '.yaml'), 'w', encoding='utf8') as written_file:
            yaml.safe_dump(params, written_file, default_flow_style=False)
        print('Settings saved!')
    else:
        print('No settings save in test mode.')

    # Evaluation/Validation of model
    evaluation_settings = params['evaluation']
    if snrs is None:
        snrs = mop.snr_range2snrlist(
            evaluation_settings['snr_range'], evaluation_settings['snr_step_size'])
    # Evaluation mode: (0) default: Validation for SNR range, (2) t-SNE embedding for visualization
    evaluation_mode = evaluation_settings.get('mode', 0)
    if evaluation_mode == 0:
        print('Evaluate model...')
        print(filename)
        # Evaluate model for different SNRs
        if transceiver_split == 1:
            # SINFONY/RL-SINFONY
            if rl >= 1:
                accuracy, loss = model_evaluation.evaluate_rlsinfony(model, test_input, test_labels,
                                                                     snrs=snrs, validation_rounds=evaluation_settings['validation_rounds'])
            else:
                accuracy, loss = model_evaluation.evaluate_sinfony(model, test_input, test_labels,
                                                                   snrs=snrs, validation_rounds=evaluation_settings['validation_rounds'])
        else:
            # Standard image recognition: Evaluate model accuracy once for test data
            if rl >= 1:
                _, _, loss_i, accuracy_i = model(
                    test_input, test_labels, sigma=keras.ops.convert_to_tensor([0, 0], dtype='float32'))
            else:
                accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                    model, test_input, test_labels)
            # Independent from SNR / constant, but plotted over SNR range
            loss = np.array(loss_i) * np.ones(snrs.shape)
            accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
            print(f'Validation Accuracy: {accuracy_i * 100.0:.3f}%')
        results['snr'] = snrs
        results['val_loss'] = loss
        results['val_acc'] = accuracy

    return results, pathfile_results, save_object


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # Default: 'mnist/semantic_config_mnist_sinfony'
    SETTINGS_FILE = 'mnist/semantic_config_mnist_rlsinfony'
    # Change from 'settings' to 'models' to reload simulations settings
    SETTINGS_FOLDER = 'settings'
    # Load the provided configuration file or the default one
    # python SINFONY.py semantic_config
    # Workaround for interactive sessions: Only allow config file names starting 'semantic_config'
    SETTINGS_FILE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][0:15].lower(
    ) == 'semantic_config' else SETTINGS_FILE
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)

    results, pathfile_results, save_object = simulation_sinfony(settings_path)

    # Show performance curve
    plt.figure(1)
    plt.semilogy(results['snr'], 1 - results['val_acc'])
    plt.xlabel('SNR')
    plt.ylabel(
        'semantic performance measure: classification error rate')
    plt.figure(2)
    plt.semilogy(results['snr'], results['val_loss'])
    plt.xlabel('SNR')
    plt.ylabel('crossentropy loss')

    # Check for existing evaluation and find unique filename
    pathfile_results, counter, accuracy_equal = find_unique_results_path(
        pathfile_results, save_object, results['val_acc'], results['snr'], tolerance=2e-2)

    # Save evaluation
    if counter == 1:
        print('Save evaluation...')
        save_object.save(pathfile_results, results)
        print('Evaluation saved.')
    else:
        print('Evaluation not saved as file is already available.')


# EOF
