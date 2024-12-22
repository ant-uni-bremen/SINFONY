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
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

# Tensorflow 2 packages
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam  # , Nadam

# Own packages
import datasets
import model_evaluation
import resnet
import resnet_sinfony
import resnet_rl_sinfony
import my_math_operations as mop
from my_functions import savemodule
import my_training as mt
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import my_training as mf

# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def compute_confusion_matrix(true_labels, pred_labels, num_classes):
    """
    Compute the confusion matrix manually.

    Parameters:
    - true_labels: List or array of true class labels.
    - pred_labels: List or array of predicted class labels.
    - num_classes: Number of classes in the labels.

    Returns:
    - Confusion matrix as a 2D numpy array.
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1
    return conf_matrix

# aprob = model(test_input_normalized)
# num_classes = 5
# true_labels= np.argmax(test_labels, axis=1)
# pred_labels= np.argmax(aprob, axis=1)
# compute_confusion_matrix(true_labels, pred_labels, num_classes)


def visualize_tsne_embedding(visualized_model, snr_evaluation_test, test_input, test_labels, visualized_dataset):
    '''Visualize t-SNE embedding
    '''
    # TODO: Script so far only works with SINFONY models
    cmap = plt.cm.jet					# define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist,
                          cmap.N)       # create the new map
    # x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    x1 = visualized_model.layers[1](test_input)		# Output of Tx layer
    sigma_tsne = mop.snr2standard_deviation(snr_evaluation_test)
    sigma_test_tsne = np.array([sigma_tsne, sigma_tsne])
    visualized_model.layers[2].set_weights([sigma_test_tsne])
    x2 = visualized_model.layers[2](x1)			# Channel
    x3 = visualized_model.layers[3].layers[0](x2)  # Input layer
    x4 = visualized_model.layers[3].layers[1](x3)  # Rx layer
    x5 = visualized_model.layers[3].layers[2](
        x4)  # Global average pooling layer
    # t-SNE
    # Choose output to cluster and visualize
    # x = x4[:, 0, 0, :]				# Output of Rx layer
    x = x5								# Output after Global average pooling layer, just before softmax layer
    x_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random').fit_transform(x)
    # Plot
    plt.figure(1)
    labels_number = np.argmax(test_labels, axis=-1)
    # Estimated labels
    # labels_number_est = np.argmax(model.predict(testnorm), axis = -1)
    # labels_number = labels_number_est
    plt.scatter(x_embedded[:, 0],
                x_embedded[:, 1], c=labels_number, cmap=cmap)
    custom_lines = []
    for cl in np.unique(labels_number):
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmaplist[int(
            cl * (len(cmaplist) - 1) / (test_labels.shape[-1] - 1))]))
    if visualized_dataset == 'cifar10':
        dlabel = ['0: Airplane', '1: Automobile', '2: Bird', '3: Cat',
                  '4: Deer', '5: Dog', '6: Frog', '7: Horse', '8: Ship', '9: Truck']
    elif visualized_dataset == 'mnist':
        dlabel = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        dlabel = []
        print('No labels available!')
    plt.legend(custom_lines, dlabel, loc='center left',
               bbox_to_anchor=(1, 0.5))


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # Default: 'mnist/semantic_config_mnist_sinfony.yaml'
    SETTINGS_FILE = 'mnist/semantic_config_mnist_sinfony.yaml'
    # Avoid error messages
    import logging
    tf.get_logger().setLevel(logging.ERROR)
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
    tf.keras.backend.clear_session()          	                    # Clearing graphs
    tf.keras.backend.set_floatx(load_settings['numerical_precision'])
    # Random seed in every run, predictable random numbers for debugging with np.random.seed(0)
    np.random.seed()

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
    # TODO: l2 regularization only for ResNet feature extractor? Yes, better performance
    if model_settings['resnet']['weight_decay'] == 0:
        weight_decay = None
    else:
        weight_decay = tf.keras.regularizers.l2(
            model_settings['resnet']['weight_decay'])
    if model_settings['communication']['weight_decay'] == 0:
        weight_decay_communication = None
    else:
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
    valX = mt.create_batch(test_input_normalized, validation_dataset_size, 0)
    # Summarize loaded dataset
    if show_dataset is True:
        datasets.summarize_dataset(
            train_input, train_labels, test_input, test_labels)

    # Create/load model
    resnet_layer_number = resnet.calculate_resnet_layer_number(
        number_resnet_blocks, number_residual_units, model_settings['resnet']['bottleneck'])
    print('ResNet', resnet_layer_number, ' chosen')

    if load is False:
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
    else:
        # Load existing model:
        print('Loading model...')
        if rl >= 1 and transceiver_split == 1:
            # Load RL-SINFONY via weights
            if rl == 1:
                model = resnet_rl_sinfony.ResnetRLSinfony(
                    resnet_config=resnet_config, communication_config=communication_config)
            elif rl == 2:
                model = resnet_rl_sinfony.ResnetAE2(
                    resnet_config=resnet_config, communication_config=communication_config)
            model.load_weights(os.path.join(
                path_script, pathfile, filename))
        else:
            # Load SINFONY model
            model = tf.keras.models.load_model(pathfile)
        print('Model loaded.')

    if rl == 0:
        # Summarize AE-based SINFONY
        model.summary()

    # Compile and train model
    if load is False:
        if rl >= 1:
            # RL-SINFONY
            sigma_train = tf.constant(sigma_train, dtype='float32')
            results = resnet_rl_sinfony.rl_based_training(model, train_input_normalized, train_labels, optimizer, optimizer_tx, optimizer_rx2, validation_input=valX,
                                                          validation_labels=valY, epochs=number_epochs, training_batch_size=batch_size, sigma=sigma_train, stochastic_policy_gradient_config=spg_config)
            # Save model weights:
            print('Saving model weights...')
            model.save_weights(os.path.join(
                path_script, pathfile, filename))
            print('Model weigths saved.')
            # Save training history to avoid data loss, if validation fails
            print('Save training history...')
        else:
            # SINFONY AE-like
            # Note: For loading, compile is not necessary: optimizer, loss and metric are saved with model
            model.compile(
                optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(train_input_normalized, train_labels, epochs=number_epochs, batch_size=batch_size, validation_data=(
                valX, valY), callbacks=[batch_tracking, model_checkpoint, early_stopping], verbose=VERBOSE)
            results = history.history
            # Save Keras model:
            print('Saving model...')
            model.save(pathfile)
            print('Model saved.')
            # Save history
            print('Save training history...')
            if results != {}:
                results['rx_loss'] = batch_tracking.batch_end_loss
                results['acc'] = batch_tracking.batch_end_acc
                # Save val_loss from model.fit() history under different name since it will be overwritten otherwise
                results['val_loss_history'] = results.pop('val_loss')
        saveobj.save(pathfile2, results)
        print('Saved!')
    else:
        # Load training history to include evaluation
        print('Load training history...')
        results = saveobj.load(pathfile2)
        if results is None:
            results = {}
        else:
            results = dict(results)
        print('Loaded!')

    # Save settings when training is done
    SETTINGS_SAVED_FOLDER = 'settings_saved'
    saved_settings_path = os.path.join(path_script, SETTINGS_SAVED_FOLDER)
    with open(os.path.join(saved_settings_path, filename + '.yaml'), 'w', encoding='utf8') as written_file:
        yaml.safe_dump(params, written_file, default_flow_style=False)
    print('Settings saved!')

    # Evaluation of model
    snrs = mop.snr_range2snrlist(snr_range, snr_step_size)
    if evaluation_mode == 0:
        print('Evaluate model...')
        print(filename)
        # Evaluate model for different SNRs
        if transceiver_split == 1:
            # SINFONY/RL-SINFONY
            if rl >= 1:
                accuracy, loss = model_evaluation.evaluate_rlsinfony(model, test_input_normalized, test_labels,
                                                                     snrs=snrs, validation_rounds=validation_rounds)
            else:
                accuracy, loss = model_evaluation.evaluate_sinfony(model, test_input_normalized, test_labels,
                                                                   snrs=snrs, validation_rounds=validation_rounds)
        else:
            # Standard image recognition: Evaluate model accuracy once for test data
            if rl >= 1:
                _, _, loss_i, accuracy_i = model(
                    test_input_normalized, test_labels, sigma=tf.constant([0, 0], dtype='float32'))
            else:
                accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                    model, test_input_normalized, test_labels)
            # Independent from SNR / constant, but plotted over SNR range
            loss = np.array(loss_i) * np.ones(snrs.shape)
            accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
            print(f'> {accuracy_i * 100.0:.3f}')
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
    elif evaluation_mode == 2:
        # t-SNE embedding for visualization
        if rl == 0:
            visualize_tsne_embedding(
                model, snr_evaluation, test_input_normalized, test_labels, dataset)

# EOF
