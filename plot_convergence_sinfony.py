#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: beck
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# Own packages
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tplt
import os
import plot_results_sinfony as plot_sinfony


def load_curves_to_list(plot, path, x_axis='snr', y_axis='val_acc', error_mode=True, x_axis_normalization=0):
    '''Performance curve script modified: Load all curves of same kind into one list for statistical evaluation
    '''
    x_list = []
    y_list = []
    plt.figure(1)
    for _, table_entries in plot.items():
        # Iterate over all elements in chosen dictionary
        x_values, y_validation = plot_sinfony.load_result(table_entries, path, x_axis=x_axis,
                                                          y_axis=y_axis, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
        if x_values is not None and y_validation is not None:
            y_list.append(y_validation)
            x_list.append(x_values)
    return x_list, y_list


def convergence_analysis(y_list, plot, select_rl_loss=1, number_averaged_batches2=0, select_percentile=1, curve_area=0, extract_x_datapoints_logarithmic=0, x_starts_with_zero=False):
    '''Convergence Analysis: Calculate the average of performance curves and some statistical area around it
    '''
    number_epochs2 = 0
    if number_averaged_batches2 == 0:
        if 'params' in plot:
            # Dataset size (MNIST: 60000, CIFAR10: 50000)
            dataset_size = plot['params'][0]
            # Batch size (SGD: 64/128/512, Adam: 500)
            number_batches = plot['params'][1]
            # Calculate number of training iterations per epoch
            iterations_per_epoch = int(dataset_size / number_batches)
            # Set it to iterations_per_epoch for epoch resolution
            number_averaged_batches2 = iterations_per_epoch + \
                plot['params'][2]
            # MNIST: One epoch SGD: int(60000 / 64) = 937 (938 for AE!)
        else:
            # Set number_averaged_batches to RL default value of one alternating training step
            number_averaged_batches2 = 20

    # RL settings
    if 'rlparams' in plot:
        # Rx finetuning epochs increase factor: 1 + supervised Rx finetuning steps / alternating RL-SINFONY training steps  (default: 1.1)
        rx_finetuning_increment_factor = plot['rlparams'][0]
        # Number of succeeding Rx training steps before Tx training (default: 10)
        rl_rx_steps = plot['rlparams'][1]
        # Number of succeeding Tx training steps before Rx training (default: 10)
        rl_tx_steps = plot['rlparams'][2]
    else:
        rx_finetuning_increment_factor = 1.1
        rl_rx_steps = 10
        rl_tx_steps = 10
    # Total number of Rx/Tx alternating iterations steps for RL-SINFONY
    rl_alternating_steps = rl_rx_steps + rl_tx_steps

    if select_rl_loss == 1 or select_rl_loss == 2:
        # One Tx and Tx step are equal to one AE iteration -> in fact each training step only sees half of an epoch
        # Double the number of averaging iterations for single Tx and Rx steps,
        # since then number of seen data points (on average) is equal to AE setting
        # Exception: Rx finetuning -> Only Rx steps through entire epoch
        number_averaged_batches2 = 2 * number_averaged_batches2

    yl = np.array(y_list)
    y2 = np.transpose(np.reshape(
        yl, (yl.shape[0], -1, number_averaged_batches2)), axes=(1, 0, 2))

    if select_rl_loss != 0:
        number_epochs = int(np.float32(
            y2.shape[0] / rx_finetuning_increment_factor))
        number_epochs_rx = y2.shape[0] - number_epochs
    else:
        number_epochs = y2.shape[0]

    # Separation of Rx and Tx alternating training steps
    if select_rl_loss == 1 or select_rl_loss == 2:
        y2_rl = y2[:number_epochs, ...]
        y2_rl2 = y2_rl[..., :int(y2_rl.shape[-1] / rl_alternating_steps) *
                       rl_alternating_steps].reshape((y2_rl.shape[0], y2_rl.shape[1], -1, 10))
    if select_rl_loss == 1:
        # Extract Rx training steps
        y3 = y2_rl2[:, :, 0::2, :].reshape(
            (y2_rl2.shape[0], y2_rl2.shape[1], -1))
        y3_cut = y2_rl[..., int(y2_rl.shape[-1] / rl_alternating_steps)
                       * rl_alternating_steps:][..., :rl_rx_steps]
        y3 = np.concatenate([y3, y3_cut], axis=-1)
    elif select_rl_loss == 2:
        # Extract Tx training steps
        y3 = y2_rl2[:, :, 1::2, :].reshape(
            (y2_rl2.shape[0], y2_rl2.shape[1], -1))
        y3_cut = y2_rl[..., int(y2_rl.shape[-1] / rl_alternating_steps)
                       * rl_alternating_steps:][..., rl_rx_steps:]
        y3 = np.concatenate([y3, y3_cut], axis=-1)
    elif select_rl_loss == 3:
        # Exctract Rx finetuning steps
        y3 = y2[number_epochs:, ...]
        number_epochs2 = number_epochs / 2  # Alternating training steps number / 2
        number_epochs = number_epochs_rx
    else:
        y3 = y2

    y_mean = np.mean(np.mean(y3, axis=-1), axis=-1)

    if select_percentile == 1:
        # Calculate percentile across curves/runs
        y_area = np.percentile(np.mean(y3, axis=-1),
                               100 - curve_area, axis=-1)
        y_area2 = np.percentile(
            np.mean(y3, axis=-1), 0 + curve_area, axis=-1)
    else:
        # Calculate standard deviation across curves/runs
        y_area = np.std(np.mean(y3, axis=-1), axis=-1)
        y_area2 = y_area
        y_area = y_mean + curve_area * y_area
        y_area2 = y_mean - curve_area * y_area2

    # Select x axis data points
    if extract_x_datapoints_logarithmic == 1:
        # Extract data points on logarithmic scale (for data compression with many curve data points)
        log_index = (np.arange(1, 10)[
            np.newaxis] * 10 ** np.arange(0, int(np.log10(number_epochs)))[:, np.newaxis]).flatten()
        x2 = np.concatenate([log_index, np.arange(
            1, int(number_epochs / (10 ** int(np.log10(number_epochs)))) + 1) * 10 ** int(np.log10(number_epochs))])
        y_mean = y_mean[x2 - 1]
        y_area = y_area[x2 - 1]
        y_area2 = y_area2[x2 - 1]
    else:
        x2 = np.arange(y_mean.shape[0]) + (x_starts_with_zero is False)

    if select_rl_loss == 3:
        # Shift x axis for Rx finetuning plot
        x3 = x2 + number_epochs2
    else:
        x3 = x2
    return x3, y_mean, y_area, y_area2


def plot_results_semcom_convergence(selected_plots, x_axis='snr', y_axis='val_acc', datapath='models',
                                    error_mode=True, x_axis_normalization=0, logplot=True, plot_same=False,
                                    select_rl_loss=1, number_averaged_batches=0, select_percentile=1,
                                    curve_area=0, extract_x_datapoints_logarithmic=0, x_starts_with_zero=False):
    '''Statistical evaluation of multiple simulation runs provided by paper_lots
    Convergence Analysis: Calculate the average of performance curves and some statistical area around it
    '''
    # Plot all selected curves
    plot_index = 0
    figures = []

    for plot in selected_plots:
        path = datapath
        if 'title' in plot:
            subpath = plot['title'][1]
            if isinstance(subpath, str) and subpath != '' and subpath is not None:
                path = os.path.join(datapath, subpath)
            if isinstance(plot['title'][2], int) and not isinstance(plot['title'][2], bool) and x_axis_normalization != 0:
                x_axis_normalization = plot['title'][2]
        if plot_same is True:
            figure = plt.figure(1)
        _, y_list = load_curves_to_list(plot, path, x_axis=x_axis, y_axis=y_axis,
                                        error_mode=error_mode, x_axis_normalization=x_axis_normalization)

        # Convergence Analysis: Calculate the average of performance curves and some statistical area around it
        x3, y_mean, y_area, y_area2 = convergence_analysis(
            y_list, plot, select_rl_loss=select_rl_loss, number_averaged_batches2=number_averaged_batches, select_percentile=select_percentile, curve_area=curve_area, extract_x_datapoints_logarithmic=extract_x_datapoints_logarithmic, x_starts_with_zero=x_starts_with_zero)

        # Save curves in dat-file
        pathfile = 'plots/SINFONY_convergence' + str(plot_index)
        np.savetxt(pathfile + '.dat',
                   np.c_[x3, y_mean, y_area2, y_area],
                   fmt=['%d', '%.18e', '%.18e', '%.18e'],
                   # header = 'A B C D',
                   )
        print('Saved training curves to "' + pathfile + '.dat".')

        # Plot the average curve with two statistic curves
        if plot_same is False:
            figure = plt.figure(plot_index)
        if logplot is True:
            plt.loglog(x3, y_area, color='r')
            plt.loglog(x3, y_area2, color='r')
        else:
            plt.plot(x3, y_area, color='r')
            plt.plot(x3, y_area2, color='r')
        plt.loglog(x3, y_mean)
        # Add shaded area with percentiles or standard deviations
        # plt.fill_between(x3, y_area2, y_area, color = 'r')

        # Plot settings
        # title = list(plot.keys())[-1][:-1:]
        if 'title' in plot:
            title = plot['title'][0]
        else:
            title = None
        plot_sinfony.set_plot_settings(
            x_axis, y_axis, error_mode=error_mode, x_axis_normalization=x_axis_normalization, title=title)
        plt.show()

        # Save plot with tikzplotlib
        plot_sinfony.tikzplotlib_fix_ncols(figure)
        tplt.save(pathfile + '.tikz')
        plot_index = plot_index + 1
        figures.append(figure)
    return figures


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Settings
    y_axis = 'rx_loss'		# val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
    error_mode = True		# Show classification error instead of accuracy
    # (0) w/o SNR normalization, (56 or 64) SNR normalization by [number of channel uses/number of features 56 or 64]
    x_axis_normalization = 0
    x_axis = 'default'		# (snr) snr value on x axis, (default) index on x axis
    logplot = True			# Logarithmic plot?
    select_plot = False     # Select one plot or plot all preselected plots
    # Fixed
    datapath = 'models'
    filename_prefix = 'RES_'
    dn = filename_prefix

    # Convergence Analysis Settings
    plot_same = False						# Plot all curves into one plot?
    # Number of averaged batches, one alternating training iteration: 20 (10 iterations for AE), (0: default, average over on epoch)
    number_averaged_batches = 0
    # Select alternating training iterations for reinforcement learning:
    select_rl_loss = 1
    # (0, default) no selection, (1) rx steps rx_loss, (2) tx steps rx_loss, (3) rx finetuning steps
    # Use percentile (1) or standard deviation (0) area
    select_percentile = 1
    # Lower percentile, higher is 1 - curve_area (select_percentile = 1), multiple of standard deviations (select_percentile = 0)
    curve_area = 0
    # Extract data points on logarithmic scale (for data compression with many curve data points)
    extract_x_datapoints_logarithmic = 0
    # We have no initial evaluation of the model before training: 0 (default)
    x_starts_with_zero = False

    # Plot tables

    # SINFONY

    # Explaining the abbreviations/numbers:
    # CIFAR1:       Train and evaluate central classififier based on full image information on CIFAR10 [Central]
    # CIFAR2:       Train without Tx and Rx module and no noise [SINFONY - perfect comm.]
    # CIFAR2 snr:   Like CIFAR2, but evaluation with noise [SINFONY - AWGN]
    # CIFAR3:       Train without Tx and Rx module, but with noise [SINFONY - AWGN + training]
    # CIFAR4:       NTx = NFeat / 4 [SINFONY - Tx/Rx (NTx < NFeat)]
    # CIFAR5:       NTx = NFeat / 4 + no channel
    # CIFAR6:       NTx = NFeat [SINFONY - Tx/Rx (NTx = NFeat)]
    # CIFAR7:       NTx = NFeat + No channel
    # Same holds for MNIST dataset with MNIST1, etc.

    selected_plots = []

    mnist_conv = {
        # Options: ['title', subpath, x_axis_normalization, False],
        'title': ['SINFONY: MNIST4 sgdlr conv', 'mnist', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 64, 1, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 sgdlr conv0': [dn + 'ResNet14_MNIST4_sgdlr_conv0', 'g-', 14, True],
        'MNIST4 sgdlr conv1': [dn + 'ResNet14_MNIST4_sgdlr_conv1', 'g-', 14, True],
        'MNIST4 sgdlr conv2': [dn + 'ResNet14_MNIST4_sgdlr_conv2', 'g-', 14, True],
        'MNIST4 sgdlr conv3': [dn + 'ResNet14_MNIST4_sgdlr_conv3', 'g-', 14, True],
        'MNIST4 sgdlr conv4': [dn + 'ResNet14_MNIST4_sgdlr_conv4', 'g-', 14, True],
        'MNIST4 sgdlr conv5': [dn + 'ResNet14_MNIST4_sgdlr_conv5', 'g-', 14, True],
        'MNIST4 sgdlr conv6': [dn + 'ResNet14_MNIST4_sgdlr_conv6', 'g-', 14, True],
        'MNIST4 sgdlr conv7': [dn + 'ResNet14_MNIST4_sgdlr_conv7', 'g-', 14, True],
        'MNIST4 sgdlr conv8': [dn + 'ResNet14_MNIST4_sgdlr_conv8', 'g-', 14, True],
        'MNIST4 sgdlr conv9': [dn + 'ResNet14_MNIST4_sgdlr_conv9', 'g-', 14, True],
    }
    # selected_plots.append(mnist_conv)

    mnist_conv2 = {
        'title': ['SINFONY: MNIST4 sgdlr snr-4 6 conv', 'mnist', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 64, 1, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 sgdlr snr-4 6 conv0': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv0', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv1': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv1', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv2': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv2', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv3': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv3', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv4': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv4', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv5': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv5', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv6': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv6', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv7': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv7', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv8': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv8', 'g-', 14, True],
        'MNIST4 sgdlr snr-4 6 conv9': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv9', 'g-', 14, True],
    }
    selected_plots.append(mnist_conv2)

    mnist_adam_conv = {
        'title': ['SINFONY: MNIST4 adam conv', 'mnist', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 500, 0, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 adam conv0': [dn + 'ResNet14_MNIST4_adam_Ne100_conv0', 'g-', 14, True],
        'MNIST4 adam conv1': [dn + 'ResNet14_MNIST4_adam_Ne100_conv1', 'g-', 14, True],
        'MNIST4 adam conv2': [dn + 'ResNet14_MNIST4_adam_Ne100_conv2', 'g-', 14, True],
        'MNIST4 adam conv3': [dn + 'ResNet14_MNIST4_adam_Ne100_conv3', 'g-', 14, True],
        'MNIST4 adam conv4': [dn + 'ResNet14_MNIST4_adam_Ne100_conv4', 'g-', 14, True],
        'MNIST4 adam conv5': [dn + 'ResNet14_MNIST4_adam_Ne100_conv5', 'g-', 14, True],
        'MNIST4 adam conv6': [dn + 'ResNet14_MNIST4_adam_Ne100_conv6', 'g-', 14, True],
        'MNIST4 adam conv7': [dn + 'ResNet14_MNIST4_adam_Ne100_conv7', 'g-', 14, True],
        'MNIST4 adam conv8': [dn + 'ResNet14_MNIST4_adam_Ne100_conv8', 'g-', 14, True],
        'MNIST4 adam conv9': [dn + 'ResNet14_MNIST4_adam_Ne100_conv9', 'g-', 14, True],
    }
    # selected_plots.append(mnist_adam_conv)

    mnist_adam_conv2 = {
        'title': ['SINFONY: MNIST4 adam snr-4 6 conv', 'mnist', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 500, 0, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 adam snr-4 6 conv0': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv0', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv1': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv1', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv2': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv2', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv3': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv3', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv4': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv4', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv5': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv5', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv6', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv7': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv7', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv8': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv8', 'g-', 14, True],
        'MNIST4 adam snr-4 6 conv9': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv9', 'g-', 14, True],
    }
    selected_plots.append(mnist_adam_conv2)

    mnist_rl = {
        'title': ['RL-SINFONY: MNIST4 ntx14 RL Ne3000 SGD lr1e-3', 'mnist_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 64, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_2', 'k-', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_3', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 4': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_4', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 5': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_5', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_6', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 7': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_7', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 8': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_8', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 9': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_9', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 10': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_10', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 11': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_11', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3', 'k--x', 14, True],
        # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3_2', 'k-x', 14, True],
    }
    # selected_plots.append(mnist_rl)

    mnist_rl2 = {
        'title': ['RL-SINFONY: MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6', 'mnist_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 64, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_snr-4_6', 'k-o', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_2', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_3', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 4': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_4', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 5': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_5', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_6', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 7': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_7', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 8': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_8', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 9': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_9', 'k--', 14, True],
        'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 10': [dn + 'ResNet14_MNIST4_sgd_Ne3000_snr-4_6_conv10', 'k--', 14, True],
        # Test
        # 'MNIST4 ntx14 RL Ne3000 SGD snr-4_6 test (w/o AE + RL call tf.function)': [dn + 'ResNet14_MNIST4_RL_snr-4_6_test', 'b-', 14, True],
        # 'MNIST4 ntx14 RL Ne3000 SGD snr-4_6 test2 (w/o AE tf.function)': [dn + 'ResNet14_MNIST4_RL_snr-4_6_test2', 'b--', 14, True],
    }
    selected_plots.append(mnist_rl2)

    mnist_rl_adam = {
        'title': ['RL-SINFONY: MNIST4 ntx14 RL Ne2000 Adam', 'mnist_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 500, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 ntx14 RL Ne2000 Adam np 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_np', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 1': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv1', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 2': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv2', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 3': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv3', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 4': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv4', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 5': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv5', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv6', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 7': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv7', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 8': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv8', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 9': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv9', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne2000 Adam 10': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv10', 'm-', 14, True],
    }
    # selected_plots.append(mnist_rl_adam)

    mnist_rl_adam2 = {
        'title': ['RL-SINFONY: MNIST4 ntx14 RL Ne3000 Adam snr-4 6', 'mnist_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 500, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv0', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 1': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv1', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 2': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv2', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 3': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv3', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 4': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv4', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 5': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv5', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv6', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 7': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv7', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 8': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv8', 'm-', 14, True],
        'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 9': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv9', 'm-', 14, True],
    }
    # selected_plots.append(mnist_rl_adam2)

    mnist_rl_adam2_Ne = {
        'title': ['RL-SINFONY: MNIST4 ntx14 RL Ne6000 Adam snr-4 6', 'mnist_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [60000, 500, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        # 'MNIST4 ntx14 RL Ne4000 Adam snr-4 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne4000_snr-4_6_conv0', 'b--x', 14, True],
        # 'MNIST4 ntx14 RL Ne5000 Adam snr-4 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne5000_snr-4_6_conv0', 'b--x', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv0', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 1': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv1', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 2': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv2', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 3': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv3', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 4': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv4', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 5': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv5', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv6', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 7': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv7', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 8': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv8', 'b-', 14, True],
        'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 9': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv9', 'b-', 14, True],
        # 'MNIST4 ntx14 RL Ne8000 Adam snr-4 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne8000_snr-4_6_conv0', 'b--x', 14, True],
        # 'MNIST4 RL adam Ne3000 pstd=0.15 snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_pstd15_snr-4_6_conv0', 'm-', 16, True],
        # 'MNIST4 RL adam Ne4000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne4000_pstds_snr-4_6_conv0', 'm-', 16, True],
        # 'MNIST4 RL adam Ne6000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_pstds_snr-4_6_conv0', 'm-', 16, True],
        # 'MNIST4 RL adam Ne10000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne10000_pstds_snr-4_6_conv0', 'r-', 16, True],
    }
    selected_plots.append(mnist_rl_adam2_Ne)

    cifar_conv = {
        'title': ['SINFONY: MNIST4 ntx14 RL Ne6000 Adam snr-4 6', 'cifar10', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [50000, 64, 1, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'CIFAR4 sgdlr[1e-1,1e-2,1e-3][100,150] Ne200 snr-4 6 conv0': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv0', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv1': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv1', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv2': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv2', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv3': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv3', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv4': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv4', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv5': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv5', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv6': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv6', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv7': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv7', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv8': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv8', 'm-', 16, True],
        'CIFAR4 sgdlr snr-4 6 conv9': [dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv9', 'm-', 16, True],
    }
    selected_plots.append(cifar_conv)

    cifar_adam_conv = {
        'title': ['SINFONY: CIFAR4 adam Nb500 lr1e-4 Ne200 snr-4 6 conv', 'cifar10', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [50000, 500, 0, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        # 'CIFAR4 adam Nb500 lr1e-3 Ne200 snr-4 6 conv0': [dn + 'ResNet20_CIFAR4_adam_Ne200_snr-4_6_conv0', 'g-', 16, True],
        'CIFAR4 adam Nb500 lr1e-4 Ne200 snr-4 6 conv0': [dn + 'ResNet20_CIFAR4_adam_lr1e-4_Ne200_snr-4_6_conv0', 'g--x', 16, True],
    }
    selected_plots.append(cifar_adam_conv)

    cifar_rl = {
        'title': ['RL-SINFONY: CIFAR4 RL sgd Nb128 lr1e-4 Ne100000 snr-4 6', 'cifar10_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [50000, 128, 0, False],
        # 'params': [50000, 512, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        # 'CIFAR4 RL sgd Nb64 lr1e-3 Ne1000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgd_lr1e-3_Ne1000_snr-4_6_0', 'b--<', 16, True],
        # 'CIFAR4 RL sgd Ne3000 pstd 0.15 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgd_Ne3000_pstd15_snr-4_6_conv0', 'k--x', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-3 Ne10000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-3_Ne10000_snr-4_6_conv0', 'm--', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne10000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne10000_snr-4_6_conv0', 'm-->', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-5 Ne10000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-5_Ne10000_snr-4_6_conv0', 'm--<', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-6 Ne10000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-6_Ne10000_snr-4_6_conv0', 'm--x', 16, True],
        # 'CIFAR4 RL sgd Nb512 lrs0 Ne100000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lrs0_Ne100000_snr-4_6_conv0', 'm-->', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne100000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne100000_snr-4_6_conv0', 'm-->', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne50000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne50000_snr-4_6_conv0', 'm-->', 16, True],
        # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne200000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne200000_snr-4_6_conv0', 'm-->', 16, True],
        # Last simulations
        # 'CIFAR4 RL sgd Nb64 lr1e-4 Ne100000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb64_lr1e-4_Ne100000_snr-4_6_conv0', 'k--<', 16, True],
        'CIFAR4 RL sgd Nb128 lr1e-4 Ne100000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgdNb128_lr1e-4_Ne100000_snr-4_6_conv0', 'k-->', 16, True],
    }
    selected_plots.append(cifar_rl)

    cifar_rl_adam = {
        'title': ['RL-SINFONY: CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6', 'cifar10_rl', 0, False],
        # 'Tag': ['Data set size', 'batch size', 'AE mode', 'subpath', 'on/off'],
        'params': [50000, 500, 0, False],
        # 'Tag': ['rx_finetuning_increment_factor', 'rl_rx_steps', 'rl_tx_steps', 'on/off'],
        'rlparams': [1.1, 10, 10, False],
        # 'Tag': ['Data name', 'Color in plot', 'Channel uses', 'on/off'],
        'CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_adam_lr1e-4_Ne100000_snr-4_6_conv0', 'g--<', 16, True],
        'CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6 1': [dn + 'ResNet20_CIFAR4_RL_adam_lr1e-4_Ne100000_snr-4_6_conv1', 'g--<', 16, True],
    }
    selected_plots.append(cifar_rl_adam)

    # MNIST
    # RL-Paper (Adam): mnist_adam_conv2, mnist_rl_adam2_Ne
    # SGD: mnist_conv2, mnist_rl2
    # SNR [6,16]: Labels without SNR range
    # CIFAR (only single runs since computation time high)
    # SINFONY: cifar_conv -> 'CIFAR4 sgdlr snr-4 6 conv5', 'CIFAR4 adam Nb500 Ne200 snr-4 6 conv0'
    # RL-SINFONY: cifar_rl -> 'CIFAR4 RL sgd Nb512 lr1e-4 Ne100000 snr-4 6 0'

    # Set here one dictionary to be analyzed
    if select_plot is True:
        selected_plots = [cifar_rl]

    figures = plot_results_semcom_convergence(selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath, error_mode=error_mode, x_axis_normalization=x_axis_normalization, logplot=logplot,
                                              plot_same=plot_same, select_rl_loss=select_rl_loss, number_averaged_batches=number_averaged_batches, select_percentile=select_percentile, curve_area=curve_area, extract_x_datapoints_logarithmic=extract_x_datapoints_logarithmic, x_starts_with_zero=x_starts_with_zero)
