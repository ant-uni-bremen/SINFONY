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
from my_functions import savemodule
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tplt
import os


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


ACCURACY_MEASURES = ['val_acc', 'accuracy', 'val_accuracy',
                     'acc', 'acc_val']  # Accuracy measures


def load_result(table_entries, path, x_axis='snr', y_axis='val_acc', error_mode=True, x_axis_normalization=0):
    '''Load one element result in chosen dictionary
    '''
    x = None
    y_validation = None
    load = savemodule()
    if table_entries[-1]:
        # If curve activated in plot, load data
        pathfile = os.path.join(path, table_entries[0])
        load_file = load.load(pathfile, form='npz')
        if load_file is not None:
            # If data existent, then proceed
            results = load_file
            if y_axis in results:
                # If chosen performance measure existent, then proceed
                if x_axis in results:
                    if results[x_axis].shape != results[y_axis].shape:
                        y_validation = results[y_axis].repeat(
                            results[x_axis].shape[0])
                    else:
                        y_validation = results[y_axis]
                    # SNR normalization by number of channel uses
                    if x_axis_normalization == 0 or table_entries[-2] == 0:
                        # No normalization if x_axis_normalization or table_entries[-2] set to 0
                        x = results[x_axis]
                    else:
                        x = results[x_axis] + 10 * \
                            np.log10(table_entries[-2] /
                                     x_axis_normalization)
                else:
                    # If chosen x axis does not exist, just plot the curve as a function of index
                    y_validation = results[y_axis]
                    if y_validation.shape == ():
                        x = np.arange(1)
                    else:
                        x = np.arange(y_validation.shape[0])
                if error_mode is True and y_axis in ACCURACY_MEASURES:
                    # Choose classification error rate instead of accuracy
                    y_validation = 1 - y_validation
    return x, y_validation


def plot_results_semcom(selected_plots, x_axis='snr', y_axis='val_acc', datapath='models', logplot=True, error_mode=True, x_axis_normalization=0):
    '''Plot results from dictionary ml_methods
    '''
    # Plot all selected curves
    plot_index = 0
    figures = []
    for plot in selected_plots:
        # Performance curves
        path = datapath
        if 'title' in plot:
            subpath = plot['title'][1]
            if isinstance(subpath, str) and subpath != '' and subpath is not None:
                path = os.path.join(datapath, subpath)
            if isinstance(plot['title'][2], int) and not isinstance(plot['title'][2], bool) and x_axis_normalization != 0:
                x_axis_normalization = plot['title'][2]
        figure = plt.figure(plot_index)
        for table_key, table_entries in plot.items():
            # Iterate over all elements in chosen dictionary
            x, y_validation = load_result(table_entries, path, x_axis=x_axis, y_axis=y_axis,
                                          error_mode=error_mode, x_axis_normalization=x_axis_normalization)
            if x is not None and y_validation is not None:
                if logplot is True:
                    # Use absolute value if logarithmic plot
                    plt.semilogy(x, np.abs(y_validation),
                                 table_entries[1], label=table_key)
                else:
                    # Linear plot
                    plt.plot(x, y_validation,
                             table_entries[1], label=table_key)

        # Plot Settings
        if 'title' in plot:
            title = plot['title'][0]
        else:
            title = None
        set_plot_settings(x_axis, y_axis, error_mode=error_mode,
                          x_axis_normalization=x_axis_normalization, title=title)

        # Save curves with tikzplotlib
        pathfile = "plots/SINFONY" + str(plot_index) + ".tikz"
        tikzplotlib_fix_ncols(figure)
        tplt.save(pathfile)
        print('Saved performance curves to "' + pathfile + '".')

        plot_index = plot_index + 1
        figures.append(figure)
    return figures


def set_plot_settings(x_axis, y_axis, error_mode=False, x_axis_normalization=0, title=None):
    '''Set default plot settings
    '''
    # Plot Settings
    # plt.ylim(10 ** -3, 1)
    # plt.xlim(10, 20)
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor',
             color='#999999', linestyle='-', alpha=0.3)
    if x_axis_normalization == 0:
        plt.xlabel(x_axis)
    else:
        plt.xlabel(x_axis + ' normalized by number of channel uses')
    if error_mode is True and y_axis in ACCURACY_MEASURES:
        plt.ylabel(y_axis + ': error rate')
    else:
        plt.ylabel(y_axis)
    # if 'title' in plot:
    #     plt.title(plot['title'][0])
    if title is not None:
        plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Settings
    y_axis = 'val_acc'   	# val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
    error_mode = True       # Show classification error instead of accuracy
    # (0) w/o SNR normalization, (56 or 64) SNR normalization by [number of channel uses/number of features 56 or 64]
    x_axis_normalization = 1
    # (snr) snr value on x axis, (default) index on x axis
    x_axis = 'snr'
    logplot = True          # Logarithmic plot?
    select_plot = False     # Select one plot or plot all preselected plots
    # Fixed
    datapath = 'models'
    filename_prefix = 'RES_'
    dn = filename_prefix

    # Plot tables

    selected_plots = []

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

    # The different SINFONY designs are investigated to distinguish the influences of the layers (including noise layer)
    # We omitted CIFAR5 and CIFAR7 in the article due to limited space and contribution -> similar to CIFAR2
    # Default SNR training range is here [6, 16] as it was the first guess, default in journal is [-4, 6] as it leads to better results
    # ntx = Ntx
    # nrx = Nw -> Layer width in Rx module

    # Investigations on SINFONY design

    # [Published]
    cifar = {
        # Options: ['title', subpath, x_axis_normalization, False],
        'title': ['SINFONY design: CIFAR', 'cifar10', 64, False],
        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
        'CIFAR1': [dn + 'ResNet20_CIFAR', 'k--', 0, True],
        # 'CIFAR1 test': [dn + 'ResNet20_CIFAR_test', 'k--', 0, True],
        'CIFAR2': [dn + 'ResNet20_CIFAR2_nosnr', 'b--', 0, True],
        'CIFAR2 snr': [dn + 'ResNet20_CIFAR2', 'b-', 64, True],
        # 'CIFAR3 snr6 16': [dn + 'ResNet20_CIFAR3', 'b--x', 64, True],
        'CIFAR3 snr-4 6': [dn + 'ResNet20_CIFAR3_snr-4_6', 'b-x', 64, True],
        # 'CIFAR4 ntx16 nrx64 snr6 16': [dn + 'ResNet20_CIFAR4', 'g--D', 16, True],
        'CIFAR4 ntx16 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR4_snr-4_6', 'g-D', 16, True],
        'CIFAR4 ntx16 nrx64 rx individual snr-4 6': [dn + 'ResNet20_CIFAR4_rx_snr-4_6', 'g--x', 16, True],
        'CIFAR4 ntx16 nrx64 rx individual snr-4 6 2': [dn + 'ResNet20_CIFAR4_rx_snr-4_6_2', 'g--', 16, True],
        # 'CIFAR5 ntx16': [dn + 'ResNet20_CIFAR5', 'g--', 0, True],
        # 'CIFAR6 ntx64 nrx64 snr6 16': [dn + 'ResNet20_CIFAR6', 'r--s', 64, True],
        'CIFAR6 ntx64 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR6_snr-4_6', 'r-s', 64, True],
        # 'CIFAR7 ntx64': [dn + 'ResNet20_CIFAR7', 'r--', 0, True],
    }
    selected_plots.append(cifar)

    # [Published]
    mnist = {'title': ['SINFONY design: MNIST', 'mnist', 56, False],
             # 'Tag': ['data name', 'color in plot', channel uses, on/off],
             # 'MNIST1 Ne10': [dn + 'ResNet14_MNIST', 'k:', 0, True],
             'MNIST1 Ne20': [dn + 'ResNet14_MNIST_Ne20', 'k--', 0, True],
             # 'MNIST2 Ne10': [dn + 'ResNet14_MNIST2_nosnr', 'b.', 0, True],
             'MNIST2 Ne20': [dn + 'ResNet14_MNIST2_Ne20', 'b--', 0, True],
             # 'MNIST2 Ne10 snr': [dn + 'ResNet14_MNIST2', 'b-.', 56, True],
             'MNIST2 Ne20 snr': [dn + 'ResNet14_MNIST2_Ne20_snr', 'b-', 56, True],
             # 'MNIST3 Ne10 snr6 16': [dn + 'ResNet14_MNIST3', 'b--x', 56, True],
             # 'MNIST3 Ne10 snr-4 6': [dn + 'ResNet14_MNIST3_snr-4_6', 'b--x', 56, True],
             'MNIST3 Ne20 snr-4 6': [dn + 'ResNet14_MNIST3_Ne20_snr-4_6', 'b-x', 56, True],
             # 'MNIST4 ntx14 Ne10 snr6 16': [dn + 'ResNet14_MNIST4', 'g--D', 14, True],
             # 'MNIST4 ntx14 Ne10 snr-4 6': [dn + 'ResNet14_MNIST4_snr-4_6', 'g--d', 14, True],
             'MNIST4 ntx14 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g-D', 14, True],
             # 'MNIST4 ntx14 nrx56 Ne20 snr-4 6 2': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6_2', 'g-d', 14, True],
             # 'MNIST5 ntx14 Ne10': [dn + 'ResNet14_MNIST5', 'g--', 14, True],
             # 'MNIST6 ntx56 Ne10 snr6 16': [dn + 'ResNet14_MNIST6', 'r--s', 56, True],
             # 'MNIST6 ntx56 Ne10 snr-4 6': [dn + 'ResNet14_MNIST6snr-4_6', 'r--s', 56, True],
             'MNIST6 ntx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_Ne20_snr-4_6', 'r-s', 56, True],
             # 'MNIST6 ntx56 Ne10 snr-10 10': [dn + 'ResNet14_MNIST6_snr-10_10', 'r-x', 56, True],
             # 'MNIST7 ntx56 Ne10': [dn + 'ResNet14_MNIST7', 'r--', 56, True],
             }
    selected_plots.append(mnist)

    # Investigations on number of channel uses: Beginning from MNIST4 [ntx / 2, nrx / 2]
    # [Unpublished]
    mnist_ntx = {'title': ['SINFONY MNIST: Number of channel uses NTx', 'mnist', 56, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'MNIST1': [dn + 'ResNet14_MNIST', 'k--', 0, True],
                 'MNIST2': [dn + 'ResNet14_MNIST2_nosnr', 'b--', 0, True],
                 # 'MNIST6 ntx56 nrx56 snr6 16': [dn + 'ResNet14_MNIST6', 'r--s', 56, True],
                 'MNIST6 ntx56 nrx56 snr-4 6': [dn + 'ResNet14_MNIST6snr-4_6', 'r-s', 56, True],
                 # 'MNIST4 ntx14 nrx56 snr6 16': [dn + 'ResNet14_MNIST4', 'g--D', 14, True],
                 'MNIST4 ntx14 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_snr-4_6', 'g-D', 14, True],
                 # 'MNIST ntx7 nrx28 snr6 16': [dn + 'ResNet14_MNIST4_ntx7', 'g--x', 7, True],
                 'MNIST ntx7 nrx28 snr-4 6': [dn + 'ResNet14_MNIST_ntx7snr-4_6', 'g-x', 7, True],
                 # 'MNIST ntx4 nrx14 snr6 16': [dn + 'ResNet14_MNIST_ntx4', 'g--o', 4, True],
                 'MNIST ntx4 nrx14 snr-4 6': [dn + 'ResNet14_MNIST_ntx4snr-4_6', 'g-o', 4, True],
                 # 'MNIST ntx2 nrx8 snr6 16': [dn + 'ResNet14_MNIST_ntx2', 'g--^', 2, True],
                 'MNIST ntx2 nrx8 snr-4 6': [dn + 'ResNet14_MNIST_ntx2_snr-4_6', 'g-^', 2, True],
                 # 'MNIST ntx2 nrx4 snr-4 6': [dn + 'ResNet14_MNIST4_ntx2_nrx4_snr-4_6', 'g-<', 2, True],
                 }
    # selected_plots.append(mnist_ntx)

    # After the first tries, we used the same Rx layer width Nw=56 to isolate the influence of Ntx
    # [Published]
    mnist_ntx_nrx56 = {'title': ['SINFONY MNIST: Number of channel uses NTx with equal Nw=56', 'mnist', 56, False],
                       # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                       'MNIST2': [dn + 'ResNet14_MNIST2_nosnr', 'b--', 0, True],
                       'MNIST6 ntx56 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_Ne20_snr-4_6', 'r-s', 56, True],
                       'MNIST4 ntx14 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g-D', 14, True],
                       # 'MNIST ntx7 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_ntx7_nrx56_snr-4_6', 'g--x', 7, True],
                       'MNIST ntx7 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST_ntx7_Ne20_snr-4_6', 'm-x', 7, True],
                       # 'MNIST ntx7 nrx56 Ne20 snr6 16': [dn + 'ResNet14_MNIST_ntx7_Ne20_snr6_16', 'm--x', 7, True],
                       'MNIST ntx7 nrx56 Ne20 snr10 20': [dn + 'ResNet14_MNIST_ntx7_Ne20_snr10_20', 'm:x', 7, True],
                       # 'MNIST ntx4 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_ntx4_nrx56_snr-4_6', 'g--o', 4, True],
                       'MNIST ntx4 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST_ntx4_Ne20_snr-4_6', 'm-o', 4, True],
                       # 'MNIST ntx2 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_ntx2_nrx56_snr-4_6', 'g--^', 2, True],
                       'MNIST ntx2 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST_ntx2_Ne20_snr-4_6', 'm-^', 2, True],
                       }
    selected_plots.append(mnist_ntx_nrx56)

    # Investigations on Tx/Rx module layer number: More Tx/Rx layers worse performance in simulation runs
    # [Unpublished]
    mnist_txrxlayers = {'title': ['SINFONY MNIST: Number of Tx/Rx layers', 'mnist', 56, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        'MNIST1': [dn + 'ResNet14_MNIST', 'k--', 0, True],
                        'MNIST2': [dn + 'ResNet14_MNIST2_nosnr', 'b--', 0, True],
                        'MNIST6 ntx56 nrx56 snr-4 6': [dn + 'ResNet14_MNIST6snr-4_6', 'r-s', 56, True],
                        'MNIST6 ntx56 nrx56 snr-4 6 2layer Ne20': [dn + 'ResNet14_MNIST6_2layer_snr-4_6', 'm--s', 56, True],
                        'MNIST4 ntx14 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_snr-4_6', 'g-d', 14, True],
                        'MNIST4 ntx14 nrx56 snr-4 6 Ne20': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g--d', 14, True],
                        # 'MNIST4 ntx14 nrx56 snr-4 6 1layer': [dn + 'ResNet14_MNIST4_1layer_snr-4_6', 'm--', 14, True],
                        'MNIST4 ntx14 nrx56 snr-4 6 2layer Ne20': [dn + 'ResNet14_MNIST4_2layer_snr-4_6', 'm-d', 14, True],
                        # 'MNIST4 ntx14 nrx56 snr-4 6 2layer Ne10': [dn + 'ResNet14_MNIST4_layer2Ne10_snr-4_6', 'm-D', 14, True],
                        'MNIST4 ntx14 nrx56 snr-4 6 3layer Ne20': [dn + 'ResNet14_MNIST4_3layer_snr-4_6', 'm--x', 14, True],
                        }
    selected_plots.append(mnist_txrxlayers)

    # Investigations on Rx module design
    # rx: Individual instead of one Rx module for each received signals yi, yi still enters separately
    # rxjoint: All yi are processed jointly by one large Rx module
    # Results: rx enables slightly better accuracy, rxjoint does not really improve upon rx, more layers do not improve rx/rxjoint
    # [Unpublished, but prepared for PhD Thesis]
    mnist_rx = {'title': ['SINFONY MNIST: Rx design', 'mnist', 56, False],
                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                'MNIST1': [dn + 'ResNet14_MNIST', 'k--', 0, True],
                'MNIST2': [dn + 'ResNet14_MNIST2_nosnr', 'b--', 0, True],
                'MNIST6 ntx56 nrx56 snr-4 6': [dn + 'ResNet14_MNIST6snr-4_6', 'r-s', 56, True],
                # 'MNIST6 rx Ne10 snr-4 6': [dn + 'ResNet14_MNIST6_rx_snr-4_6', 'r--', 56, True],
                'MNIST6 rx Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_rx_Ne20_snr-4_6', 'r--d', 56, True],
                # 'MNIST6 rx Ne30 snr-4 6': [dn + 'ResNet14_MNIST6_rx_Ne30_snr-4_6', 'r--o', 56, True],
                # 'MNIST6 rxjoint Ne10 snr-4 6': [dn + 'ResNet14_MNIST6_rxjointNe10_snr-4_6', 'r-.x', 56, True],
                # 'MNIST6 rxjoint Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_rxjointNe20_snr-4_6', 'r:x', 56, True],
                'MNIST6 rxjoint Ne30 snr-4 6': [dn + 'ResNet14_MNIST6_rxjointNe30_snr-4_6', 'r--x', 56, True],
                # 'MNIST4 ntx14 nrx56 snr-4 6': [dn + 'ResNet14_MNIST4_snr-4_6', 'g-d', 14, True],
                # 'MNIST4 snr-4 6 test': [dn + 'ResNet14_MNIST4_snr-4_6_test', 'm--', 14, True],
                'MNIST4 ntx14 nrx56 snr-4 6 Ne20': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g--d', 14, True],
                # 'MNIST4 rx Ne10 snr-4 6': [dn + 'ResNet14_MNIST4_rxNe10_snr-4_6', 'm-<', 14, True],
                'MNIST4 rx Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_rx_snr-4_6', 'm-d', 14, True],
                # 'MNIST4 rx Ne20 2layer snr-4 6': [dn + 'ResNet14_MNIST4_rx_2layer_snr-4_6', 'm-x', 14, True],
                # 'MNIST4 rxjoint Ne10 snr-4 6': [dn + 'ResNet14_MNIST4_rxjoint_snr-4_6', 'm-x', 14, True],
                # 'MNIST4 rxjoint Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_rxjointNe20_snr-4_6', 'm--', 14, True],
                'MNIST4 rxjoint Ne30 snr-4 6': [dn + 'ResNet14_MNIST4_rxjointNe30_snr-4_6', 'm--x', 14, True],
                # 'MNIST4 rxjoint Ne30 2layer snr-4 6': [dn + 'ResNet14_MNIST4_rxjoint_2layerNe30_snr-4_6', 'm-x', 14, True],
                }
    selected_plots.append(mnist_rx)

    # Investigations on number of channel uses on CIFAR10: Beginning from CIFAR4 [ntx / 2, nrx / 2]
    # [Unpublished]
    cifar_ntx = {'title': ['SINFONY CIFAR: Number of channel uses NTx', 'cifar10', 64, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'CIFAR1': [dn + 'ResNet20_CIFAR', 'k--', 0, True],
                 'CIFAR2': [dn + 'ResNet20_CIFAR2_nosnr', 'b--', 0, True],
                 # 'CIFAR6 ntx64 nrx64 snr6 16': [dn + 'ResNet20_CIFAR6', 'r--s', 64, True],
                 'CIFAR6 ntx64 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR6_snr-4_6', 'r-s', 64, True],
                 # 'CIFAR4 ntx16 nrx64 snr6 16': [dn + 'ResNet20_CIFAR4', 'g--D', 16, True],
                 'CIFAR4 ntx16 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR4_snr-4_6', 'g-D', 16, True],
                 # 'CIFAR ntx8 nrx32 snr6 16': [dn + 'ResNet20_CIFAR4_ntx8', 'g-x', 8, True],
                 'CIFAR ntx8 nrx32 snr-4 6': [dn + 'ResNet20_CIFAR_ntx8_snr-4_6', 'g--x', 8, True],
                 # 'CIFAR ntx4 nrx16 snr6 16': [dn + 'ResNet20_CIFAR_ntx4', 'g-o', 4, True],
                 'CIFAR ntx4 nrx16 snr-4 6': [dn + 'ResNet20_CIFAR_ntx4_snr-4_6', 'g--o', 4, True],
                 'CIFAR ntx2 nrx8 snr6 16': [dn + 'ResNet20_CIFAR_ntx2', 'g-^', 2, True],
                 'CIFAR ntx2 nrx8 snr-4 6': [dn + 'ResNet20_CIFAR_ntx2_snr-4_6', 'g--^', 2, True],
                 }
    selected_plots.append(cifar_ntx)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        selected_plots = [cifar]

    figures = plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                  logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
