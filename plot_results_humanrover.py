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
import plot_results_sinfony as plot_sinfony

HUFF_GAIN_FRAESER_IMAGES = 1.4294542972112954
HUFF_GAIN_FRAESER_FEATURES = 1.427700537422152

if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Settings
    # val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
    y_axis = 'val_acc'
    error_mode = True       # Show classification error instead of accuracy
    # (0) w/o SNR normalization, SNR normalization by [number of channel uses/number of features]
    x_axis_normalization = 0
    # (snr) snr value on x axis, (default) index on x axis
    x_axis = 'snr'
    logplot = True          # Logarithmic plot?
    select_plot = False     # Select one plot or plot all preselected plots
    copy_models = False     # Copy published models to public repository
    # Fixed
    datapath = 'models'
    filename_prefix = 'RES_'
    if copy_models:
        dn = ''
    else:
        dn = filename_prefix

    # Plot tables

    selected_plots = []

    # SINFONY for human rover journal with sociologists, e.g., with tool dataset

    tools = {'title': ['SINFONY: Human Rover Tools (705, 380, 1)', '', 256+256, False],
             # 'Tag': ['data name', 'color in plot', channel uses, on/off],
             'ResNet18': ['fraeser/' + dn + 'ResNet18_fraeser_test', 'k--', 0, True],
             'ResNet6 64x64': ['fraeser/' + dn + 'ResNet6_fraeser64_test', 'k-', 0, True],
             'ResNet18 2': ['fraeser/' + dn + 'ResNet18_fraeser', 'k--x', 0, True],
             # 'ResNet18 3': ['fraeser/' + dn + 'ResNet18_fraeser_2', 'k-x', 0, True],
             # 'ResNet20': ['fraeser/' + dn + 'ResNet20_fraeser', 'k--<', 0, True],
             # 'ResNet20 filters16': ['fraeser/' + dn + 'ResNet20_fraeser_filters16', 'k--D', 0, True],
             # 'ResNet50': ['fraeser/' + dn + 'ResNet50_fraeser_test', 'k--o', 0, True],
             # 'SINFONY snr-4 6 lr1e-2': ['fraeser/' + dn + 'sinfony18_fraeser', 'g--D', 512, True],
             # 'SINFONY snr-4 6 lr1e-2 txrx weight_decay': ['fraeser/' + dn + 'sinfony18_fraeser_test', 'g--x', 512, True],
             # 'SINFONY snr-4 6': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3', 'b-o', 512, True],
             # 'SINFONY snr-4 6 txrx weight_decay': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_test', 'b--x', 512, True],
             # 'SINFONY snr-4 6 2': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_2', 'b-s', 512, True],
             'SINFONY snr-4 6 3': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_3', 'b--s', 512, True],
             # 'SINFONY snr-4 6 4': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_4', 'b--o', 512, True],
             # 'SINFONY snr-4 6 5': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_5', 'b--D', 512, True],
             # 'SINFONY snr-4 6 6': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_6', 'b--<', 512, True],
             'SINFONY snr-4 6 full image': ['fraeser/' + dn + 'sinfony18_fraeser_fullimage', 'k--s', 512, True],
             'SINFONY snr-4 6 ntx128': ['fraeser/' + dn + 'sinfony18_fraeser_ntx128', 'r-o', 128, True],
             # 'SINFONY snr-4 6 ntx256': ['fraeser/' + dn + 'sinfony18_fraeser_ntx256', 'r--x', 256, True],
             # 'SINFONY snr-4 6 ntx64': ['fraeser/' + dn + 'sinfony18_fraeser_ntx64', 'r--o', 64, True],
             # 'SINFONY snr-4 6 ntx16': ['fraeser/' + dn + 'sinfony18_fraeser_ntx16', 'r--s', 16, True],
             # 'SINFONY snr-4 6 ntx32': ['fraeser/' + dn + 'sinfony18_fraeser_ntx32', 'r--<', 32, True],
             # 'SINFONY snr-4 6 ntx64 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx64_2', 'g-.o', 64, True],
             # 'SINFONY snr-4 6 ntx16 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx16_2', 'g-.s', 16, True],
             # 'SINFONY snr-4 6 ntx32 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx32_2', 'g-.<', 32, True],
             # 'ResNet18 image classic rc25 n=15360 h10': ['classic/' + dn + 'classic_image_' + 'ResNet18_fraeser_rc25_n15360_h10_test', 'k-x', (218 + 487) * 380 / 2 * 1 * 8 / (0.25 * HUFF_GAIN_FRAESER_IMAGES), True],
             'ResNet18 image classic rc25 n=15360 h1': ['classic/' + dn + 'classic_image_' + 'ResNet18_fraeser_rc25_n15360_h1_test', 'k--x', (218 + 487) * 380 / 2 * 1 * 8 / (0.25 * HUFF_GAIN_FRAESER_IMAGES), True],
             # 'ResNet18 features classic rc25 n=15360 h100': ['classic/' + dn + 'classic_' + 'ResNet18_fraeser_rc25_n15360_h100_test', 'k-o', 1024 / 2 * 16 / (0.25 * HUFF_GAIN_FRAESER_FEATURES), True],
             'ResNet18 features classic rc25 n=15360 h10': ['classic/' + dn + 'classic_' + 'ResNet18_fraeser_rc25_n15360_h10_test', 'k--o', 1024 / 2 * 16 / (0.25 * HUFF_GAIN_FRAESER_FEATURES), True],
             }
    selected_plots.append(tools)

    tools64 = {'title': ['SINFONY: Human Rover Tools (118, 64, 1)', 'fraeser', 36+64, False],
               # 'Tag': ['data name', 'color in plot', channel uses, on/off],
               'ResNet6 64x64': [dn + 'ResNet6_fraeser64_test', 'k-', 0, True],
               # 'ResNet6 64x64 2': [dn + 'ResNet6_fraeser64', 'k--x', 0, True],
               # 'ResNet6 64x64 3': [dn + 'ResNet6_fraeser64_2', 'k--<', 0, True],
               # 'ResNet14 64x64': [dn + 'ResNet14_fraeser64', 'k--D', 0, True],
               # 'SINFONY 64x64 txrx weight_decay': [dn + 'sinfony6_fraeser64_test', 'b--o', 100, True],
               'SINFONY 64x64': [dn + 'sinfony6_fraeser64', 'b-o', 100, True],
               # 'SINFONY 64x64 2': [dn + 'sinfony6_fraeser64_2', 'b-D', 100, True],
               # 'SINFONY 64x64 ResNet14 ': [dn + 'sinfony14_fraeser64', 'b-x', 100, True],
               # 'SINFONY 64x64 snr6 16': [dn + 'sinfony6_fraeser64_snr6_16', 'b-s', 100, True],
               'SINFONY 64x64 full image': [dn + 'sinfony6_fraeser64_fullimage', 'k--s', 100, True],
               # 'SINFONY 64x64 full image 2': [dn + 'sinfony6_fraeser64_full_image_2', 'k--o', 100, True],
               'SINFONY 64x64 ntx16': [dn + 'sinfony6_fraeser64_ntx16', 'r-o', 16, True],
               # 'SINFONY 64x64 ntx16 snr6 16': [dn + 'sinfony6_fraeser64_ntx16_snr6_16', 'r-s', 16, True],
               }
    selected_plots.append(tools64)

    human_rover_cifar = {'title': ['SINFONY: Human Rover CIFAR10', 'cifar10', 64, False],
                         # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                         'ResNet': [dn + 'ResNet20_CIFAR', 'k--', 0, True],
                         'ResNet human': [dn + 'ResNet20_CIFAR_human', 'k--x', 0, True],
                         'CIFAR4 ntx16 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR4_snr-4_6', 'g-D', 16, True],
                         'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human', 'b--D', 16, True],
                         'SINFONY ntx16 nrx64 snr-4 6 test': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human_test', 'b--o', 16, True],
                         'CIFAR6 ntx64 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR6_snr-4_6', 'r-s', 64, True],
                         'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human', 'b--s', 64, True],
                         'SINFONY ntx64 nrx64 snr-4 6 test': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human_test', 'b--o', 64, True],
                         }
    selected_plots.append(human_rover_cifar)

    human_rover_mnist = {'title': ['SINFONY: Human Rover MNIST', 'mnist', 56, False],
                         # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                         'ResNet Ne20': [dn + 'ResNet14_MNIST_Ne20', 'k--', 0, True],
                         'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human', 'k--x', 0, True],
                         'SINFONY ntx14 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g-D', 14, True],
                         'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'b--D', 14, True],
                         'SINFONY ntx14 nrx56 Ne20 snr-4 6 human test': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human_test', 'b--o', 14, True],
                         'SINFONY ntx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_Ne20_snr-4_6', 'r-s', 56, True],
                         'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'b--s', 56, True],
                         'SINFONY ntx56 Ne20 snr-4 6 human test': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human_test', 'b--o', 56, True],
                         }
    selected_plots.append(human_rover_mnist)

    # hirise dataset idea for joint results with Psychologists and Sociologists: Deprecated
    # [Unpublished]
    hirise = {'title': ['SINFONY: hirise', 'hirise', 0, False],
              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
              'hirise20 32x32': [dn + 'ResNet20_hirise', 'g-', 0, True],
              'hirise20 32x32 dist': [dn + 'ResNet20_hirise32_dist_nosnr', 'g-o', 0, True],
              'hirise20 32x32 dist snr': [dn + 'ResNet20_hirise32_dist', 'g--', 0, True],
              # 'hirise20 32x32 ntx16': [dn + 'ResNet20_hirise32_ntx16_snr-4_6', 'g--<', 0, True],
              'hirise32 64x64': [dn + 'ResNet32_hirise64', 'b-', 0, True],
              'hirise32 64x64 dist': [dn + 'ResNet32_hirise64_dist', 'b-o', 0, True],
              'hirise32 64x64 ntx32 snr-4_6': [dn + 'ResNet32_hirise64_ntx32_snr-4_6', 'b--x', 0, True],
              # 'hirise32 64x64 ntx32 snr6_16 rx individual': [dn + 'ResNet32_hirise64_ntx32_rx', 'b-->', 0, True],
              # 'hirise32 64x64 ntx16 snr6_16 rx individual': [dn + 'ResNet32_hirise64_ntx16_rx', 'b--<', 0, True],
              # 'hirise32 64x64 ntx16 snr6_16': [dn + 'ResNet32_hirise64_ntx16_snr6_16', 'r--<', 0, True],
              'hirise32 64x64 ntx16 snr-4_6': [dn + 'ResNet32_hirise64_ntx16', 'r--o', 0, True],
              # 'hirise56 64x64': [dn + 'ResNet56_hirise64', 'b-x', 0, True],
              # 'hirise32 128x128': [dn + 'ResNet32_hirise128', 'r-<', 0, True],
              # 'hirise42 rblock4 64x64': [dn + 'ResNet42_rblock4_hirise64', 'm-', 0, True],
              # 'hirise52 rblock5 128x128': [dn + 'ResNet52_rblock5_hirise128', 'm--', 0, True],
              }
    # selected_plots.append(hirise)
    # Only decision between crater or no crater
    hirisecrater = {'title': ['SINFONY: hirisecrater', 'hirise', 0, False],
                    # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                    'hirisecrater20 32x32': [dn + 'ResNet20_hirisecrater32', 'b-', 0, True],
                    'hirisecrater20 32x32 dist': [dn + 'ResNet20_hirisecrater32_dist', 'b-o', 0, True],
                    'hirisecrater20 64x64': [dn + 'ResNet20_hirisecrater64', 'g-', 0, True],
                    'hirisecrater20 64x64 dist': [dn + 'ResNet20_hirisecrater64_dist', 'g-o', 0, True],
                    # 'hirisecrater32 64x64': [dn + 'ResNet32_hirisecrater64', 'b-x', 0, True],
                    # 'hirisecrater20 32x32 ntx16 snr-4_6': [dn + 'ResNet20_hirisecrater32_ntx16_snr-4_6', 'b--<', 0, True],
                    'hirisecrater20 64x64 ntx32 snr-4_6': [dn + 'ResNet20_hirisecrater64_ntx32_snr-4_6', 'g--<', 0, True],
                    # 'hirisecrater32 64x64 ntx16 snr-4_6': [dn + 'ResNet32_hirisecrater64_ntx16', 'r--x', 0, True],
                    'hirisecrater20 64x64 ntx16 snr-4_6': [dn + 'ResNet20_hirisecrater64_ntx16_snr-4_6', 'r-->', 0, True],
                    # 'hirisecrater32 64x64 ntx16 snr6_16': [dn + 'ResNet32_hirisecrater64_ntx16_snr6_16', 'r--o', 0, True],
                    # 'hirisecrater20 64x64 ntx16 snr6_16': [dn + 'ResNet20_hirisecrater64_ntx16_snr6_16', 'r--s', 0, True],
                    }
    # selected_plots.append(hirisecrater)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_classic, mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [tools]

    if copy_models:
        plot_sinfony.copy_published_models2repository(
            selected_plots, datapath=datapath, simulation_filename_prefix=filename_prefix)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
