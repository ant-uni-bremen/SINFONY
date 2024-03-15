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


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Settings
    y_axis = 'val_acc'   	# val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
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

    # SINFONY applied to HiSE project datasets
    # TODO: Normalization for hise dataset
    # TODO: Evaluation hise sinfony ntx16 imagenet 256x256

    hise = {'title': ['SINFONY: HiSE', 'hise', 0, False],
            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
            'hise resnet 64x64': [dn + 'ResNet20_hise64', 'k-', 0, True],
            'hise resnet imagenet 256x256': [dn + 'ResNet18_hise256_imagenet', 'k--', 0, True],
            # 'hise resnet imagenet cifarpar 256x256': [dn + 'ResNet20_hise256_imagenet_cifarpar', 'k-x', 0, True],
            # 'hise sinfony 64x64 snr6 10 1stlr1e-1': [dn + 'sinfony_hise64', 'r-', 0, True],
            # 'hise sinfony 64x64 snr6 10 1stlr1e-2': [dn + 'sinfony_hise64_1stlr1e-2', 'r--', 0, True],
            'hise sinfony 64x64 snr-4 6': [dn + 'sinfony_hise64_snr-4_6', 'r-o', 0, True],
            'hise sinfony imagenet 256x256': [dn + 'sinfony_hise256_imagenet', 'r--o', 0, True],
            'hise sinfony ntx16 64x64 snr-4 6': [dn + 'sinfony_hise64_ntx16', 'g-o', 0, True],
            'hise sinfony ntx16 imagenet 256x256': [dn + 'sinfony_hise256_imagenet_ntx16', 'g--o', 0, True],
            }
    selected_plots.append(hise)

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
        selected_plots = [hise]

    if copy_models:
        plot_sinfony.copy_published_models2repository(
            selected_plots, datapath=datapath, simulation_filename_prefix=filename_prefix)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
