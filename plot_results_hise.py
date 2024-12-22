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

    # 512 features per subimage, Ntx counted per subimage
    # Bits per subimage: 256*256*3*8
    hise_v2 = {'title': ['SINFONY: HiSE 256 v2', 'hise', 512, False],
               # 'Tag': ['data name', 'color in plot', channel uses, on/off],
               'hise resnet imagenet 256x256': [dn + 'ResNet18_hise256_v2_imagenet', 'k--', 0, True],
               # 'hise resnet imagenet 256x256 final layer ?': [dn + 'ResNet18_hise256_v2_imagenet_test', 'k-o', 0, True],
               'hise resnet imagenet 256x256 one image': [dn + 'ResNet18_hise256_v2_imagenet_oneimage', 'k:', 0, True],
               'hise sinfony ntx512 imagenet 256x256': [dn + 'sinfony_hise256_v2_imagenet', 'r--o', 512, True],
               'hise sinfony ntx64 imagenet 256x256': [dn + 'sinfony_hise256_v2_imagenet_ntx64', 'g--o', 64, True],
               }
    selected_plots.append(hise_v2)

    hise64_v2 = {'title': ['SINFONY: HiSE 64 v2', 'hise', 64, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'hise resnet 64x64': [dn + 'ResNet20_hise64_v2', 'k-', 0, True],
                 'hise resnet 64x64 one image': [dn + 'ResNet20_hise64_v2_oneimage', 'k:', 0, True],
                 'hise sinfony ntx64 64x64 snr-4 6': [dn + 'sinfony_hise64_v2', 'r-o', 64, True],
                 'hise sinfony ntx16 64x64 snr-4 6': [dn + 'sinfony_hise64_v2_ntx16', 'g-o', 16, True],
                 }
    selected_plots.append(hise64_v2)

    # Old versions

    hise = {'title': ['SINFONY: HiSE 256', 'hise', 512, False],
            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
            'hise resnet imagenet 256x256': [dn + 'ResNet18_hise256_imagenet', 'k--', 0, True],
            # 'hise resnet imagenet cifarpar 256x256': [dn + 'ResNet20_hise256_imagenet_cifarpar', 'k-x', 0, True],
            'hise sinfony ntx512 imagenet 256x256': [dn + 'sinfony_hise256_imagenet', 'r--o', 512, True],
            'hise sinfony ntx64 imagenet 256x256': [dn + 'sinfony_hise256_imagenet_ntx64', 'g--o', 64, True],
            }
    # selected_plots.append(hise)

    hise64 = {'title': ['SINFONY: HiSE 64', 'hise', 64, False],
              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
              'hise resnet 64x64': [dn + 'ResNet20_hise64', 'k-', 0, True],
              # 'hise sinfony 64x64 snr6 10 1stlr1e-1': [dn + 'sinfony_hise64', 'r-', 64, True],
              # 'hise sinfony 64x64 snr6 10 1stlr1e-2': [dn + 'sinfony_hise64_1stlr1e-2', 'r--', 64, True],
              'hise sinfony ntx64 64x64 snr-4 6': [dn + 'sinfony_hise64_snr-4_6', 'r-o', 64, True],
              'hise sinfony ntx16 64x64 snr-4 6': [dn + 'sinfony_hise64_ntx16', 'g-o', 16, True],
              }
    # selected_plots.append(hise64)

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
