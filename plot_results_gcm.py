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
    error_mode = False       # Show classification error instead of accuracy
    # (0) w/o SNR normalization, SNR normalization by [number of channel uses/number of features]
    x_axis_normalization = 0
    # (snr) snr value on x axis, (default) index on x axis
    x_axis = 'snr'
    logplot = 1             # Logarithmic plot?
    select_plot = False     # Select one plot or plot all preselected plots
    copy_models = False     # Copy published models to public repository
    # Fixed
    datapath = 'models'
    filename_prefix = ''
    suf = '_results'
    if copy_models:
        dn = ''
    else:
        dn = filename_prefix

    # Plot tables

    selected_plots = []

    # SINFONY for human rover journal with sociologists, e.g., with tool dataset

    tools = {'title': ['SINFONY: Tool Wear (705, 380, 1)', '', 256 + 256, False],
             # 'Tag': ['data name', 'color in plot', channel uses, on/off],
             'ResNet18': ['fraeser/' + dn + 'ResNet18_fraeser_test' + suf, 'k--', 0, True],
             'ResNet6 64x64': ['fraeser/' + dn + 'ResNet6_fraeser64_test' + suf, 'k-', 0, True],
             'ResNet18 2': ['fraeser/' + dn + 'ResNet18_fraeser' + suf, 'k--x', 0, True],
             # 'ResNet18 3': ['fraeser/' + dn + 'ResNet18_fraeser_2' + suf, 'k-x', 0, True],
             # 'ResNet20': ['fraeser/' + dn + 'ResNet20_fraeser' + suf, 'k--<', 0, True],
             # 'ResNet20 filters16': ['fraeser/' + dn + 'ResNet20_fraeser_filters16' + suf, 'k--D', 0, True],
             # 'ResNet50': ['fraeser/' + dn + 'ResNet50_fraeser_test' + suf, 'k--o', 0, True],
             # 'SINFONY snr-4 6 lr1e-2': ['fraeser/' + dn + 'sinfony18_fraeser' + suf, 'g--D', 512, True],
             # 'SINFONY snr-4 6 lr1e-2 txrx weight_decay': ['fraeser/' + dn + 'sinfony18_fraeser_test' + suf, 'g--x', 512, True],
             # 'SINFONY snr-4 6': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3' + suf, 'b-o', 512, True],
             # 'SINFONY snr-4 6 txrx weight_decay': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_test' + suf, 'b--x', 512, True],
             # 'SINFONY snr-4 6 2': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_2' + suf, 'b-s', 512, True],
             'SINFONY snr-4 6 3': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_3' + suf, 'b--s', 512, True],
             # 'SINFONY snr-4 6 4': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_4' + suf, 'b--o', 512, True],
             # 'SINFONY snr-4 6 5': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_5' + suf, 'b--D', 512, True],
             # 'SINFONY snr-4 6 6': ['fraeser/' + dn + 'sinfony18_fraeser_lr1e-3_6' + suf, 'b--<', 512, True],
             # 'SINFONY snr-4 6 full image': ['fraeser/' + dn + 'sinfony18_fraeser_fullimage' + suf, 'k--s', 512, True],
             'SINFONY snr-4 6 ntx128': ['fraeser/' + dn + 'sinfony18_fraeser_ntx128' + suf, 'r-o', 128, True],
             # 'SINFONY snr-4 6 ntx256': ['fraeser/' + dn + 'sinfony18_fraeser_ntx256' + suf, 'r--x', 256, True],
             # 'SINFONY snr-4 6 ntx64': ['fraeser/' + dn + 'sinfony18_fraeser_ntx64' + suf, 'r--o', 64, True],
             # 'SINFONY snr-4 6 ntx16': ['fraeser/' + dn + 'sinfony18_fraeser_ntx16' + suf, 'r--s', 16, True],
             # 'SINFONY snr-4 6 ntx32': ['fraeser/' + dn + 'sinfony18_fraeser_ntx32' + suf, 'r--<', 32, True],
             # 'SINFONY snr-4 6 ntx64 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx64_2' + suf, 'g-.o', 64, True],
             # 'SINFONY snr-4 6 ntx16 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx16_2' + suf, 'g-.s', 16, True],
             # 'SINFONY snr-4 6 ntx32 2': ['fraeser/' + dn + 'sinfony18_fraeser_ntx32_2' + suf, 'g-.<', 32, True],
             'ResNet18 image classic rc25 n=15360 h1': ['classic/' + dn + 'classic_image_' + 'ResNet18_fraeser_rc25_n15360_h1_fine' + suf, 'k-x', (218 + 487) * 380 / 2 * 1 * 8 / (0.25 * HUFF_GAIN_FRAESER_IMAGES), True],
             'ResNet18 image classic rc25 n=15360 h10': ['classic/' + dn + 'classic_image_' + 'ResNet18_fraeser_rc25_n15360_h10_fine' + suf, 'k-*', (218 + 487) * 380 / 2 * 1 * 8 / (0.25 * HUFF_GAIN_FRAESER_IMAGES), True],
             # 'ResNet18 image classic rc25 n=15360 h1': ['classic/' + dn + 'classic_image_' + 'ResNet18_fraeser_rc25_n15360_h1_test' + suf, 'k--x', (218 + 487) * 380 / 2 * 1 * 8 / (0.25 * HUFF_GAIN_FRAESER_IMAGES), True],
             # 'ResNet18 features classic rc25 n=15360 h100': ['classic/' + dn + 'classic_' + 'ResNet18_fraeser_rc25_n15360_h100_test' + suf, 'k-o', 1024 / 2 * 16 / (0.25 * HUFF_GAIN_FRAESER_FEATURES), True],
             # 'ResNet18 features classic rc25 n=15360 h10': ['classic/' + dn + 'classic_' + 'ResNet18_fraeser_rc25_n15360_h10_test' + suf, 'k--o', 1024 / 2 * 16 / (0.25 * HUFF_GAIN_FRAESER_FEATURES), True],
             }
    selected_plots.append(tools)

    gcm_tools = {'title': ['SINFONY+GCM: Tool Wear (705, 380, 1)', 'fraeser', 256 + 256, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'ResNet18 2': [dn + 'ResNet18_fraeser' + suf, 'k--x', 0, True],
                 'SINFONY snr-4 6 3': [dn + 'sinfony18_fraeser_lr1e-3_3' + suf, 'k-', 512, True],
                 'SINFONY snr-4 6 ntx128': [dn + 'sinfony18_fraeser_ntx128' + suf, 'k--', 128, True],
                 'GCM prob + SINFONY ntx128 snr-4 6': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf, 'r--', 128, True],
                 'GCM opt + SINFONY ntx128': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf + '_opt', 'r--x', 128, True],
                 'GCM Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures5+sinfony18_fraeser_ntx128' + suf, 'm--', 128, True],
                 'GCM Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures10+sinfony18_fraeser_ntx128' + suf, 'g--', 128, True],
                 'GCM Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures20+sinfony18_fraeser_ntx128' + suf, 'y--', 128, True],
                 'GCM Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures40+sinfony18_fraeser_ntx128' + suf, 'b--', 128, True],
                 'GCM Nfeat max Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures-1+sinfony18_fraeser_ntx128' + suf, 'c--', 128, True],
                 'GCM prob + SINFONY ntx512 snr-4 6': [dn + 'GCM+sinfony18_fraeser_lr1e-3_3' + suf, 'k-o', 512, True],
                 'GCM opt + SINFONY ntx512': [dn + 'GCM+sinfony18_fraeser_lr1e-3_3' + suf + '_opt', 'k--o', 512, True],
                 #  'GCM snr20 prob + SINFONY ntx128 snr-4 6': [dn + 'GCM_snr20+sinfony18_fraeser_ntx128' + suf, 'r:', 128, True],
                 #  'GCM snr20 opt + SINFONY ntx128': [dn + 'GCM_snr20+sinfony18_fraeser_ntx128' + suf + '_opt', 'r:x', 128, True],
                 #  'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures5+sinfony18_fraeser_ntx128' + suf, 'm:', 128, True],
                 #  'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures10+sinfony18_fraeser_ntx128' + suf, 'g:', 128, True],
                 #  'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures20+sinfony18_fraeser_ntx128' + suf, 'y:', 128, True],
                 #  'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures40+sinfony18_fraeser_ntx128' + suf, 'b:', 128, True],
                 #  'GCM snr20 Nfeat max Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures-1+sinfony18_fraeser_ntx128' + suf, 'c:', 128, True],
                 }
    selected_plots.append(gcm_tools)

    tools64 = {'title': ['SINFONY: Tool Wear (118, 64, 1)', 'fraeser', 36 + 64, False],
               # 'Tag': ['data name', 'color in plot', channel uses, on/off],
               'ResNet6 64x64': [dn + 'ResNet6_fraeser64_test' + suf, 'k-', 0, True],
               # 'ResNet6 64x64 2': [dn + 'ResNet6_fraeser64' + suf, 'k--x', 0, True],
               # 'ResNet6 64x64 3': [dn + 'ResNet6_fraeser64_2' + suf, 'k--<', 0, True],
               # 'ResNet14 64x64': [dn + 'ResNet14_fraeser64' + suf, 'k--D', 0, True],
               # 'SINFONY 64x64 txrx weight_decay': [dn + 'sinfony6_fraeser64_test' + suf, 'b--o', 100, True],
               'SINFONY 64x64': [dn + 'sinfony6_fraeser64' + suf, 'b-o', 100, True],
               # 'SINFONY 64x64 2': [dn + 'sinfony6_fraeser64_2' + suf, 'b-D', 100, True],
               # 'SINFONY 64x64 ResNet14 ': [dn + 'sinfony14_fraeser64' + suf, 'b-x', 100, True],
               # 'SINFONY 64x64 snr6 16': [dn + 'sinfony6_fraeser64_snr6_16' + suf, 'b-s', 100, True],
               'SINFONY 64x64 full image': [dn + 'sinfony6_fraeser64_fullimage' + suf, 'k--s', 100, True],
               # 'SINFONY 64x64 full image 2': [dn + 'sinfony6_fraeser64_full_image_2' + suf, 'k--o', 100, True],
               'SINFONY 64x64 ntx16': [dn + 'sinfony6_fraeser64_ntx16' + suf, 'r-o', 16, True],
               # 'SINFONY 64x64 ntx16 snr6 16': [dn + 'sinfony6_fraeser64_ntx16_snr6_16' + suf, 'r-s', 16, True],
               }
    # selected_plots.append(tools64)

    human_rover_cifar = {'title': ['SINFONY: Human Rover CIFAR10', 'cifar10', 64, False],
                         # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                         'ResNet': [dn + 'ResNet20_CIFAR' + suf, 'k--', 0, True],
                         'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k--x', 0, True],
                         'CIFAR4 ntx16 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR4_snr-4_6' + suf, 'g-D', 16, True],
                         'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'b--D', 16, True],
                         'SINFONY ntx16 nrx64 snr-4 6 test': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'b--o', 16, True],
                         'CIFAR6 ntx64 nrx64 snr-4_6': [dn + 'ResNet20_CIFAR6_snr-4_6' + suf, 'r-s', 64, True],
                         'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b--s', 64, True],
                         'SINFONY ntx64 nrx64 snr-4 6 test': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human_test' + suf, 'b--o', 64, True],
                         }
    # selected_plots.append(human_rover_cifar)

    gcm_cifar = {'title': ['SINFONY+GCM: CIFAR10 ntx64', 'cifar10', 64, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k--x', 0, True],
                 'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'k-', 64, True],
                 'GCM prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'r--', 64, True],
                 'GCM opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf + '_opt', 'r--x', 64, True],
                 'GCM Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'm--', 64, True],
                 'GCM Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'g--', 64, True],
                 'GCM Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'y--', 64, True],
                 'GCM Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b--', 64, True],
                 'GCM Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'c--', 64, True],
                 #  'GCM snr20 prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'r:', 64, True],
                 #  'GCM snr20 opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf + '_opt', 'r:x', 64, True],
                 #  'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'm:', 64, True],
                 #  'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'g:', 64, True],
                 #  'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'y:', 64, True],
                 #  'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b:', 64, True],
                 #  'GCM snr20 Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'c:', 64, True],
                 }
    selected_plots.append(gcm_cifar)

    gcm_cifar_ntx16 = {'title': ['SINFONY+GCM: CIFAR10 ntx16', 'cifar10', 64, False],
                       # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                       'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k--x', 0, True],
                       'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k--', 16, True],
                       'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'k-', 64, True],
                       'GCM prob + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'r--', 16, True],
                       'GCM opt + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf + '_opt', 'r--x', 16, True],
                       # 'GCM old prob + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'r--<', 16, True],
                       # 'GCM old opt + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf + '_opt', 'r->', 16, True],
                       'GCM Nfeat 5 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures5+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'm--', 16, True],
                       # 'GCM test Nfeat 5 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test_Nfeatures5+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'm--<', 16, True],
                       # 'GCM Nfeat 7 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures7+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k--^', 16, True],
                       # 'GCM Nfeat 8 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures8+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k--<', 16, True],
                       # 'GCM Nfeat 9 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures9+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k-->', 16, True],
                       'GCM Nfeat 10 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures10+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'g--', 16, True],
                       # 'GCM test Nfeat 10 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test_Nfeatures10+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'g--<', 16, True],
                       'GCM Nfeat 20 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures20+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'y--', 16, True],
                       # 'GCM test Nfeat 20 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test_Nfeatures20+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'y--<', 16, True],
                       'GCM Nfeat 40 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures40+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'b--', 16, True],
                       # 'GCM test Nfeat 40 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test_Nfeatures40+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'b--<', 16, True],
                       'GCM Nfeat max Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures-1+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'c--', 16, True],
                       # 'GCM test Nfeat max Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_test_Nfeatures-1+sinfony20_CIFAR_ntx16_snr-4_6_human_test' + suf, 'c--<', 16, True],
                       }
    # selected_plots.append(gcm_cifar_ntx16)

    human_rover_mnist = {'title': ['SINFONY: Human Rover MNIST', 'mnist', 56, False],
                         # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                         # 'ResNet Ne20': [dn + 'ResNet14_MNIST_Ne20' + suf, 'k--', 0, True],
                         'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                         # 'SINFONY ntx14 nrx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST4_Ne20_snr-4_6' + suf, 'g-D', 14, True],
                         'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'b--D', 14, True],
                         # 'SINFONY ntx14 nrx56 Ne20 snr-4 6 human test': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human_test' + suf, 'b--o', 14, True],
                         # 'SINFONY ntx56 Ne20 snr-4 6': [dn + 'ResNet14_MNIST6_Ne20_snr-4_6' + suf, 'r-s', 56, True],
                         'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b--s', 56, True],
                         # 'SINFONY ntx56 Ne20 snr-4 6 human test': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human_test' + suf, 'b--o', 56, True],
                         }
    # selected_plots.append(human_rover_mnist)

    gcm_mnist = {'title': ['SINFONY+GCM: MNIST ntx56', 'mnist', 56, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k-', 56, True],
                 'GCM prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'r--', 56, True],
                 'GCM opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 56, True],
                 'GCM Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'm--', 56, True],
                 'GCM Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'g--', 56, True],
                 'GCM Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'y--', 56, True],
                 'GCM Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b--', 56, True],
                 'GCM Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'c--', 56, True],
                 #  'GCM snr20 prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'r:', 56, True],
                 #  'GCM snr20 opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'r:x', 56, True],
                 #  'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'm:', 56, True],
                 #  'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'g:', 56, True],
                 #  'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'y:', 56, True],
                 #  'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b:', 56, True],
                 #  'GCM snr20 Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'c:', 56, True],
                 }
    selected_plots.append(gcm_mnist)

    gcm_mnist_ntx14 = {'title': ['SINFONY+GCM: MNIST ntx14', 'mnist', 56, False],
                       # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                       # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                       # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k-', 56, True],
                       'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k--^', 14, True],
                       'GCM prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'r--', 14, True],
                       'GCM prob Ne10 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'r--x', 14, True],
                       'GCM opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r-', 14, True],
                       'GCM opt Ne10 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r-x', 14, True],
                       'GCM Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'm--', 14, True],
                       'GCM Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'g--', 14, True],
                       # 'GCM new Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_new_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'g--<', 14, True],
                       # 'GCM old Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_old_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'g-->', 14, True],
                       'GCM Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'y--', 14, True],
                       'GCM Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'b--', 14, True],
                       'GCM Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'c--', 14, True],
                       }
    # selected_plots.append(gcm_mnist_ntx14)

    gcm_opt_mnist_ntx14 = {'title': ['SINFONY+GCM: MNIST ntx14 optimal policy', 'mnist', 56, False],
                           # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                           'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                           'SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k-', 56, True],
                           'GCM opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r--', 14, True],
                           'GCM opt Ne10 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 14, True],
                           'GCM Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'm--', 14, True],
                           'GCM Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'g--', 14, True],
                           'GCM Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'y--', 14, True],
                           'GCM Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'b--', 14, True],
                           'GCM Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'c--', 14, True],
                           }
    # selected_plots.append(gcm_opt_mnist_ntx14)

    gcm_mnist_ntx14_joint_training = {'title': ['SINFONY+GCM: MNIST ntx14 joint training', 'mnist', 56, False],
                                      # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                      # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                                      # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k-', 56, True],
                                      'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k--^', 14, True],
                                      'GCM prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'r--', 14, True],
                                      'GCM opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 14, True],
                                      'GCM prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf, 'y--', 14, True],
                                      # 'GCM opt Ne1 joint training Nalt20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf + '_opt', 'y--*', 14, True],
                                      # 'GCM prob Ne1 joint training Nalt100 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf, 'g-*', 14, True],
                                      # 'GCM opt Ne1 joint training Nalt100 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf + '_opt', 'g--*', 14, True],
                                      'GCM prob Ne1 joint training Nalt20 Ne3 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne3+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf, 'g--', 14, True],
                                      # 'GCM opt Ne1 joint training Nalt20 Ne3 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne3+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf + '_opt', 'g--*', 14, True],
                                      'GCM Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'c--', 14, True],
                                      'GCM Nfeat max Ne1 joint training Nalt20 Ne10 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf, 'c-*', 14, True],
                                      'GCM diff prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human+joint_training' + suf, 'y-.*', 14, True],
                                      }
    # selected_plots.append(gcm_mnist_ntx14_joint_training)

    gcm_mnist_ntx56_joint_training = {'title': ['SINFONY+GCM: MNIST ntx56 joint training', 'mnist', 56, False],
                                      # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                      # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k--x', 0, True],
                                      'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k-', 56, True],
                                      'GCM prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'r--', 56, True],
                                      # 'GCM opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 56, True],
                                      'GCM Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'm--', 56, True],
                                      'GCM Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'g--', 56, True],
                                      'GCM Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'y--', 56, True],
                                      'GCM Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b--', 56, True],
                                      'GCM Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'c--', 56, True],
                                      'GCM prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'r--*', 56, True],
                                      # 'GCM opt Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf + '_opt', 'y--*', 56, True],
                                      # 'GCM prob Ne1 joint training Nalt20 Ne5 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne5_Na20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'b--', 56, True],
                                      # 'GCM opt Ne1 joint training Nalt20 Ne5 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne5_Na20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf + '_opt', 'b--*', 56, True],
                                      'GCM Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'm--*', 56, True],
                                      'GCM Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'g--*', 56, True],
                                      'GCM Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'y--*', 56, True],
                                      'GCM Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'b--*', 56, True],
                                      'GCM Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'c--*', 56, True],
                                      'GCM diff prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'r-.', 56, True],
                                      'GCM diff Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'm-.', 56, True],
                                      'GCM diff Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'g-.', 56, True],
                                      'GCM diff Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'y-.', 56, True],
                                      'GCM diff Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'b-.', 56, True],
                                      'GCM diff Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human+joint_training' + suf, 'c-.', 56, True],
                                      }
    selected_plots.append(gcm_mnist_ntx56_joint_training)

    gcm_cifar_joint_training = {'title': ['SINFONY+GCM: CIFAR10 ntx64 joint training', 'cifar10', 64, False],
                                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k--x', 0, True],
                                'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'k-', 64, True],
                                'GCM prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'r--', 64, True],
                                # 'GCM opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf + '_opt', 'r--x', 64, True],
                                'GCM Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'm--', 64, True],
                                'GCM Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'g--', 64, True],
                                'GCM Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'y--', 64, True],
                                'GCM Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b--', 64, True],
                                'GCM Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'c--', 64, True],
                                'GCM prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'r--*', 64, True],
                                'GCM Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'm--*', 64, True],
                                'GCM Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'g--*', 64, True],
                                'GCM Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'y--*', 64, True],
                                'GCM Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'b--*', 64, True],
                                'GCM Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'c--*', 64, True],
                                'GCM diff prob Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'r-.', 64, True],
                                'GCM diff Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'm-.', 64, True],
                                'GCM diff Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'g-.', 64, True],
                                'GCM diff Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'y-.', 64, True],
                                'GCM diff Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'b-.', 64, True],
                                'GCM diff Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human+joint_training' + suf, 'c-.', 64, True],
                                }
    selected_plots.append(gcm_cifar_joint_training)

    gcm_tools_joint_training = {'title': ['SINFONY+GCM: Tool Wear (705, 380, 1) joint training', 'fraeser', 256 + 256, False],
                                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                'ResNet18 2': [dn + 'ResNet18_fraeser' + suf, 'k--x', 0, True],
                                'SINFONY snr-4 6 ntx128': [dn + 'sinfony18_fraeser_ntx128' + suf, 'k--', 128, True],
                                'GCM prob + SINFONY ntx128 snr-4 6': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf, 'r--', 128, True],
                                # 'GCM opt + SINFONY ntx128': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf + '_opt', 'r--x', 128, True],
                                'GCM Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures5+sinfony18_fraeser_ntx128' + suf, 'm--', 128, True],
                                'GCM Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures10+sinfony18_fraeser_ntx128' + suf, 'g--', 128, True],
                                'GCM Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures20+sinfony18_fraeser_ntx128' + suf, 'y--', 128, True],
                                'GCM Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures40+sinfony18_fraeser_ntx128' + suf, 'b--', 128, True],
                                'GCM Nfeat max Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures-1+sinfony18_fraeser_ntx128' + suf, 'c--', 128, True],
                                'GCM prob joint training Nalt20 Ne1 + SINFONY ntx128 snr-4 6': [dn + 'GCM_Ne1_Na20+sinfony18_fraeser_ntx128+joint_training' + suf, 'r--*', 128, True],
                                'GCM Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx128': [dn + 'GCM_Ne1_Na20_Nfeatures5+sinfony18_fraeser_ntx128+joint_training' + suf, 'm--*', 128, True],
                                'GCM Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx128': [dn + 'GCM_Ne1_Na20_Nfeatures10+sinfony18_fraeser_ntx128+joint_training' + suf, 'g--*', 128, True],
                                'GCM Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx128': [dn + 'GCM_Ne1_Na20_Nfeatures20+sinfony18_fraeser_ntx128+joint_training' + suf, 'y--*', 128, True],
                                'GCM Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx128': [dn + 'GCM_Ne1_Na20_Nfeatures40+sinfony18_fraeser_ntx128+joint_training' + suf, 'b--*', 128, True],
                                'GCM Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx128': [dn + 'GCM_Ne1_Na20_Nfeatures-1+sinfony18_fraeser_ntx128+joint_training' + suf, 'c--*', 128, True],
                                'GCM diff prob joint training Nalt20 Ne1 + SINFONY ntx128 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable+sinfony18_fraeser_ntx128+joint_training' + suf, 'r-.', 128, True],
                                }
    selected_plots.append(gcm_tools_joint_training)

    # hirise dataset idea for joint results with Psychologists and Sociologists: Deprecated
    # [Unpublished]
    hirise = {'title': ['SINFONY: hirise', 'hirise', 0, False],
              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
              'hirise20 32x32': [dn + 'ResNet20_hirise' + suf, 'g-', 0, True],
              'hirise20 32x32 dist': [dn + 'ResNet20_hirise32_dist_nosnr' + suf, 'g-o', 0, True],
              'hirise20 32x32 dist snr': [dn + 'ResNet20_hirise32_dist' + suf, 'g--', 0, True],
              # 'hirise20 32x32 ntx16': [dn + 'ResNet20_hirise32_ntx16_snr-4_6' + suf, 'g--<', 0, True],
              'hirise32 64x64': [dn + 'ResNet32_hirise64' + suf, 'b-', 0, True],
              'hirise32 64x64 dist': [dn + 'ResNet32_hirise64_dist' + suf, 'b-o', 0, True],
              'hirise32 64x64 ntx32 snr-4_6': [dn + 'ResNet32_hirise64_ntx32_snr-4_6' + suf, 'b--x', 0, True],
              # 'hirise32 64x64 ntx32 snr6_16 rx individual': [dn + 'ResNet32_hirise64_ntx32_rx' + suf, 'b-->', 0, True],
              # 'hirise32 64x64 ntx16 snr6_16 rx individual': [dn + 'ResNet32_hirise64_ntx16_rx' + suf, 'b--<', 0, True],
              # 'hirise32 64x64 ntx16 snr6_16': [dn + 'ResNet32_hirise64_ntx16_snr6_16' + suf, 'r--<', 0, True],
              'hirise32 64x64 ntx16 snr-4_6': [dn + 'ResNet32_hirise64_ntx16' + suf, 'r--o', 0, True],
              # 'hirise56 64x64': [dn + 'ResNet56_hirise64' + suf, 'b-x', 0, True],
              # 'hirise32 128x128': [dn + 'ResNet32_hirise128' + suf, 'r-<', 0, True],
              # 'hirise42 rblock4 64x64': [dn + 'ResNet42_rblock4_hirise64' + suf, 'm-', 0, True],
              # 'hirise52 rblock5 128x128': [dn + 'ResNet52_rblock5_hirise128' + suf, 'm--', 0, True],
              }
    # selected_plots.append(hirise)
    # Only decision between crater or no crater
    hirisecrater = {'title': ['SINFONY: hirisecrater', 'hirise', 0, False],
                    # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                    'hirisecrater20 32x32': [dn + 'ResNet20_hirisecrater32' + suf, 'b-', 0, True],
                    'hirisecrater20 32x32 dist': [dn + 'ResNet20_hirisecrater32_dist' + suf, 'b-o', 0, True],
                    'hirisecrater20 64x64': [dn + 'ResNet20_hirisecrater64' + suf, 'g-', 0, True],
                    'hirisecrater20 64x64 dist': [dn + 'ResNet20_hirisecrater64_dist' + suf, 'g-o', 0, True],
                    # 'hirisecrater32 64x64': [dn + 'ResNet32_hirisecrater64' + suf, 'b-x', 0, True],
                    # 'hirisecrater20 32x32 ntx16 snr-4_6': [dn + 'ResNet20_hirisecrater32_ntx16_snr-4_6' + suf, 'b--<', 0, True],
                    'hirisecrater20 64x64 ntx32 snr-4_6': [dn + 'ResNet20_hirisecrater64_ntx32_snr-4_6' + suf, 'g--<', 0, True],
                    # 'hirisecrater32 64x64 ntx16 snr-4_6': [dn + 'ResNet32_hirisecrater64_ntx16' + suf, 'r--x', 0, True],
                    'hirisecrater20 64x64 ntx16 snr-4_6': [dn + 'ResNet20_hirisecrater64_ntx16_snr-4_6' + suf, 'r-->', 0, True],
                    # 'hirisecrater32 64x64 ntx16 snr6_16': [dn + 'ResNet32_hirisecrater64_ntx16_snr6_16' + suf, 'r--o', 0, True],
                    # 'hirisecrater20 64x64 ntx16 snr6_16': [dn + 'ResNet20_hirisecrater64_ntx16_snr6_16' + suf, 'r--s', 0, True],
                    }
    # selected_plots.append(hirisecrater)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_classic, mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [tools]

    if copy_models:
        plot_sinfony.copy_published_models2repository(selected_plots, datapath=datapath,
                                                      simulation_filename_prefix=filename_prefix, simulation_filename_suffix=suf)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
