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
    # val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
    y_axis = 'val_acc'
    error_mode = False      # Show classification error instead of accuracy
    # (0) w/o SNR normalization, SNR normalization by [number of channel uses/number of features]
    x_axis_normalization = 0
    # (snr) snr value on x axis, (default) index on x axis
    x_axis = 'memory_size'
    logplot = 2             # Logarithmic plot?
    select_plot = False     # Select one plot or plot all preselected plots
    copy_models = False     # Copy published models to public repository
    # Fixed
    datapath = 'models'
    filename_prefix = ''
    suf = '_results_memory'
    dn = filename_prefix

    # Plot tables

    selected_plots = []

    gcm_memory_tools = {'title': ['SINFONY: Human Rover Tools (705, 380, 1)', 'fraeser', 256 + 256, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet18 2': [dn + 'ResNet18_fraeser' + suf, 'k-', 0, True],
                        # 'SINFONY snr-4 6 3': [dn + 'sinfony18_fraeser_lr1e-3_3' + suf, 'k--x', 512, True],
                        # 'SINFONY snr-4 6 ntx128': [dn + 'sinfony18_fraeser_ntx128' + suf, 'k--', 128, True],
                        'GCM prob + SINFONY ntx128 snr-4 6': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf, 'r-', 128, True],
                        # 'GCM opt + SINFONY ntx128': [dn + 'GCM+sinfony18_fraeser_ntx128' + suf + '_opt', 'r--x', 128, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures5+sinfony18_fraeser_ntx128' + suf, 'm-', 128, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures10+sinfony18_fraeser_ntx128' + suf, 'g-', 128, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures20+sinfony18_fraeser_ntx128' + suf, 'y-', 128, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures40+sinfony18_fraeser_ntx128' + suf, 'b-', 128, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx128': [dn + 'GCM_Nfeatures-1+sinfony18_fraeser_ntx128' + suf, 'c-', 128, True],
                        # 'GCM prob + SINFONY ntx512 snr-4 6': [dn + 'GCM+sinfony18_fraeser_lr1e-3_3' + suf, 'r-o', 512, True],
                        # 'GCM opt + SINFONY ntx512': [dn + 'GCM+sinfony18_fraeser_lr1e-3_3' + suf + '_opt', 'r--o', 512, True],
                        # 'GCM snr20 prob + SINFONY ntx128 snr-4 6': [dn + 'GCM_snr20+sinfony18_fraeser_ntx128' + suf, 'r:', 128, True],
                        # 'GCM snr20 opt + SINFONY ntx128': [dn + 'GCM_snr20+sinfony18_fraeser_ntx128' + suf + '_opt', 'r:x', 128, True],
                        # 'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures5+sinfony18_fraeser_ntx128' + suf, 'm:', 128, True],
                        # 'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures10+sinfony18_fraeser_ntx128' + suf, 'g:', 128, True],
                        # 'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures20+sinfony18_fraeser_ntx128' + suf, 'y:', 128, True],
                        # 'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures40+sinfony18_fraeser_ntx128' + suf, 'b:', 128, True],
                        # 'GCM snr20 Nfeat max Ne1 + SINFONY ntx128': [dn + 'GCM_snr20_Nfeatures-1+sinfony18_fraeser_ntx128' + suf, 'c:', 128, True],
                        }
    selected_plots.append(gcm_memory_tools)

    gcm_memory_mnist = {'title': ['SINFONY: GCM MNIST ntx14 Memory', 'mnist', 56, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k-', 0, True],
                        # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k--x', 56, True],
                        # 'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k--', 14, True],
                        'GCM prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'r-', 14, True],
                        # 'GCM opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 14, True],
                        # 'GCM val100 prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_val100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'r-<', 14, True],
                        # 'GCM val100 opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_val100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'r->', 14, True],
                        'GCM snr-4 6 prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_snr-4_6+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k-', 14, True],
                        # 'GCM snr-4 6 opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_snr-4_6+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'k-x', 14, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'm-', 14, True],
                        # 'GCM opt Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'm-x', 14, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'g-', 14, True],
                        # 'GCM opt Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'g-x', 14, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'y-', 14, True],
                        # 'GCM opt Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'y-x', 14, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'b-', 14, True],
                        # 'GCM opt Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'b-x', 14, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'c-', 14, True],
                        # 'GCM opt Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf + '_opt', 'c-x', 14, True],
                        }
    # selected_plots.append(gcm_memory_mnist)

    gcm_memory_mnist_ntx56 = {'title': ['SINFONY: GCM MNIST ntx56 Memory', 'mnist', 56, False],
                              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                              # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human' + suf, 'k-', 0, True],
                              # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'k--', 56, True],
                              # 'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human' + suf, 'k--x', 14, True],
                              'GCM prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'r-', 56, True],
                              # 'GCM opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'r--x', 56, True],
                              'GCM Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'm-', 56, True],
                              # 'GCM opt Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'm-x', 56, True],
                              'GCM Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'g-', 56, True],
                              # 'GCM opt Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'g-x', 56, True],
                              'GCM Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'y-', 56, True],
                              # 'GCM opt Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'y-x', 56, True],
                              'GCM Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b-', 56, True],
                              # 'GCM opt Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'b-x', 56, True],
                              'GCM Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'c-', 56, True],
                              # 'GCM opt Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'c-x', 56, True],
                              #   'GCM snr20 prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'r:', 56, True],
                              #   'GCM snr20 opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf + '_opt', 'r:x', 56, True],
                              #   'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'm:', 56, True],
                              #   'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'g:', 56, True],
                              #   'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'y:', 56, True],
                              #   'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'b:', 56, True],
                              #   'GCM snr20 Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'GCM_snr20_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human' + suf, 'c:', 56, True],
                              }
    selected_plots.append(gcm_memory_mnist_ntx56)

    gcm_memory_cifar = {'title': ['SINFONY: GCM CIFAR10 ntx16', 'cifar10', 64, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k-', 0, True],
                        # 'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k--x', 16, True],
                        # 'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'k--', 64, True],
                        'GCM prob + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'r-', 16, True],
                        # 'GCM opt + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf + '_opt', 'r--x', 16, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures5+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'm-', 16, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures10+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'g-', 16, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures20+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'y-', 16, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures40+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'b-', 16, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'GCM_Nfeatures-1+sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'c-', 16, True],
                        }
    # selected_plots.append(gcm_memory_cifar)

    gcm_memory_cifar_ntx64 = {'title': ['SINFONY: GCM CIFAR10 ntx64', 'cifar10', 64, False],
                              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                              # 'ResNet human': [dn + 'ResNet20_CIFAR_human' + suf, 'k-', 0, True],
                              # 'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human' + suf, 'k--x', 16, True],
                              # 'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'k--', 64, True],
                              'GCM prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'r-', 64, True],
                              # 'GCM opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf + '_opt', 'r--x', 64, True],
                              'GCM Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'm-', 64, True],
                              'GCM Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'g-', 64, True],
                              'GCM Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'y-', 64, True],
                              'GCM Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b-', 64, True],
                              'GCM Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'c-', 64, True],
                              #   'GCM_snr20 prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'r:', 64, True],
                              #   'GCM_snr20 opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf + '_opt', 'r:x', 64, True],
                              #   'GCM_snr20 Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'm:', 64, True],
                              #   'GCM_snr20 Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'g:', 64, True],
                              #   'GCM_snr20 Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'y:', 64, True],
                              #   'GCM_snr20 Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'b:', 64, True],
                              #   'GCM_snr20 Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'GCM_snr20_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human' + suf, 'c:', 64, True],
                              }
    selected_plots.append(gcm_memory_cifar_ntx64)

    gcm_memory_speechcmd = {'title': ['SINFONY: speech commands', 'speechcommands', 512, False],
                            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                            # 'resnet18 imagenet': [dn + 'ResNet18_speechcommands_imagenet' + suf, 'k-', 0, True],
                            # 'sinfony ntx512 imagenet': [dn + 'sinfony_speechcommands_imagenet' + suf, 'k--x', 512, True],
                            # 'sinfony ntx64 imagenet': [dn + 'sinfony_speechcommands_imagenet_ntx64' + suf, 'k--', 64, True],
                            'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_speechcommands_imagenet_ntx64' + suf, 'r-', 64, True],
                            # 'GCM opt + SINFONY ntx64': [dn + 'GCM+sinfony_speechcommands_imagenet_ntx64' + suf + '_opt', 'r--x', 64, True],
                            'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_speechcommands_imagenet_ntx64' + suf, 'm-', 64, True],
                            'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_speechcommands_imagenet_ntx64' + suf, 'g-', 64, True],
                            'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_speechcommands_imagenet_ntx64' + suf, 'y-', 64, True],
                            'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_speechcommands_imagenet_ntx64' + suf, 'b-', 64, True],
                            'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64' + suf, 'c-', 64, True],
                            # 'GCM Nfeat max Ne1 bs16 + SINFONY ntx64': [dn + 'GCM_bs16_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64' + suf, 'c-x', 64, True],
                            }
    selected_plots.append(gcm_memory_speechcmd)

    gcm_memory_urbansound8k = {'title': ['SINFONY: urbansound8k', 'urbansound8k', 512, False],
                               # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                               # 'resnet18 imagenet': [dn + 'ResNet18_urbansound8k_imagenet' + suf, 'k-', 0, True],
                               # 'sinfony ntx512 imagenet': [dn + 'sinfony_urbansound8k_imagenet' + suf, 'k--x', 512, True],
                               # 'sinfony ntx64 imagenet': [dn + 'sinfony_urbansound8k_imagenet_ntx64' + suf, 'k--', 64, True],
                               'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r-', 64, True],
                               # 'GCM opt + SINFONY ntx64': [dn + 'GCM+sinfony_urbansound8k_imagenet_ntx64' + suf + '_opt', 'r--x', 64, True],
                               'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64' + suf, 'm-', 64, True],
                               'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64' + suf, 'g-', 64, True],
                               'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64' + suf, 'y-', 64, True],
                               'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64' + suf, 'b-', 64, True],
                               'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64' + suf, 'c-', 64, True],
                               }
    selected_plots.append(gcm_memory_urbansound8k)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_classic, mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [gcm_memory_mnist]

    if copy_models:
        plot_sinfony.copy_published_models2repository(selected_plots, datapath=datapath,
                                                      simulation_filename_prefix=filename_prefix, simulation_filename_suffix=suf)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
