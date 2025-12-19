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
    filename_prefix = 'RES_'
    if copy_models:
        dn = ''
    else:
        dn = filename_prefix

    # Plot tables

    selected_plots = []

    gcm_memory_mnist = {'title': ['SINFONY: GCM MNIST ntx14 Memory', 'mnist', 56, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human', 'k--x', 0, True],
                        # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'k-', 56, True],
                        # 'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'k--^', 14, True],
                        'GCM prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'r--', 14, True],
                        'GCM opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'r--x', 14, True],
                        # 'GCM val100 prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_val100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'r-<', 14, True],
                        # 'GCM val100 opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_val100+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'r-->', 14, True],
                        'GCM snr-4 6 prob Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr-4_6+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'k--', 14, True],
                        # 'GCM snr-4 6 opt Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_snr-4_6+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'k--x', 14, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'm--', 14, True],
                        # 'GCM opt Nfeat 5 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures5+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'm--x', 14, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'g--', 14, True],
                        # 'GCM opt Nfeat 10 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures10+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'g--x', 14, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'y--', 14, True],
                        # 'GCM opt Nfeat 20 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures20+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'y--x', 14, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'b--', 14, True],
                        # 'GCM opt Nfeat 40 Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures40+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'b--x', 14, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'c--', 14, True],
                        # 'GCM opt Nfeat max Ne1 + SINFONY ntx14 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures-1+sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'c--x', 14, True],
                        }
    selected_plots.append(gcm_memory_mnist)

    gcm_memory_mnist_ntx56 = {'title': ['SINFONY: GCM MNIST ntx56 Memory', 'mnist', 56, False],
                              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                              # 'ResNet Ne20 human': [dn + 'ResNet14_MNIST_Ne20_human', 'k--x', 0, True],
                              # 'SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'k-', 56, True],
                              # 'SINFONY ntx14 nrx56 Ne20 snr-4 6 human': [dn + 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human', 'k--^', 14, True],
                              'GCM prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'r--', 56, True],
                              'GCM opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'r--x', 56, True],
                              'GCM Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'm--', 56, True],
                              # 'GCM opt Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'm--x', 56, True],
                              'GCM Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'g--', 56, True],
                              # 'GCM opt Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'g--x', 56, True],
                              'GCM Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'y--', 56, True],
                              # 'GCM opt Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'y--x', 56, True],
                              'GCM Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'b--', 56, True],
                              # 'GCM opt Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'b--x', 56, True],
                              'GCM Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'c--', 56, True],
                              # 'GCM opt Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'c--x', 56, True],
                              'GCM snr prob Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'r:', 56, True],
                              'GCM snr opt Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_opt_GCM_snr+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'r:x', 56, True],
                              'GCM snr Nfeat 5 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr_Nfeatures5+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'm:', 56, True],
                              'GCM snr Nfeat 10 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr_Nfeatures10+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'g:', 56, True],
                              'GCM snr Nfeat 20 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr_Nfeatures20+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'y:', 56, True],
                              'GCM snr Nfeat 40 Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr_Nfeatures40+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'b:', 56, True],
                              'GCM snr Nfeat max Ne1 + SINFONY ntx56 Ne20 snr-4 6 human': [dn + 'memory_GCM_snr_Nfeatures-1+sinfony14_MNIST_ntx56_Ne20_snr-4_6_human', 'c:', 56, True],
                              }
    selected_plots.append(gcm_memory_mnist_ntx56)

    gcm_memory_cifar = {'title': ['SINFONY: GCM CIFAR10 ntx16', 'cifar10', 64, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet human': [dn + 'ResNet20_CIFAR_human', 'k--x', 0, True],
                        # 'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human', 'k--', 16, True],
                        # 'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human', 'k-', 64, True],
                        'GCM prob + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM+sinfony20_CIFAR_ntx16_snr-4_6_human', 'r--', 16, True],
                        'GCM opt + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_opt_GCM+sinfony20_CIFAR_ntx16_snr-4_6_human', 'r--x', 16, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures5+sinfony20_CIFAR_ntx16_snr-4_6_human', 'm--', 16, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures10+sinfony20_CIFAR_ntx16_snr-4_6_human', 'g--', 16, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures20+sinfony20_CIFAR_ntx16_snr-4_6_human', 'y--', 16, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures40+sinfony20_CIFAR_ntx16_snr-4_6_human', 'b--', 16, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx16 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures-1+sinfony20_CIFAR_ntx16_snr-4_6_human', 'c--', 16, True],
                        }
    selected_plots.append(gcm_memory_cifar)

    gcm_memory_cifar_ntx64 = {'title': ['SINFONY: GCM CIFAR10 ntx64', 'cifar10', 64, False],
                              # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                              # 'ResNet human': [dn + 'ResNet20_CIFAR_human', 'k--x', 0, True],
                              # 'SINFONY ntx16 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx16_snr-4_6_human', 'k--', 16, True],
                              # 'SINFONY ntx64 nrx64 snr-4 6': [dn + 'sinfony20_CIFAR_ntx64_snr-4_6_human', 'k-', 64, True],
                              'GCM prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM+sinfony20_CIFAR_ntx64_snr-4_6_human', 'r--', 64, True],
                              'GCM opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_opt_GCM+sinfony20_CIFAR_ntx64_snr-4_6_human', 'r--x', 64, True],
                              'GCM Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human', 'm--', 64, True],
                              'GCM Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human', 'g--', 64, True],
                              'GCM Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human', 'y--', 64, True],
                              'GCM Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human', 'b--', 64, True],
                              'GCM Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human', 'c--', 64, True],
                              'GCM_snr20 prob + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human', 'r:', 64, True],
                              'GCM_snr20 opt + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_opt_GCM_snr20+sinfony20_CIFAR_ntx64_snr-4_6_human', 'r:x', 64, True],
                              'GCM_snr20 Nfeat 5 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20_Nfeatures5+sinfony20_CIFAR_ntx64_snr-4_6_human', 'm:', 64, True],
                              'GCM_snr20 Nfeat 10 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20_Nfeatures10+sinfony20_CIFAR_ntx64_snr-4_6_human', 'g:', 64, True],
                              'GCM_snr20 Nfeat 20 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20_Nfeatures20+sinfony20_CIFAR_ntx64_snr-4_6_human', 'y:', 64, True],
                              'GCM_snr20 Nfeat 40 Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20_Nfeatures40+sinfony20_CIFAR_ntx64_snr-4_6_human', 'b:', 64, True],
                              'GCM_snr20 Nfeat max Ne1 + SINFONY ntx64 Ne20 snr-4 6': [dn + 'memory_GCM_snr20_Nfeatures-1+sinfony20_CIFAR_ntx64_snr-4_6_human', 'c:', 64, True],
                              }
    selected_plots.append(gcm_memory_cifar_ntx64)

    gcm_memory_tools = {'title': ['SINFONY: Human Rover Tools (705, 380, 1)', 'fraeser', 256+256, False],
                        # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                        # 'ResNet18 2': [dn + 'ResNet18_fraeser', 'k--x', 0, True],
                        # 'SINFONY snr-4 6 3': [dn + 'sinfony18_fraeser_lr1e-3_3', 'k-', 512, True],
                        # 'SINFONY snr-4 6 ntx128': [dn + 'sinfony18_fraeser_ntx128', 'k--', 128, True],
                        'GCM prob + SINFONY ntx128 snr-4 6': [dn + 'memory_GCM+sinfony18_fraeser_ntx128', 'r--', 128, True],
                        'GCM opt + SINFONY ntx128': [dn + 'memory_opt_GCM+sinfony18_fraeser_ntx128', 'r--x', 128, True],
                        'GCM Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_Nfeatures5+sinfony18_fraeser_ntx128', 'm--', 128, True],
                        'GCM Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_Nfeatures10+sinfony18_fraeser_ntx128', 'g--', 128, True],
                        'GCM Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_Nfeatures20+sinfony18_fraeser_ntx128', 'y--', 128, True],
                        'GCM Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_Nfeatures40+sinfony18_fraeser_ntx128', 'b--', 128, True],
                        'GCM Nfeat max Ne1 + SINFONY ntx128': [dn + 'memory_GCM_Nfeatures-1+sinfony18_fraeser_ntx128', 'c--', 128, True],
                        'GCM prob + SINFONY ntx512 snr-4 6': [dn + 'memory_GCM+sinfony18_fraeser_lr1e-3_3', 'k-o', 512, True],
                        'GCM opt + SINFONY ntx512': [dn + 'memory_opt_GCM+sinfony18_fraeser_lr1e-3_3', 'k--o', 512, True],
                        # 'GCM snr20 prob + SINFONY ntx128 snr-4 6': [dn + 'memory_GCM_snr20+sinfony18_fraeser_ntx128', 'r:', 128, True],
                        # 'GCM snr20 opt + SINFONY ntx128': [dn + 'memory_opt_GCM_snr20+sinfony18_fraeser_ntx128', 'r:x', 128, True],
                        # 'GCM snr20 Nfeat 5 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_snr20_Nfeatures5+sinfony18_fraeser_ntx128', 'm:', 128, True],
                        # 'GCM snr20 Nfeat 10 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_snr20_Nfeatures10+sinfony18_fraeser_ntx128', 'g:', 128, True],
                        # 'GCM snr20 Nfeat 20 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_snr20_Nfeatures20+sinfony18_fraeser_ntx128', 'y:', 128, True],
                        # 'GCM snr20 Nfeat 40 Ne1 + SINFONY ntx128': [dn + 'memory_GCM_snr20_Nfeatures40+sinfony18_fraeser_ntx128', 'b:', 128, True],
                        # 'GCM snr20 Nfeat max Ne1 + SINFONY ntx128': [dn + 'memory_GCM_snr20_Nfeatures-1+sinfony18_fraeser_ntx128', 'c:', 128, True],
                        }
    selected_plots.append(gcm_memory_tools)

    gcm_memory_speechcmd = {'title': ['SINFONY: speech commands', 'speechcommands', 512, False],
                            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                            # 'resnet18 imagenet': [dn + 'ResNet18_speechcommands_imagenet', 'k-', 0, True],
                            # 'sinfony ntx512 imagenet': [dn + 'sinfony_speechcommands_imagenet', 'r-o', 512, True],
                            # 'sinfony ntx64 imagenet': [dn + 'sinfony_speechcommands_imagenet_ntx64', 'g-o', 64, True],
                            'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'memory_GCM+sinfony_speechcommands_imagenet_ntx64', 'r--', 64, True],
                            'GCM opt + SINFONY ntx64': [dn + 'memory_opt_GCM+sinfony_speechcommands_imagenet_ntx64', 'r--x', 64, True],
                            'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures5+sinfony_speechcommands_imagenet_ntx64', 'm--', 64, True],
                            'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures10+sinfony_speechcommands_imagenet_ntx64', 'g--', 64, True],
                            'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures20+sinfony_speechcommands_imagenet_ntx64', 'y--', 64, True],
                            'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures40+sinfony_speechcommands_imagenet_ntx64', 'b--', 64, True],
                            'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64', 'c--', 64, True],
                            'GCM Nfeat max Ne1 bs16 + SINFONY ntx64': [dn + 'memory_GCM_bs16_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64', 'c--x', 64, True],
                            }
    selected_plots.append(gcm_memory_speechcmd)

    gcm_memory_urbansound8k = {'title': ['SINFONY: urbansound8k', 'urbansound8k', 512, False],
                               # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                               # 'resnet18 imagenet': [dn + 'ResNet18_urbansound8k_imagenet', 'k--', 0, True],
                               # 'sinfony ntx512 imagenet': [dn + 'sinfony_urbansound8k_imagenet', 'r-o', 512, True],
                               # 'sinfony ntx64 imagenet': [dn + 'sinfony_urbansound8k_imagenet_ntx64', 'g-->', 64, True],
                               'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'memory_GCM+sinfony_urbansound8k_imagenet_ntx64', 'r--', 64, True],
                               'GCM opt + SINFONY ntx64': [dn + 'memory_opt_GCM+sinfony_urbansound8k_imagenet_ntx64', 'r--x', 64, True],
                               'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64', 'm--', 64, True],
                               'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64', 'g--', 64, True],
                               'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64', 'y--', 64, True],
                               'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64', 'b--', 64, True],
                               'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'memory_GCM_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64', 'c--', 64, True],
                               }
    selected_plots.append(gcm_memory_urbansound8k)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_classic, mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [gcm_memory_mnist]

    if copy_models:
        plot_sinfony.copy_published_models2repository(
            selected_plots, datapath=datapath, simulation_filename_prefix=filename_prefix)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
