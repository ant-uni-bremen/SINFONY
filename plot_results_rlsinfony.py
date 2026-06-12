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
import plot_results_sinfony_classic as classic

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
    copy_models = False     # Copy published models to public repository
    # Fixed
    datapath = 'models'
    filename_prefix = ''
    suf = '_results'
    dn = filename_prefix

    # Plot tables

    selected_plots = []

    # RL-SINFONY

    # Final choice of curves for the paper "Model-free Reinforcement Learning of Semantic Communication via Stochastic Gradient Descent"
    # RL: Training via Stochastic Policy Gradient with exploration variance 0.15
    # Unfortunately, training with CIFAR10 did not converge to a minimum with good generalization performance

    # [Published]

    mnist_rl_paper = {'title': ['RL-SINFONY design: MNIST - Paper', '', 56, False],
                      # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                      'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr' + suf, 'g--', 0, True],
                      'MNIST image classic rc25 n=15360 h100': ['classic/' + dn + 'classic_image_' + 'ResNet14_MNIST_rc25_n15360_h100' + suf, 'k-x', 28 * 28 * 1 * 8 / (0.25 * classic.HUFF_GAIN2), True],
                      'MNIST4 adam conv5 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv5' + suf, 'r-s', 14, True],
                      # TODO: Investigate performance/convergence with only one agent?
                      # 'MNIST6 ntx56 Adam snr-4 6 one transmitter': ['mnist/' + dn + 'ResNet14_MNIST6_adam_Ne100_snr-4_6_onetransmitter_test' + suf, 'r--s', 14, True],
                      'MNIST6 ntx56 RL Ne6000 Adam snr-4 6 one transmitter': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL_adam_Ne6000_snr-4_6_onetransmitter_test' + suf, 'b--x', 14, True],
                      'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 1': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv1' + suf, 'b-x', 14, True],
                      }
    selected_plots.append(mnist_rl_paper)

    # [Published]
    cifar_rl_paper = {'title': ['RL-SINFONY design: CIFAR - Paper', '', 64, False],
                      # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                      # 'CIFAR sgdlr[1e-1,1e-2,1e-3][100,150]': ['cifar10/' + dn + 'ResNet20_CIFAR' + suf, 'r-', 0, True],
                      'CIFAR2 sgdlr[1e-1,1e-2,1e-3][100,150] Ne200 snr-4 6': ['cifar10/' + dn + 'ResNet20_CIFAR2_nosnr' + suf, 'g--', 0, True],
                      'CIFAR4 sgdlr[1e-1,1e-2,1e-3][100,150] Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv0' + suf, 'r--s', 16, True],
                      # 'CIFAR4 adam lr1e-3 Nb500 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adam_Ne200_snr-4_6_conv0' + suf, 'g-', 16, True],
                      'CIFAR4 adam lr1e-4 Nb500 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adam_lr1e-4_Ne200_snr-4_6_conv0' + suf, 'r-s', 16, True],
                      # 'CIFAR4 adam lr1e-4 Nb500 Ne1000 snr-4 6 0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adam_lr1e-4_Ne1000_snr-4_6_test' + suf, 'r-s', 16, True],
                      'CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'b-x', 16, True],
                      'CIFAR4 RL sgd Nb128 lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb128_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'b--^', 16, True],
                      # / 4
                      'CIFAR image classic rc25 n=15360 h100': ['classic/' + dn + 'classic_image_' + 'ResNet20_CIFAR_rc25_n15360_h100' + suf, 'k-x', 32 * 32 * 3 * 8 / (0.25 * classic.HUFF_GAIN_CIFAR_IMAGES), True],
                      # 'CIFAR image classic rc25 n=15360 h1000': ['classic/' + dn + 'classic_image_' + 'ResNet20_CIFAR_rc25_n15360_h1000' + suf, 'k--x', 32 * 32 * 3 * 8 / (0.25 * classic.HUFF_GAIN_CIFAR_IMAGES), True],
                      }
    selected_plots.append(cifar_rl_paper)

    # Many trials until we achieved convergence; main reason was that the convergence time as well as the computation time is very long

    # RL-SINFONY: First trials for algorithm (eager execution, batch shuffle) and hyperparameter finetuning (epochs, optimizer, ntx)
    # Note that SNR range is here SNR=[6,16]
    # [Unpublished]
    mnist_rl = {'title': ['RL-SINFONY MNIST: Algorithm and hyperparameter tuning', '', 56, False],
                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                # 'MNIST1': [dn + 'ResNet14_MNIST' + suf, 'g--', 0, True],
                'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr' + suf, 'b--', 0, True],
                'MNIST4 ntx14': ['mnist/' + dn + 'ResNet14_MNIST4' + suf, 'r-<', 14, True],
                # 'MNIST4 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_snr-4_6' + suf, 'r-', 14, True],
                # 'MNIST4 RL Ne100': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL' + suf, 'r-o', 14, True],
                # 'MNIST4 RL Ne200': ['mnist_rl/' + 'ResNet14_MNIST4_RL_Ne200', 'r-o', 14, True],
                # 'MNIST4 RL Ne200 Nefine100 optrx2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_Nefine100' + suf, 'r--<', 14, True],
                # 'MNIST4 RL Ne200 Nefine100': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_Nefine100_2' + suf, 'r-->', 14, True],
                # 'MNIST4 RL Ne200 shuffle': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_shuffle' + suf, 'r--x', 14, True],
                # 'MNIST7 ntx56 RL Ne200 nonoise': ['mnist_rl/' + dn + 'ResNet14_MNIST7_RL_Ne200' + suf, 'y--', 56, True],
                # 'MNIST4 ntx14 RL Ne1000 nonoise': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne1000_nonoise' + suf, 'y-<', 14, True],
                # 'MNIST4 ntx14 RL Ne1000': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne1000' + suf, 'm-x', 14, True],
                # 'MNIST4 ntx14 RL Ne10000': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne10000' + suf, 'b--', 14, True],
                # 'MNIST4 ntx14 RL Ne2000 sh': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne2000' + suf, 'm-', 14, True],
                'MNIST4 ntx14 RL Ne1000 adam shuffle': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne1000_shuffle' + suf, 'b-', 14, True],
                # 'MNIST4 ntx14 RL Ne1000 adam lr schedule': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne1000_lrs' + suf, 'm-<', 14, True],
                # 'MNIST4 ntx14 RL Ne2000 adam lr schedule': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne2000_lrs' + suf, 'm--', 14, True],
                # 'MNIST4 ntx14 RL Ne200 nonoise': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_nonoise' + suf, 'y->', 14, True],
                # 'MNIST4 RL Ne200 adam+sgd': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_adam+sgd' + suf, 'r--x', 14, True],  # hat nichts gebracht, adam und im finetuning sgd
                # 'MNIST4 RL Ne200 rx2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_rx2' + suf, 'g-x', 14, True],
                # 'MNIST4 RL adam2 np': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam2_np' + suf, 'm-', 14, True],
                # 'MNIST4 RL adam2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam2' + suf, 'm-x', 14, True],
                # 'MNIST4 RL Ne200 bs1000': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_bs1000' + suf, 'r-->', 14, True],
                # 'MNIST4 RL Ne200 bs256': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_bs256' + suf, 'r--<', 14, True],
                # 'MNIST4 Ne20 Adam': ['mnist/' + dn + 'ResNet14_MNIST4_adam' + suf, 'r--s', 14, True],
                # 'MNIST4 RL Ne200 SGD': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd' + suf, 'g--', 14, True],
                'MNIST4 RL Ne2000 SGD': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000' + suf, 'g-', 14, True],
                'MNIST4 RL Ne2000 Adam np': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_np' + suf, 'm-', 14, True],
                'MNIST4 RL Ne1000 SGD': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000' + suf, 'k-', 14, True],
                # 'MNIST4 RL Ne200 SGD shuffle': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne200' + suf, 'g--', 14, True],
                # 'MNIST4 RL Ne300 numpy': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne300' + suf, 'r-x', 14, True],
                # 'MNIST4 RL Ne300 tf.function': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne300_2' + suf, 'r--x', 14, True],
                # 'MNIST4 RL Ne200 Tx1step': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Tx1step' + suf, 'r--x', 14, True],
                # 'MNIST4 RL Ne200 Rx5step': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Rx5step' + suf, 'r--', 14, True],
                # 'MNIST6 ntx56 RL deleted': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL' + suf, 'y-o', 56, True],
                # 'MNIST6 ntx56': ['mnist/' + dn + 'ResNet14_MNIST6' + suf, 'y-<', 56, True],
                # 'MNIST6 ntx56 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST6snr-4_6' + suf, 'y-o', 56, True],
                }
    # selected_plots.append(mnist_rl)

    # RL-SINFONY: Investigations on exploration variance: TODO: Just a few iterations... Retry?
    # Note that SNR range is here SNR=[6,16]
    # [Unpublished]
    mnist_rl_pv = {'title': ['RL-SINFONY MNIST: Exploration variance', '', 56, False],
                   # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                   # 'MNIST1': ['mnist/' + dn + 'ResNet14_MNIST' + suf, 'g--', 0, True],
                   'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr' + suf, 'b--', 0, True],
                   'MNIST4 ntx14': ['mnist/' + dn + 'ResNet14_MNIST4' + suf, 'r-<', 14, True],
                   'MNIST4 RL Ne200 pv schedule[0.15, 0.07, 0.03][150, 175]': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_pvars' + suf, 'r-x', 14, True],
                   'MNIST4 RL Ne200 pv02': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_pv02' + suf, 'g-', 14, True],
                   'MNIST4 RL Ne200 pv10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_pv10' + suf, 'g--', 14, True],
                   'MNIST4 RL Ne2000 SGD pv15': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000' + suf, 'g-', 14, True],
                   'MNIST4 RL Ne200 pv20': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_pv20' + suf, 'g-x', 14, True],
                   'MNIST4 RL Ne200 pv25': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_pv25' + suf, 'g--x', 14, True],
                   'MNIST4 RL Ne200 pv30': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_pv30' + suf, 'g-->', 14, True],
                   'MNIST4 RL Ne200 bs1000': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_bs1000' + suf, 'r-->', 14, True],
                   'MNIST4 RL Ne200 bs256': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne200_bs256' + suf, 'r--<', 14, True],
                   }
    # selected_plots.append(mnist_rl_pv)

    # RL-SINFONY: Investigations on performance after final convergence with SGD optimizer, large number of epochs and SNR=[6,16]
    # [Unpublished]
    mnist_rl2 = {'title': ['RL-SINFONY MNIST: Convergence performance for SGD and SNR=[6,16]', '', 56, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 # 'MNIST1': ['mnist/' + dn + 'ResNet14_MNIST' + suf, 'g--', 0, True],
                 'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr' + suf, 'b--', 0, True],
                 # 'MNIST2 snr': ['mnist/' + dn + 'ResNet14_MNIST2' + suf, 'b-', 56, True],
                 'MNIST4 ntx14': ['mnist/' + dn + 'ResNet14_MNIST4' + suf, 'r-<', 14, True],
                 'MNIST4 Ne20 Adam': ['mnist/' + dn + 'ResNet14_MNIST4_adam' + suf, 'm--s', 14, True],
                 # 'MNIST4 Ne100 Adam': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv0' + suf, 'm--o', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lrs 0.01 0.001': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000' + suf, 'g-', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 SGD lrs': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000' + suf, 'k-', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 SGD lr1e-4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000_lr0.0001' + suf, 'k--x', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-4' + suf, 'k--<', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 SGD lrs 1e-3 1e-4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000_lrs' + suf, 'k--s', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 SGD lr1e-4 Nb 128': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000_lr1e-4_Nb128' + suf, 'k--o', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne1000_lr1e-3' + suf, 'k-<', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-3' + suf, 'k--', 14, True],
                 # 'MNIST4 ntx14 RL Ne4000 SGD lr1e-3 Nb128': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne4000_lr1e-3_Nb128' + suf, 'g-', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_2' + suf, 'k-', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_3' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_4' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_5' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_6' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_7' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_8' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_9' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_10' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 11': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_11' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3' + suf, 'k--x', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3_2' + suf, 'k-x', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_snr-4_6' + suf, 'k-o', 14, True],
                 'MNIST4 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_snr-4_6' + suf, 'r-', 14, True],
                 # 'MNIST4 ntx14 RL Ne3000 SGD lrs[0: 1e-3, 1000:1e-4, 2000: 1e-5]': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lrs' + suf, 'k--s', 14, True],
                 # 'MNIST4 ntx14 RL Ne4000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne4000_lr1e-3' + suf, 'r--', 14, True],
                 # 'MNIST4 ntx14 RL Ne5000 SGD lr1e-3 Nb128': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne5000_lr1e-3_Nb128' + suf, 'r--x', 14, True],
                 # 'MNIST4 ntx14 RL Ne5000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne5000_lr1e-3' + suf, 'r--o', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-3 Nb128': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-3_Nb128' + suf, 'k-o', 14, True],
                 # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 Nb128': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_Nb128' + suf, 'y-', 14, True],
                 'MNIST4 ntx14 RL Ne2000 Adam np': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_np' + suf, 'm-', 14, True],
                 # 'MNIST4 ntx14 RL Ne1000 Adam shuffle': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_Ne1000_shuffle' + suf, 'b-', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-3 Nb256 pv07': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-3_Nb256_pv07' + suf, 'g--', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-3 Nb128 pv07': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-3_Nb128_pv07' + suf, 'g--x', 14, True],
                 # 'MNIST4 ntx14 RL Ne2000 SGD lr1e-3 pv30': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne2000_lr1e-3_pv30' + suf, 'g-->', 14, True],
                 }
    # selected_plots.append(mnist_rl2)

    # RL-SINFONY: Investigations on final convergence performance for default range SNR=[-4,6]
    # [Unpublished]
    mnist_rl3 = {'title': ['RL-SINFONY MNIST: Convergence performance for SGD and SNR=[-4,6]', '', 56, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 # 'MNIST1': ['mnist/' + dn + 'ResNet14_MNIST' + suf, 'g--', 0, True],
                 'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr' + suf, 'b--', 0, True],
                 'MNIST2 Adam Ne100': ['mnist/' + dn + 'ResNet14_MNIST2_adam_Ne100' + suf, 'b--x', 0, True],
                 'MNIST4 AE snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_snr-4_6' + suf, 'r-', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_snr-4_6' + suf, 'k-', 14, True],
                 'MNIST4 ntx14 RL Ne2000 Adam snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_snr-4_6' + suf, 'b-x', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_2' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_3' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_4' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_5' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_6' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_7' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_8' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_9' + suf, 'k--', 14, True],
                 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_sgd_Ne3000_snr-4_6_conv10' + suf, 'k--', 14, True],
                 # 'MNIST6 ntx56': ['mnist/' + dn + 'ResNet14_MNIST6' + suf, 'y-<', 56, True],
                 'MNIST6 ntx56 snr-4_6': ['mnist/' + dn + 'ResNet14_MNIST6snr-4_6' + suf, 'y--<', 56, True],
                 'MNIST6 ntx56 RL Ne3000 SGD lr1e-3 snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL_sgd_Ne3000_snr-4_6' + suf, 'y--', 56, True],
                 # 'MNIST6 ntx56 RL Ne3000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL_sgd_Ne3000' + suf, 'y--o', 56, True],
                 # 'MNIST6 ntx56 RL Ne3000 Adam': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL_adam_Ne3000' + suf, 'y--x', 56, True],
                 'MNIST6 ntx56 RL Ne2000 Adam snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST6_RL_adam_Ne2000_snr-4_6' + suf, 'y-x', 56, True],
                 # 'MNIST4 ntx14 RL Ne2000 Adam traject snr-4 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_traject_snr-4_6' + suf, 'm--', 14, True],
                 }
    # selected_plots.append(mnist_rl3)

    # RL-SINFONY: Comparison of SGD's and Adam's convergence performance with multiple tries (for both SNR ranges)
    # [Unpublished]
    mnist_rl_conv = {'title': ['RL-SINFONY MNIST: Comparison SGD and Adam', 'mnist_rl', 56, False],
                     # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_2' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_3' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 4': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_4' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 5': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_5' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_6' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 7': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_7' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 8': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_8' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 9': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_9' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 10': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_10' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 11': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_11' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3' + suf, 'g-', 14, True],
                     # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3_2' + suf, 'g-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_snr-4_6' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 2': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_2' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 3': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_3' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 4': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_4' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 5': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_5' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 6': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_6' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 7': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_7' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 8': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_8' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 9': [dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_9' + suf, 'm-', 14, True],
                     'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 10': [dn + 'ResNet14_MNIST4_sgd_Ne3000_snr-4_6_conv10' + suf, 'm-', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam np 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_np' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 1': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv1' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 2': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv2' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 3': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv3' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 4': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv4' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 5': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv5' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv6' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 7': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv7' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 8': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv8' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 9': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv9' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 10': [dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv10' + suf, 'g--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv0' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 1': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv1' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 2': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv2' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 3': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv3' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 4': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv4' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 5': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv5' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 6': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv6' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 7': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv7' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 8': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv8' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 9': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv9' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL adam Ne3000 pstd=0.15 snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne3000_pstd15_snr-4_6_conv0' + suf, 'm-->', 16, True],
                     'MNIST4 ntx14 RL adam Ne4000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne4000_pstds_snr-4_6_conv0' + suf, 'b--<', 16, True],
                     'MNIST4 ntx14 RL adam Ne6000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': [dn + 'ResNet14_MNIST4_RL_adam_Ne6000_pstds_snr-4_6_conv0' + suf, 'b-->', 16, True],
                     }
    # selected_plots.append(mnist_rl_conv)

    # SINFONY: Comparison of SGD's and Adam's convergence performance with multiple tries (for both SNR ranges)
    # [Unpublished]
    mnist_conv = {'title': ['SINFONY MNIST: Comparison SGD and Adam', 'mnist', 56, False],
                  # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                  'MNIST4 ntx14': [dn + 'ResNet14_MNIST4' + suf, 'r-<', 14, True],
                  'MNIST4 snr-4 6': [dn + 'ResNet14_MNIST4_snr-4_6' + suf, 'r-s', 14, True],
                  # 'MNIST4 conv0': [dn + 'ResNet14_MNIST4_conv0' + suf, 'b-', 14, True],
                  # 'MNIST4 sgdlr conv0': [dn + 'ResNet14_MNIST4_sgdlr_conv0' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv1': [dn + 'ResNet14_MNIST4_sgdlr_conv1' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv2': [dn + 'ResNet14_MNIST4_sgdlr_conv2' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv3': [dn + 'ResNet14_MNIST4_sgdlr_conv3' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv4': [dn + 'ResNet14_MNIST4_sgdlr_conv4' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv5': [dn + 'ResNet14_MNIST4_sgdlr_conv5' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv6': [dn + 'ResNet14_MNIST4_sgdlr_conv6' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv7': [dn + 'ResNet14_MNIST4_sgdlr_conv7' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv8': [dn + 'ResNet14_MNIST4_sgdlr_conv8' + suf, 'g-', 14, True],
                  # 'MNIST4 sgdlr conv9': [dn + 'ResNet14_MNIST4_sgdlr_conv9' + suf, 'g-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv0': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv0' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv1': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv1' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv2': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv2' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv3': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv3' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv4': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv4' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv5': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv5' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv6': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv6' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv7': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv7' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv8': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv8' + suf, 'm-', 14, True],
                  'MNIST4 sgdlr snr-4 6 conv9': [dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv9' + suf, 'm-', 14, True],
                  # 'MNIST4 adam Ne30 conv0': [dn + 'ResNet14_MNIST4_adam_conv0' + suf, 'g-', 14, True],
                  # 'MNIST4 adam conv0': [dn + 'ResNet14_MNIST4_adam_Ne100_conv0' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv1': [dn + 'ResNet14_MNIST4_adam_Ne100_conv1' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv2': [dn + 'ResNet14_MNIST4_adam_Ne100_conv2' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv3': [dn + 'ResNet14_MNIST4_adam_Ne100_conv3' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv4': [dn + 'ResNet14_MNIST4_adam_Ne100_conv4' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv5': [dn + 'ResNet14_MNIST4_adam_Ne100_conv5' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv6': [dn + 'ResNet14_MNIST4_adam_Ne100_conv6' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv7': [dn + 'ResNet14_MNIST4_adam_Ne100_conv7' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv8': [dn + 'ResNet14_MNIST4_adam_Ne100_conv8' + suf, 'g--x', 14, True],
                  # 'MNIST4 adam conv9': [dn + 'ResNet14_MNIST4_adam_Ne100_conv9' + suf, 'g--x', 14, True],
                  'MNIST4 adam conv0 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv0' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv1 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv1' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv2 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv2' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv3 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv3' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv4 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv4' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv5 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv5' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv6 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv6' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv7 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv7' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv8 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv8' + suf, 'm--x', 14, True],
                  'MNIST4 adam conv9 snr-4 6': [dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv9' + suf, 'm--x', 14, True],
                  }
    # selected_plots.append(mnist_conv)

    # Adam: Comparison of SINFONY and RL-SINFONY convergence performance with multiple tries (for both SNR ranges)
    # [Published]
    mnist_adam_rl = {'title': ['MNIST: SINFONY vs. RL-SINFONY with Adam', '', 56, False],
                     # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                     # 'MNIST4 adam Ne30 conv0': ['mnist/' + dn + 'ResNet14_MNIST4_adam_conv0' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv0': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv0' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv1': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv1' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv2': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv2' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv3': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv3' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv4': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv4' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv5': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv5' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv6' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv7': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv7' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv8': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv8' + suf, 'g-', 14, True],
                     # 'MNIST4 adam conv9': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_conv9' + suf, 'g-', 14, True],
                     'MNIST4 adam conv0 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv0' + suf, 'm-', 14, True],
                     'MNIST4 adam conv1 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv1' + suf, 'm-', 14, True],
                     'MNIST4 adam conv2 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv2' + suf, 'm-', 14, True],
                     'MNIST4 adam conv3 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv3' + suf, 'm-', 14, True],
                     'MNIST4 adam conv4 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv4' + suf, 'm-', 14, True],
                     'MNIST4 adam conv5 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv5' + suf, 'm-', 14, True],
                     'MNIST4 adam conv6 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv6' + suf, 'm-', 14, True],
                     'MNIST4 adam conv7 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv7' + suf, 'm-', 14, True],
                     'MNIST4 adam conv8 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv8' + suf, 'm-', 14, True],
                     'MNIST4 adam conv9 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_adam_Ne100_snr-4_6_conv9' + suf, 'm', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 1': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv1' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv2' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv3' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv4' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv5' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv6' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv7' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv8' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv9' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam 10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_conv10' + suf, 'g--x', 14, True],
                     # 'MNIST4 ntx14 RL Ne2000 Adam np 0': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne2000_np' + suf, 'r--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 0': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv0' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 1': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv1' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv2' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv3' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv4' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv5' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv6' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv7' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv8' + suf, 'm--x', 14, True],
                     'MNIST4 ntx14 RL Ne3000 Adam snr-4 6 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne3000_snr-4_6_conv9' + suf, 'm--x', 14, True],
                     # More epochs
                     # 'MNIST4 ntx14 RL Ne4000 Adam snr-4 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne4000_snr-4_6_conv0' + suf, 'b-->', 14, True],
                     # 'MNIST4 ntx14 RL Ne5000 Adam snr-4 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne5000_snr-4_6_conv0' + suf, 'b--', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 0': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv0' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 1': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv1' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv2' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv3' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv4' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv5' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv6' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv7' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv8' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne6000 Adam snr-4 6 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_snr-4_6_conv9' + suf, 'b-', 14, True],
                     'MNIST4 ntx14 RL Ne8000 Adam snr-4 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne8000_snr-4_6_conv0' + suf, 'b--x', 14, True],
                     'MNIST4 ntx14 RL adam Ne6000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne6000_pstds_snr-4_6_conv0' + suf, 'r-x', 16, True],
                     # 'MNIST4 RL adam Ne10000 pstds[0.15, 0.15 ** 2][2000] snr-4 6 0': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_adam_Ne10000_pstds_snr-4_6_conv0' + suf, 'r-', 16, True],
                     # Alternating training with AE
                     # 'MNIST4 adam RLtrain conv0 snr-4 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_adam_Ne200_snr-4_6_RLtrain_conv0' + suf, 'm--', 14, True],
                     }
    # selected_plots.append(mnist_adam_rl)

    # SGD: Comparison of SINFONY and RL-SINFONY convergence performance with multiple tries (for both SNR ranges)
    # [Unpublished]
    mnist_sgd_rl = {'title': ['MNIST: SINFONY vs. RL-SINFONY with SGD', '', 56, False],
                    # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                    'MNIST4 ntx14': ['mnist/' + dn + 'ResNet14_MNIST4' + suf, 'r-<', 14, True],
                    'MNIST4 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_snr-4_6' + suf, 'r-s', 14, True],
                    # 'MNIST4 sgdlr conv0': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv0' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv1': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv1' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv2': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv2' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv3': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv3' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv4': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv4' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv5': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv5' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv6': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv6' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv7': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv7' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv8': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv8' + suf, 'g-', 14, True],
                    # 'MNIST4 sgdlr conv9': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_conv9' + suf, 'g-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv0': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv0' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv1': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv1' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv2': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv2' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv3': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv3' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv4': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv4' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv5': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv5' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv6': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv6' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv7': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv7' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv8': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv8' + suf, 'm-', 14, True],
                    'MNIST4 sgdlr snr-4 6 conv9': ['mnist/' + dn + 'ResNet14_MNIST4_sgdlr_snr-4_6_conv9' + suf, 'm-', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_2' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_3' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_4' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_5' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_6' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_7' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_8' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_9' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_10' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 11': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_11' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3' + suf, 'g--x', 14, True],
                    # 'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 ml3 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_ml3_2' + suf, 'g--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_lr1e-3_snr-4_6' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 2': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_2' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 3': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_3' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 4': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_4' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 5': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_5' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 6': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_6' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 7': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_7' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 8': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_8' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 9': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_sgd_Ne3000_snr-4_6_9' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD lr1e-3 snr-4_6 10': ['mnist_rl/' + dn + 'ResNet14_MNIST4_sgd_Ne3000_snr-4_6_conv10' + suf, 'm--x', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD snr-4_6 test (w/o AE + RL call tf.function)': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_snr-4_6_test' + suf, 'b-', 14, True],
                    'MNIST4 ntx14 RL Ne3000 SGD snr-4_6 test2 (w/o AE tf.function)': ['mnist_rl/' + dn + 'ResNet14_MNIST4_RL_snr-4_6_test2' + suf, 'b--', 14, True],
                    }
    # selected_plots.append(mnist_sgd_rl)

    # Final investigations with CIFAR10: Hyperparameter tuning difficult with a long simulation time...
    # [Published]
    cifar_rl = {'title': ['CIFAR: RL-SINFONY', '', 64, False],
                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                'CIFAR sgdlr[1e-1,1e-2,1e-3][100,150]': ['cifar10/' + dn + 'ResNet20_CIFAR' + suf, 'r-', 16, True],
                # 'CIFAR sgdlr[1e-1,1e-2,1e-3][100,150] test2': ['cifar10/' + dn + 'ResNet20_CIFAR_test2' + suf, 'r--', 16, True],
                # 'CIFAR4 snr-4 6 (default)': ['cifar10/' + dn + 'ResNet20_CIFAR4_snr-4_6' + suf, 'r-x', 16, True],
                'CIFAR4 sgdlr[1e-1,1e-2,1e-3][100,150] Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv0' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv1': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv1' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv2': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv2' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv3': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv3' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv4': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv4' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv5': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv5' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv6': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv6' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv7': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv7' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv8': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv8' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr snr-4 6 conv9': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr_Ne200_snr-4_6_conv9' + suf, 'm-', 16, True],
                # 'CIFAR4 sgdlr2 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgdlr2_Ne200_snr-4_6_conv0' + suf, 'm--', 16, True],
                'CIFAR4 sgd lr1e-3 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgd_Ne200_snr-4_6_conv0' + suf, 'y--', 16, True],
                'CIFAR4 sgd lr1e-2 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_sgd_lr1e-2_Ne200_snr-4_6_conv0' + suf, 'b--', 16, True],
                'CIFAR4 adam lr1e-3 Nb500 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adam_Ne200_snr-4_6_conv0' + suf, 'g-', 16, True],
                # 'CIFAR4 adam amsgrad Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adamamsgrad_Ne200_snr-4_6_conv0' + suf, 'g-x', 16, True],
                'CIFAR4 adam lr1e-4 Nb500 Ne200 snr-4 6 conv0': ['cifar10/' + dn + 'ResNet20_CIFAR4_adam_lr1e-4_Ne200_snr-4_6_conv0' + suf, 'g--x', 16, True],
                # 'CIFAR4 RL sgdlr snr-4 6 conv0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdlr_Ne400_snr-4_6_0' + suf, 'b-', 16, True],
                # 'CIFAR4 RL sgdlr2 snr-4 6 conv0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdlr2_Ne400_snr-4_6_0' + suf, 'b--', 16, True],
                # 'CIFAR4 RL sgd lr1e-2 Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgd_lr1e-2_Ne1000_snr-4_6_0' + suf, 'b-', 16, True],
                # 'CIFAR4 RL sgd lr1e-3 Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgd_lr1e-3_Ne1000_snr-4_6_0' + suf, 'b--<', 16, True],
                # 'CIFAR4 RL sgdlr3[1e-2,1e-3,1e-4][3,500] Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdlr_Ne1000_snr-4_6_0' + suf, 'k--', 16, True],
                # 'CIFAR4 RL sgdlr4[1e-3,1e-4][500] Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdlr4_Ne1000_snr-4_6_0' + suf, 'k--x', 16, True],
                # 'CIFAR4 RL adam Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_Ne1000_snr-4_6_0' + suf, 'g--', 16, True],
                # 'CIFAR4 RL adam Ne3000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_Ne3000_snr-4_6_0' + suf, 'g-->', 16, True],
                'CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'g--<', 16, True],
                # 'CIFAR4 RL adam lr1e-4 Ne100000 snr-4 6 1': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_lr1e-4_Ne100000_snr-4_6_conv1' + suf, 'g-->', 16, True],
                # 'CIFAR4 RL adam steps25 Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_steps25_Ne1000_snr-4_6_0' + suf, 'g--x', 16, True],
                # 'CIFAR4 RL sgd Nb500 Ne1000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb500_Ne1000_snr-4_6_0' + suf, 'k--', 16, True],
                # 'CIFAR4 RL sgd Ne3000 pstd 0.15 snr-4 6 0': [dn + 'ResNet20_CIFAR4_RL_sgd_Ne3000_pstd15_snr-4_6_conv0' + suf, 'k--x', 16, True],
                # 'CIFAR4 RL adam Ne3000 pstd 0.15 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_adam_Ne3000_pstd15_snr-4_6_conv0' + suf, 'g-->', 16, True],
                # 'CIFAR4 RL sgd Nb512 lr1e-3 Ne10000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-3_Ne10000_snr-4_6_conv0' + suf, 'm--', 16, True],
                'CIFAR4 RL sgd Nb512 lr1e-4 Ne10000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne10000_snr-4_6_conv0' + suf, 'm-->', 16, True],
                'CIFAR4 RL sgd Nb512 lr1e-5 Ne10000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-5_Ne10000_snr-4_6_conv0' + suf, 'm--<', 16, True],
                # 'CIFAR4 RL sgd Nb512 lr1e-6 Ne10000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-6_Ne10000_snr-4_6_conv0' + suf, 'm--x', 16, True],
                'CIFAR4 RL sgd Nb512 lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'k--', 16, True],
                # 'CIFAR4 RL sgd Nb512 lrs0 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lrs0_Ne100000_snr-4_6_conv0' + suf, 'k--<', 16, True],
                # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne50000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne50000_snr-4_6_conv0' + suf, 'k-->', 16, True],
                # 'CIFAR4 RL sgd Nb512 lr1e-4 Ne200000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb512_lr1e-4_Ne200000_snr-4_6_conv0' + suf, 'k-', 16, True],
                'CIFAR4 RL sgd Nb64 lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb64_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'k--<', 16, True],
                'CIFAR4 RL sgd Nb128 lr1e-4 Ne100000 snr-4 6 0': ['cifar10_rl/' + dn + 'ResNet20_CIFAR4_RL_sgdNb128_lr1e-4_Ne100000_snr-4_6_conv0' + suf, 'k-->', 16, True],
                }
    # selected_plots.append(cifar_rl)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [cifar_rl]

    if copy_models:
        plot_sinfony.copy_published_models2repository(selected_plots, datapath=datapath,
                                                      simulation_filename_prefix=filename_prefix, simulation_filename_suffix=suf)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
