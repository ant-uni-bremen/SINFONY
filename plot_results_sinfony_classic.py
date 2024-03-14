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


# Huffman encoding gain for cifar features
HUFF_GAIN_CIFAR_FEATURES = 1        # TODO: Not evaluated so far, not needed
# Huffman encoding gain for cifar image data set
HUFF_GAIN_CIFAR_IMAGES = 1.005575740720937
HUFF_GAIN = 1.348970196377917       # Huffman encoding gain on mnist features
HUFF_GAIN2 = 3.5290719114797136     # Huffman encoding gain on mnist image data set


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
    select_plot = False   # Select one plot or plot all preselected plots
    # Fixed
    datapath = 'models'
    dn = 'RES_'

    # Plot tables

    selected_plots = []

    # SINFONY

    # Comparison to classic Shannon design
    # Explaining the abbreviations/numbers:
    # MNIST2 classic rc25 n=15360 h100: 5G LDPC channel code with rate 0.25 and block length 15360, Huffman encoding of feature vectors split into 100 blocks, BPSK [SINFONY - Classic digital comm. -Rc=0.25, BPSK]
    # MNIST2 classic rc25 16-QAM n=15360 h1000: 5G LDPC channel code with rate 0.25 and block length 15360, Huffman encoding of feature vectors split into 100 blocks, 16-QAM [SINFONY - Classic digital comm. -Rc=0.25, 16-QAM]
    # MNIST2 classic rc5 n=16000 h100: 5G LDPC channel code with rate 0.5 and block length 16000, Huffman encoding of feature vectors split into 100 blocks, BPSK [SINFONY - Classic digital comm. -Rc=0.5, BPSK]
    # MNIST image classic rc25 n=15360 h100: Images with 8 bit RGB color entries are transmitted digitally to central unit and joint classification based on overall image [Central - Image transmission -Rc=0.25, BPSK]
    # MNIST2 AE snr-4 6: AE trained for each floating point number [SINFONY - Analog "semantic" AE]
    # MNIST2 AE ntx56 rvec snr-4 6: AE trained for whole feature vector transmission
    # [Published]
    # 56 * 16 / (0.5 * HUFF_GAIN)       # [56 features * 16 floating bits per features] = 896 channel uses per feature vector, 0.5 code rate
    semcom_mnist_classic = {'title': ['SINFONY vs Classic communications on MNIST', '', 56, False],
                            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                            'MNIST1': ['mnist/' + dn + 'ResNet14_MNIST', 'k--', 0, True],
                            'MNIST2': ['mnist/' + dn + 'ResNet14_MNIST2_nosnr', 'b--', 0, True],
                            # 'MNIST2 classic rc5 n=1000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_h1000', 'k--', 56 * 16 / (0.5 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc5 n=1000 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_h100', 'k-x', 56 * 16 / (0.5 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc5 n=16000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc5_n16000_h1000', 'k-*', 56 * 16 / (0.5 * HUFF_GAIN), True],
                            'MNIST2 classic rc5 n=16000 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc5_n16000_h100', 'k--d', 56 * 16 / (0.5 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 n=1000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_h1000', 'r--', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 n=1000 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_h100', 'r-x', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 qam2 n=1000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_qam2_h1000', 'y--', 56 * 16 * 2 / (2 * 0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 qam4 n=1000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_qam4_h1000', 'k--d', 56 * 16 * 2 / (4 * 0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 n=15360 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_n15360_h1000', 'r-o', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 n=15360 h1000 int': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_n15360_h1000_int', 'r-o', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc25 n=15360 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_n15360_h100', 'k-o', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            'MNIST2 classic rc25 n=15360 h100 int': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_n15360_h100_int', 'k-o', 56 * 16 / (0.25 * HUFF_GAIN), True],
                            'MNIST image classic rc25 n=15360 h100': ['classic/' + dn + 'classic_image_' + 'ResNet14_MNIST_rc25_n15360_h100', 'k-x', 28 * 28 * 1 * 8 / (0.25 * HUFF_GAIN2), True],
                            'MNIST2 classic rc25 qam4 n=15360 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc25_n15360_qam4_h100', 'k-*', 56 * 16 * 2 / (4 * 0.25 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc75 n=1000 h1000': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc75_h1000', 'k--<', 56 * 16 / (0.75 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc75 n=1000 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc75_h100', 'k-<', 56 * 16 / (0.75 * HUFF_GAIN), True],
                            # 'MNIST2 classic rc75 n=11264 h100': ['classic/' + dn + 'classic_' + 'ResNet14_MNIST2_rc75_n11264_h100', 'k->', 56 * 16 / (0.75 * HUFF_GAIN), True],
                            # 'MNIST2 AE ntx4 Ne10 snr-4 6': ['classic/' + dn + 'AE_' + 'ResNet14_MNIST2_ntx4_NL8_Ne10_snr-4_6', 'g-x', 56 * 4, True],
                            'MNIST2 AE ntx16 Ne10 snr-4 6': ['classic/' + dn + 'AE_' + 'ResNet14_MNIST2_ntx16_NL32_Ne10_snr-4_6', 'g--x', 56 * 16, True],
                            # 'MNIST2 AE ntx56 NL112 rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx56_NL112_Ne100_snr-4_6', 'm-s', 56, True],
                            # 'MNIST2 AE ntx56 NL56 Ne100 adam SINFONY rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx56_NL56_SINFONY_Ne100_snr-4_6', 'm--s', 56, True],
                            'MNIST2 AE ntx56 nrx56 Ne20 sgdlrs SINFONY rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx56_NL56_SINFONY_sgdlrs_Ne20_snr-4_6', 'm:^', 56, True],
                            # 'MNIST2 AE ntx112 NL224 rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx112_NL224_Ne100_snr-4_6', 'm:s', 56, True],
                            # 'MNIST2 AE ntx56 NL56 rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx56_NL56_Ne100_snr-4_6', 'm--s', 56, True],
                            # 'MNIST2 AE ntx56 NL112 rvec ind snr-4 6': ['classic/' + dn + 'AErvec_ind_' + 'ResNet14_MNIST2_ntx56_NL112_Ne100_snr-4_6', 'm--d', 56, True],
                            # 'MNIST2 AE ntx14 NL28 rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx14_NL28_Ne100_snr-4_6', 'm-D', 14, True],
                            # 'MNIST2 AE ntx14 nrx56 Ne20 sgdlrs SINFONY rvec snr-4 6': ['classic/' + dn + 'AErvec_' + 'ResNet14_MNIST2_ntx14_SINFONY_sgdlrs_Ne20_snr-4_6', 'm:D', 14, True],
                            'MNIST6 ntx56 nrx56 Ne20 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST6_Ne20_snr-4_6', 'r-s', 56, True],
                            # 'MNIST6 ntx56 nrx56 Ne20 layer2 linear rx ind snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST6_Ne20_layer2_rxindilinear_snr-4_6', 'r--s', 56, True],
                            # 'MNIST6 ntx56 nrx56 Ne20 layer2 linear rx snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST6_Ne20_layer2_rxlinear_snr-4_6', 'r:s', 56, True],
                            # 'MNIST6 rx Ne20 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST6_rx_Ne20_snr-4_6', 'r--d', 56, True],
                            # 'MNIST4 ntx14 nrx56 Ne20 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_Ne20_snr-4_6', 'g-D', 14, True],
                            # 'MNIST4 rx Ne20 snr-4 6': ['mnist/' + dn + 'ResNet14_MNIST4_rx_snr-4_6', 'm-d', 14, True],
                            }
    selected_plots.append(semcom_mnist_classic)

    # Investigation for CIFAR dataset
    # [Partly Published]
    semcom_cifar_classic = {'title': ['SINFONY vs Classic communications on CIFAR10', '', 64, False],
                            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                            'CIFAR1': ['cifar10/' + dn + 'ResNet20_CIFAR', 'k--', 0, True],
                            'CIFAR2': ['cifar10/' + dn + 'ResNet20_CIFAR2_nosnr', 'b--', 0, True],
                            'CIFAR6 ntx64 nrx64 snr-4_6': ['cifar10/' + dn + 'ResNet20_CIFAR6_snr-4_6', 'r-s', 64, True],
                            # 'CIFAR2 classic rc25 n=15360 h100 int': ['classic/' + dn + 'classic_' + 'ResNet20_CIFAR2_rc25_n15360_h100_int', 'k-o', 64 * 16 / (0.25 * HUFF_GAIN_CIFAR_FEATURES), True],
                            'CIFAR1 image classic rc25 n=15360 h100': ['classic/' + dn + 'classic_image_' + 'ResNet20_CIFAR_rc25_n15360_h100', 'k-x', 32 * 32 * 3 * 8 / (0.25 * HUFF_GAIN_CIFAR_IMAGES), True],
                            # 'CIFAR1 image classic rc25 n=15360 h1000': ['classic/' + dn + 'classic_image_' + 'ResNet20_CIFAR_rc25_n15360_h1000', 'k--x', 32 * 32 * 3 * 8 / (0.25 * HUFF_GAIN_CIFAR_IMAGES), True],
                            }
    selected_plots.append(semcom_cifar_classic)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # semcom_mnist_classic, semcom_mnist_conv, semcom_cifar_rl, semcom_mnist_sgd_rl
        selected_plots = [semcom_mnist_classic]

    figures = plot_sinfony.plot_results_semcom(selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                               logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
