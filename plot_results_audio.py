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


HUFF_GAIN_SPEECH = 1.4564478668744127
HUFF_GAIN_URBAN = 1.2486879049014932

if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Settings
    y_axis = 'val_acc'   	# val_loss, val_acc, loss, accuracy, val_accuracy, acc, acc_val, rx_loss, rx_val_loss, tx_loss, tx_val_loss
    error_mode = True      # Show classification error instead of accuracy
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

    # SINFONY applied to audio datasets

    # 512 features per subimage, Ntx counted per subimage
    # 16 bit PCM quantization

    # Bits per subimage: 124*129*1*16/4, no color channel in spectrogram, default float16 resolution for classic transmission
    # Entries per audiosample: 16000 \approx 124*129
    speechcmd = {'title': ['SINFONY+GCM: speech commands', 'speechcommands', 512, False],
                 # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                 'resnet18 imagenet': [dn + 'ResNet18_speechcommands_imagenet' + suf, 'k-', 0, True],
                 # 'resnet18 imagenet 2': [dn + 'ResNet18_speechcommands_imagenet_2' + suf, 'k--', 0, True],
                 # 'resnet18 imagenet lrtools': [dn + 'ResNet18_speechcommands_imagenet_lrtools' + suf, 'k-v', 0, True],
                 'resnet34 imagenet': [dn + 'ResNet34_speechcommands_imagenet' + suf, 'k-.', 0, True],
                 # 'resnet34 imagenet 2': [dn + 'ResNet34_speechcommands_imagenet_2' + suf, 'k.', 0, True],
                 'sinfony ntx512 imagenet': [dn + 'sinfony_speechcommands_imagenet' + suf, 'r-o', 512, True],
                 'sinfony ntx64 imagenet': [dn + 'sinfony_speechcommands_imagenet_ntx64' + suf, 'g-o', 64, True],
                 'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_speechcommands_imagenet_ntx64' + suf, 'r--', 64, True],
                 'GCM opt + SINFONY ntx64': [dn + 'opt_GCM+sinfony_speechcommands_imagenet_ntx64' + suf, 'r--x', 64, True],
                 'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_speechcommands_imagenet_ntx64' + suf, 'm--', 64, True],
                 'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_speechcommands_imagenet_ntx64' + suf, 'g--', 64, True],
                 'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_speechcommands_imagenet_ntx64' + suf, 'y--', 64, True],
                 'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_speechcommands_imagenet_ntx64' + suf, 'b--', 64, True],
                 'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64' + suf, 'c--', 64, True],
                 }
    selected_plots.append(speechcmd)

    speechcmd_classic = {'title': ['SINFONY: speech commands classic', '', 512, False],
                         # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                         'resnet18 imagenet': ['speechcommands/' + dn + 'ResNet18_speechcommands_imagenet' + suf, 'k-', 0, True],
                         'sinfony ntx512 imagenet': ['speechcommands/' + dn + 'sinfony_speechcommands_imagenet' + suf, 'r-o', 512, True],
                         'sinfony ntx64 imagenet': ['speechcommands/' + dn + 'sinfony_speechcommands_imagenet_ntx64' + suf, 'g-o', 64, True],
                         'ResNet18 image classic rc25 n=15360 h10': ['classic/' + dn + 'classic_image_' + 'ResNet18_speechcommands_imagenet_rc25_n15360_h10' + suf, 'k-x', 16000 / 4 * 1 * 16 / (0.25 * HUFF_GAIN_SPEECH), True],
                         }
    selected_plots.append(speechcmd_classic)

    # Bits per subimage: 249*257*1*16, no color channel in spectrogram, default float16 resolution for classic transmission
    # Entries per audiosample: 176.4 k -> 64 k \approx 249*257
    urbansound8k = {'title': ['SINFONY+GCM: urbansound8k', 'urbansound8k', 512, False],
                    # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                    # 'resnet18 imagenet valsplit80': [dn + 'ResNet18_urbansound8k_imagenet_valsplit80' + suf, 'k-', 0, True],
                    # 'sinfony ntx512 imagenet valsplit80': [dn + 'sinfony_urbansound8k_imagenet_valsplit80' + suf, 'r-s', 64, True],
                    # 'sinfony ntx64 imagenet valsplit80': [dn + 'sinfony_urbansound8k_imagenet_ntx64_valsplit80' + suf, 'g->', 64, True],
                    'resnet18 imagenet': [dn + 'ResNet18_urbansound8k_imagenet' + suf, 'k--', 0, True],
                    # 'resnet18 imagenet full': [dn + 'ResNet18_urbansound8kfull_imagenet' + suf, 'k--x', 0, True],
                    # 'resnet18 imagenet half': [dn + 'ResNet18_urbansound8khalf_imagenet' + suf, 'k--^', 0, True],
                    # 'resnet18 imagenet 2': [dn + 'ResNet18_urbansound8k_imagenet_2' + suf, 'k--', 0, True],
                    # 'resnet18 imagenet lrtools': [dn + 'ResNet18_urbansound8k_imagenet_lrtools' + suf, 'k-v', 0, True],
                    'resnet34 imagenet': [dn + 'ResNet34_urbansound8k_imagenet' + suf, 'k-.', 0, True],
                    # 'resnet34 imagenet 2': [dn + 'ResNet34_urbansound8k_imagenet_2' + suf, 'k.', 0, True],
                    'sinfony ntx512 imagenet': [dn + 'sinfony_urbansound8k_imagenet' + suf, 'r-o', 512, True],
                    # 'sinfony ntx512 imagenet lrtools': [dn + 'sinfony_urbansound8k_imagenet_lrtools' + suf, 'r--', 512, True],
                    # 'sinfony ntx512 imagenet Ne300': [dn + 'sinfony_urbansound8k_imagenet_Ne300' + suf, 'r-<', 512, True],
                    'sinfony ntx64 imagenet': [dn + 'sinfony_urbansound8k_imagenet_ntx64' + suf, 'g-->', 64, True],
                    # 'sinfony ntx64 imagenet rxjoint': [dn + 'sinfony_urbansound8k_imagenet_ntx64_rxjoint' + suf, 'g--<', 64, True],
                    # 'sinfony ntx64 imagenet rxindlinear': [dn + 'sinfony_urbansound8k_imagenet_ntx64_rxindlinear' + suf, 'g--x', 64, True],
                    # 'sinfony ntx64 imagenet rxindlinear 2': [dn + 'sinfony_urbansound8k_imagenet_ntx64_rxindlinear_2' + suf, 'g--o', 64, True],
                    # 'sinfony ntx64 imagenet wo weight decay': [dn + 'sinfony_urbansound8k_imagenet_ntx64_weightdecay_0' + suf, 'g--^', 64, True],
                    'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r--', 64, True],
                    'GCM test opt + SINFONY ntx64 snr-4 6': [dn + 'opt_GCMtest+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r-*', 64, True],
                    'GCM opt + SINFONY ntx64': [dn + 'opt_GCM+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r--x', 64, True],
                    'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64' + suf, 'm--', 64, True],
                    'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64' + suf, 'g--', 64, True],
                    'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64' + suf, 'y--', 64, True],
                    'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64' + suf, 'b--', 64, True],
                    'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64' + suf, 'c--', 64, True],
                    }
    selected_plots.append(urbansound8k)

    urbansound8k_classic = {'title': ['SINFONY: urbansound8k classic', '', 512, False],
                            # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                            'resnet18 imagenet': ['urbansound8k/' + dn + 'ResNet18_urbansound8k_imagenet' + suf, 'k--', 0, True],
                            'sinfony ntx512 imagenet': ['urbansound8k/' + dn + 'sinfony_urbansound8k_imagenet' + suf, 'r-o', 512, True],
                            'sinfony ntx64 imagenet': ['urbansound8k/' + dn + 'sinfony_urbansound8k_imagenet_ntx64' + suf, 'g-->', 64, True],
                            'ResNet18 image classic rc25 n=15360 h10': ['classic/' + dn + 'classic_image_' + 'ResNet18_urbansound8k_imagenet_rc25_n15360_h10' + suf, 'k-x', 64000 / 4 * 1 * 16 / (0.25 * HUFF_GAIN_URBAN), True],
                            }
    selected_plots.append(urbansound8k_classic)

    speechcmd_joint_training = {'title': ['SINFONY+GCM: speech commands joint training', 'speechcommands', 512, False],
                                # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                'resnet18 imagenet': [dn + 'ResNet18_speechcommands_imagenet' + suf, 'k-', 0, True],
                                'sinfony ntx64 imagenet': [dn + 'sinfony_speechcommands_imagenet_ntx64' + suf, 'g-o', 64, True],
                                'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_speechcommands_imagenet_ntx64' + suf, 'r--', 64, True],
                                # 'GCM opt + SINFONY ntx64': [dn + 'opt_GCM+sinfony_speechcommands_imagenet_ntx64' + suf, 'r--x', 64, True],
                                'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_speechcommands_imagenet_ntx64' + suf, 'm--', 64, True],
                                'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_speechcommands_imagenet_ntx64' + suf, 'g--', 64, True],
                                'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_speechcommands_imagenet_ntx64' + suf, 'y--', 64, True],
                                'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_speechcommands_imagenet_ntx64' + suf, 'b--', 64, True],
                                'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64' + suf, 'c--', 64, True],
                                'GCM prob joint training Nalt20 Ne1 + SINFONY ntx64 snr-4 6': [dn + 'GCM_Ne1_Na20+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'r--*', 64, True],
                                'GCM Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures5+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'm--*', 64, True],
                                'GCM Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures10+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'g--*', 64, True],
                                'GCM Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures20+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'y--*', 64, True],
                                'GCM Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures40+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'b--*', 64, True],
                                'GCM Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures-1+sinfony_speechcommands_imagenet_ntx64+joint_training' + suf, 'c--*', 64, True],
                                }
    selected_plots.append(speechcmd_joint_training)

    urbansound8k_joint_training = {'title': ['SINFONY+GCM: urbansound8k joint training', 'urbansound8k', 512, False],
                                   # 'Tag': ['data name', 'color in plot', channel uses, on/off],
                                   'resnet18 imagenet': [dn + 'ResNet18_urbansound8k_imagenet' + suf, 'k--', 0, True],
                                   'sinfony ntx64 imagenet': [dn + 'sinfony_urbansound8k_imagenet_ntx64' + suf, 'g-->', 64, True],
                                   'GCM prob + SINFONY ntx64 snr-4 6': [dn + 'GCM+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r--', 64, True],
                                   # 'GCM opt + SINFONY ntx64': [dn + 'opt_GCM+sinfony_urbansound8k_imagenet_ntx64' + suf, 'r--x', 64, True],
                                   'GCM Nfeat 5 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64' + suf, 'm--', 64, True],
                                   'GCM Nfeat 10 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64' + suf, 'g--', 64, True],
                                   'GCM Nfeat 20 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64' + suf, 'y--', 64, True],
                                   'GCM Nfeat 40 Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64' + suf, 'b--', 64, True],
                                   'GCM Nfeat max Ne1 + SINFONY ntx64': [dn + 'GCM_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64' + suf, 'c--', 64, True],
                                   'GCM prob joint training Nalt20 Ne1 + SINFONY ntx64 snr-4 6': [dn + 'GCM_Ne1_Na20+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'r--*', 64, True],
                                   'GCM Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'm--*', 64, True],
                                   'GCM Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'g--*', 64, True],
                                   'GCM Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'y--*', 64, True],
                                   'GCM Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'b--*', 64, True],
                                   'GCM Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'c--*', 64, True],
                                   'GCM diff prob joint training Nalt20 Ne1 + SINFONY ntx64 snr-4 6': [dn + 'GCM_Ne1_Na20_differentiable+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'r-.', 64, True],
                                   'GCM diff Nfeat 5 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures5+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'm-.', 64, True],
                                   'GCM diff Nfeat 10 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures10+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'g-.', 64, True],
                                   'GCM diff Nfeat 20 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures20+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'y-.', 64, True],
                                   'GCM diff Nfeat 40 Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures40+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'b-.', 64, True],
                                   'GCM diff Nfeat max Ne1 joint training Nalt20 Ne1 + SINFONY ntx64': [dn + 'GCM_Ne1_Na20_differentiable_Nfeatures-1+sinfony_urbansound8k_imagenet_ntx64+joint_training' + suf, 'c-.', 64, True],
                                   }
    selected_plots.append(urbansound8k_joint_training)

    # Set here one dictionary to be analyzed
    if select_plot is True:
        # mnist_classic, mnist_conv, cifar_rl, mnist_sgd_rl
        selected_plots = [urbansound8k, speechcmd]

    if copy_models:
        plot_sinfony.copy_published_models2repository(
            selected_plots, datapath=datapath, simulation_filename_prefix=filename_prefix)
    else:
        figures = plot_sinfony.plot_results_semcom(selected_plots=selected_plots, x_axis=x_axis, y_axis=y_axis, datapath=datapath,
                                                   logplot=logplot, error_mode=error_mode, x_axis_normalization=x_axis_normalization)
