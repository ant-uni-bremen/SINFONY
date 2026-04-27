#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022).
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
3. E. Beck, H.- Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. doi: 10.48550/arXiv.2412.05103.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import utilities.my_math_operations as mop


def visualize_tsne_embedding(visualized_model, snr_evaluation_test, test_input, test_labels, visualized_dataset):
    '''Visualize t-SNE embedding
    '''
    # TODO: Script so far only works with SINFONY models
    cmap = plt.cm.jet					# define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist,
                          cmap.N)       # create the new map
    # x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    x1 = visualized_model.layers[1](test_input)		# Output of Tx layer
    sigma_tsne = mop.snr2standard_deviation(snr_evaluation_test)
    sigma_test_tsne = np.array([sigma_tsne, sigma_tsne])
    visualized_model.layers[2].set_weights([sigma_test_tsne])
    x2 = visualized_model.layers[2](x1)			# Channel
    x3 = visualized_model.layers[3].layers[0](x2)  # Input layer
    x4 = visualized_model.layers[3].layers[1](x3)  # Rx layer
    x5 = visualized_model.layers[3].layers[2](
        x4)  # Global average pooling layer
    # t-SNE
    # Choose output to cluster and visualize
    # x = x4[:, 0, 0, :]				# Output of Rx layer
    x = x5								# Output after Global average pooling layer, just before softmax layer
    x_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random').fit_transform(x)
    # Plot
    plt.figure(1)
    labels_number = np.argmax(test_labels, axis=-1)
    # Estimated labels
    # labels_number_est = np.argmax(model.predict(testnorm), axis = -1)
    # labels_number = labels_number_est
    plt.scatter(x_embedded[:, 0],
                x_embedded[:, 1], c=labels_number, cmap=cmap)
    custom_lines = []
    for cl in np.unique(labels_number):
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmaplist[int(
            cl * (len(cmaplist) - 1) / (test_labels.shape[-1] - 1))]))
    if visualized_dataset == 'cifar10':
        dlabel = ['0: Airplane', '1: Automobile', '2: Bird', '3: Cat',
                  '4: Deer', '5: Dog', '6: Frog', '7: Horse', '8: Ship', '9: Truck']
    elif visualized_dataset == 'mnist':
        dlabel = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        dlabel = []
        print('No labels available!')
    plt.legend(custom_lines, dlabel, loc='center left',
               bbox_to_anchor=(1, 0.5))
