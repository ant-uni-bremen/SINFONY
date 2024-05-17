#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: beck
"""

import numpy as np
import scipy.special as sp
import scipy.io as sio
import matplotlib.pyplot as plt
import tikzplotlib as tplt
import os
import myfunctions as mf

# Settings
code = 0        # 0:default, 1:codes
prior = 4       # 0: Gaussian, 1: Uniform, 2/3: Data based, 4: SINFONY data (?)
s_var = 1
N_bit = 16
floatx = 'float' + str(N_bit)
mod = 'BPSK'
y_axis = 'mse'  # ber, fer, ser, ce, mse
x_axis = 'snr'  # ebn0, snr, cebn0
load = mf.savemodule()
path = os.path.join('curves_sem', mod, floatx, 'prior' + str(prior))

# Plot tables
comp_prior1 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    # 'Analog': ['RES_' + 'analog', 'g-x', y_axis, True],
    # 'Analog quant': ['RES_' + 'analog_quant', 'g--x', y_axis, True],
    # 'Analog chuse': ['RES_' + 'analog_chuse', 'g--<', y_axis, True],
    'Hardbit': ['RES_' + 'hardbit', 'b-o', y_axis, True],
    # 'Softbit': ['RES_'+'softbit', 'r-o', y_axis, True],
    # 'Evalbit': ['RES_' + 'evalbit', 'g-o', y_axis, True],
    'DNN snr6_16': ['RES_' + 'DNN', 'm-x', y_axis, True],
    'DNN snr6_6': ['RES_' + 'DNN_snr6_6', 'm--x', y_axis, True],
    'DNN llr snr6_16': ['RES_' + 'DNN_llr', 'm-<', y_axis, True],
    'DNN het': ['RES_' + 'DNNh', 'g-<', y_axis, True],
    # 'DNN het grad': ['RES_' + 'DNNh_grad', 'g--<', y_axis, True],
    'AE': ['RES_' + 'AE', 'k-x', y_axis, True],
    'AE snr-4_6': ['RES_' + 'AE_snr-4_6', 'k--x', y_axis, True],
    # 'AE snr12_12': ['RES_' + 'AE_snr12_12', 'k--', y_axis, True],
    'AE ax1': ['RES_' + 'AE_ax1', 'k--', y_axis, True],
    'AEb': ['RES_' + 'AEb', 'k-o', y_axis, True],
    'AEb ax1': ['RES_' + 'AEb_ax1', 'k-', y_axis, True],
    # 'AE het': ['RES_' + 'AEh', 'g--o', y_axis, True],
    'HardbitNN snr6_16': ['RES_' + 'hardbitNN', 'b--x', y_axis, True],
    'HardbitNN llr snr6_16': ['RES_' + 'hardbitNN_llr', 'b--<', y_axis, True],
    # 'SoftbitNN': ['RES_' + 'softbitNN', 'r--x', y_axis, True],
    # 'EvalbitNN': ['RES_' + 'evalbitNN', 'g--x', y_axis, True],
    'MAPseq': ['RES_' + 'MAPseq', 'y-o', y_axis, True],
    'Meanseq': ['RES_' + 'Meanseq', 'p-', y_axis, True],
    'Meanseq quant': ['RES_' + 'Meanseq_quant', 'p-', y_axis, True],
    # 'MAPseq2': ['RES_' + 'MAPseq_Nb10000it10', 'y-o', y_axis, True],
    # 'Meanseq2': ['RES_' + 'Meanseq_Nb10000it10', 'p-', y_axis, True],
    'MAPseq3': ['RES_' + 'MAPseq_Nb10000it100', 'y-o', y_axis, True],
    'MAPseq3 quant': ['RES_' + 'Meanseq_quant_MAPseq_Nb10000it100', 'y--o', y_axis, True],
    'Meanseq3': ['RES_' + 'Meanseq_Nb10000it100', 'p-', y_axis, True],
    'MAPseq4': ['RES_' + 'MAPseq_Nb100000it100', 'y-o', y_axis, True],
    'MAPseq4 quant': ['RES_' + 'Meanseq_quant_MAPseq_Nb100000it100', 'y--o', y_axis, True],
    'Meanseq4': ['RES_' + 'Meanseq_Nb100000it100', 'p-', y_axis, True],
}

ml_methods = comp_prior1


# Performance curves
plt.figure(1)

res0 = 0
for algo, algo_set in ml_methods.items():
    if algo_set[-1]:
        pathfile = os.path.join(path, algo_set[0])
        res0 = load.load(pathfile, form='npz')
        if res0 is not None:
            res = res0
            if algo_set[2] in res:
                if code == 1:
                    plt.semilogy(res[algo_set[2]],
                                 res[algo_set[3]], algo_set[1], label=algo)
                else:
                    plt.semilogy(res[x_axis], res[algo_set[2]],
                                 algo_set[1], label=algo)

# Analog transmission:
if y_axis == 'mse':
    # NMSE: E[(s-s*)^2/s^2]=E[n^2/s^2]=E[n^2]/E[s^2]
    # E[n^2]= 1 / mf.dbinv(res['snr']) / N_bit, N_bit transmissions assumed
    # E[s^2] as follows:
    # isfin = np.isfinite(floatx.x_poss)
    # Es2 = np.sum(floatx.x_poss[isfin].astype('float64') ** 2 * p_s[isfin])
    # Es = np.sum(floatx.x_poss[isfin] * p_s[isfin])
    # Ess2 = s2_mean - s_mean ** 2
    if prior == 0 and floatx == 'float8':
        Es2 = 0.9998393876037812
    elif prior == 0 and floatx == 'float16':
        Es2 = 1.000097397410417
    elif prior == 1 and floatx == 'float8':
        Es2 = 19223.16190476194
    elif prior == 1 and floatx == 'float16':
        Es2 = 1430258105.4867759
    elif prior == 2 and floatx == 'float8':
        Es2 = 0.06692912634373127
    elif prior == 2 and floatx == 'float16':
        Es2 = 0.06695658838459773
    elif prior == 3 and floatx == 'float8':
        Es2 = 1650.9185033443289
    elif prior == 3 and floatx == 'float16':
        Es2 = 2052347.657767745
    else:
        print('E[s^2] not known. Set E[s^2]=1.')
        Es2 = 1
    mse_analog = Es2 / mf.dbinv(res['snr']) / N_bit
    nmse_analog = mse_analog / Es2
    plt.semilogy(res[x_axis], nmse_analog, 'g-', label='Analog')


if y_axis == 'ber':
    if mod == 'QAM16':
        M = 16
        ber_awgn = 2 / np.log2(M) * (1 - 1 / np.sqrt(M)) * sp.erfc(
            np.sqrt(3 * np.log2(M) / (2 * (M - 1)) * mf.dbinv(res[res['ebn0']])))
    else:
        ber_awgn = 0.5 * sp.erfc(np.sqrt(mf.dbinv(res['ebn0'])))
        plt.semilogy(res[x_axis], ber_awgn, 'g-', label='AWGN')
        # ber[-2, :] = 0.5 * (1 - np.sqrt(mf.dbinv(EbN0) / (1 + mf.dbinv(EbN0))))
        # ber[-1, :] = 1 - norm.cdf(np.sqrt(mf.dbinv(EbN0)))
        # plt.semilogy(EbN0, ber[-2, :], 'g--', label = 'Rayleigh fading')#, label = 'xyz')
        # plt.semilogy(EbN0, ber[-1, :], 'g--', label = 'ML approx')#, label = 'xyz') # ?


# BER settings
# plt.xlim(-6 + snr_shift, 20 + snr_shift)
# plt.ylim(10 ** -6, 1)
# MSE settings
plt.ylim(10 ** -7, 10 ** 7)  # 10 ** 8) # 0
# plt.ylim(10 ** -3, 10 ** 1) # 1
# plt.xlim(0, 14)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

# plt.show() # comment if you want to save with tplt
tplt.save("plots/comsem_" + mod + floatx + 'prior' + str(prior) + ".tikz")
