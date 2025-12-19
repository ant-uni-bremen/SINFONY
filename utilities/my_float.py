#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
from matplotlib import pyplot as plt

import my_math_operations as mop


# Float Toolbox: Functions for manipulation of floating point numbers

def compute_single_bitprob(floatx, p_s):
    '''Computes a-piori prob. for each bit b separately
    INPUT
    floatx: Floating point object
    OUTPUT
    p_s: Bit sequence pmf
    p_ba: Probability of each bit
    '''
    # Compute probability of each bit being one by multiplying each sequence probability with the sequence
    p_b1 = np.sum(p_s[:, np.newaxis] *
                  mop.int2bin(floatx.b_poss, floatx.N_bits), axis=0)
    # Output probability for 0 and 1
    p_ba = np.concatenate(
        ((1 - p_b1)[:, np.newaxis], p_b1[:, np.newaxis]), axis=-1)
    return p_ba


def sequence_prior_data(floatx, data, show=False):
    '''Probabilities p(s) of bit sequences or s computed empirically from data set samples
    INPUT
    floatx: Floating point object
    data: Data set
    show: Show data distribution
    OUTPUT
    p_s: Probabilities for each possible finite floating point value
    '''
    # Empirical probabilities of data set discretized to floatx precision
    # Flatten and discretize data set to floatx precision
    bint, _ = floatx.float2bitint(data)
    # Count sequence integer representation
    b_count = np.bincount(bint)
    p_s = np.zeros(2 ** floatx.N_bits)
    p_s[0:b_count.shape[0]] = b_count
    # Normalize to probabilities
    p_s = p_s / np.sum(p_s)

    # Plot data distribution before and after quantization
    if show == True:
        plt.figure()
        plt.hist(data.flatten(), bins=1000)
        plt.figure()
        plt.plot(floatx.x_poss, p_s, 'r-o', label='p(s)')
    return p_s


class float_toolbox():
    '''Floating point value toolbox class
    '''
    # Class Attribute
    name = 'Floating point object'
    # Initializer / Instance Attributes

    def __init__(self, float_name, mode=0):
        # Floating point type, e.g., float16, float32, float64
        self.float_name = float_name
        self.mode = mode                # Suitable mode for float16<X
        self.N_bits = 0                 # Number of bits
        self.N_sig = 0                  # Number of significand bits w/o implicit bit
        self.N_exp = 0                  # Number of exponent bits
        self.expbias = 0                # Exponent bias
        self.intx = 0                   # Integer resolution
        self.floatx = 0                 # Floating point resolution
        self.b_poss = 0                 # Possible floating point values as bit array
        self.x_poss = 0                 # Possible floating point values
        self.float2params()             # Initialize output variables
        # Generate table of bits and representatives for non-standard floating point values
        # Computational intractable for float32/64
        if not self.float_name == 'float32' and not self.float_name == 'float64':
            self.b_poss, self.x_poss = self.calculate_poss_float()
    # Instance methods

    def float2params(self):
        '''Create floating point parameter set
        OUTPUT
        N_sig: Number of significand/mantissa bits
        N_exp: Number of exponent bits
        expbias: Exponent bias
        '''
        if self.float_name == 'float32':
            self.N_bits = 32
            self.N_sig = 23
            self.N_exp = 8
            self.intx = np.uint32
            self.floatx = np.float32
        elif self.float_name == 'float16':
            self.N_bits = 16
            self.N_sig = 10
            self.N_exp = 5
            self.intx = np.uint16
            self.floatx = np.float16
        elif self.float_name == 'float64':
            self.N_bits = 64
            self.N_sig = 52
            self.N_exp = 11
            self.intx = np.uint64
            self.floatx = np.float64
        elif self.float_name == 'float8':
            self.N_bits = 8
            self.N_sig = 3
            self.N_exp = 4
            self.intx = np.uint16
            self.floatx = np.float16
        elif self.float_name == 'float4':
            self.N_bits = 4
            self.N_sig = 1
            self.N_exp = 2
            self.intx = np.uint16
            self.floatx = np.float16
        else:
            print('Float not available.')
        self.expbias = int(2 ** self.N_exp / 2) - 1
        return self.N_sig, self.N_exp, self.expbias

    def calculate_poss_float(self):
        '''Calculates possible floating point values x_poss with respective bit/integer representation b_poss
        OUTPUT
        b_poss: Array of possible bit sequences
        x_poss: Possible floating point values
        '''
        b_poss = np.arange(0, 2 ** self.N_bits, dtype=self.intx)
        if self.mode == 0 and (self.float_name == 'float16' or self.float_name == 'float32' or self.float_name == 'float64'):
            # Use pre-defined functions for higher resolutions as they are available and much more efficient
            x_poss = b_poss.view(self.floatx)
        else:
            # Use self-written functions for general floating point values
            x_poss = self.comp_bitint2float(b_poss)
        return b_poss, x_poss

    def comp_bitint2float(self, b):
        '''Computes floating point value x in float64 from bit/integer value b
        INPUT
        b: Bit sequence as integer value, e.g., 10 -> 2
        OUTPUT
        x: Floating point value representation
        '''
        b2 = mop.int2bin(b, N=self.N_bits)
        fl = (b2[:, 0], mop.bin2int(b2[:, 1:self.N_exp + 1], dtype=self.intx),
              mop.bin2int(b2[:, self.N_exp + 1:], dtype=self.intx))
        # fl[1] != 0 means implicit bit if exponent != 0
        x = (-1) ** fl[0] * (fl[2] * 2 ** - self.N_sig + self.floatx(1 * (fl[1] != 0))
                             ) * self.floatx(2) ** (fl[1].astype(np.int64) - self.expbias + (fl[1] == 0))
        # fl[1] == 2 ** self.N_exp - 1 (fl[1] == 31 for float16) means NaN, inf/-inf with fl[2] == 0
        x[fl[1] == 2 ** self.N_exp - 1] = np.NaN
        x[np.logical_and((fl[1] == 2 ** self.N_exp - 1),
                         (fl[2] == 0))] = np.array([np.inf, -np.inf])
        return x

    def bitint2float(self, bint):
        '''Tansforms a bit sequence/integer value bint into its floating point value representation x
        INPUT
        bint: Bit sequence as integer value, e.g., [10] -> 2
        OUTPUT
        x: Floating point value representation
        '''
        if self.mode == 0 and (self.float_name == 'float16' or self.float_name == 'float32' or self.float_name == 'float64'):
            # Check whether bint has correct uint resolution
            if bint.dtype == self.intx:
                x = bint.view(self.floatx)
            else:
                x = bint.astype(self.intx).view(self.floatx)
        else:
            x = self.x_poss[bint]
        return x

    def float2bitint(self, x):
        '''Decomposes a floating point value x into its bit-level/integer representation bint
        INPUT
        x: Floating point value
        OUTPUT
        bint: Bit sequence as integer value, e.g., [10] -> 2
        x_quant: Floating point values after quantization onto possible bit sequence
        '''
        if self.mode == 0 and (self.float_name == 'float16' or self.float_name == 'float32' or self.float_name == 'float64'):
            # Built-in functions, more efficient and available for respective resolutions
            # Check whether quantization is necessary
            if x.dtype.name == self.float_name:
                x_quant = x
            else:
                # Difference between +0/-0 in astype
                x_quant = x.astype(self.floatx)
            bint = x_quant.view(self.intx)
        else:
            # Quantization for custom resolutions: (Too complex for >=float16)
            indx = np.zeros(x.shape[0], dtype=int)
            xisfin = np.isfinite(x)
            indx[xisfin] = np.nanargmin(
                np.abs(x[xisfin, np.newaxis] - self.x_poss[np.newaxis, ...]), axis=-1)
            # No difference between +0/-0 -> +0:
            # Catch 0 and differentiate between +0/-0
            indx[xisfin][indx[xisfin] == 0] = np.signbit(
                x[xisfin][indx[xisfin] == 0]) * 2 ** (self.N_bits - 1)
            if x[xisfin].size != x.size:
                # Inf/-Inf exact
                indx[x == np.inf] = np.argmax(
                    x[x == np.inf, np.newaxis] == self.x_poss[np.newaxis, ...], axis=-1)
                indx[x == -np.inf] = np.argmax(x[x == -np.inf, np.newaxis]
                                               == self.x_poss[np.newaxis, ...], axis=-1)
                # Difficult to discriminate NaNs/first one is chosen
                indx[np.isnan(x)] = np.argmax(np.logical_and(np.isnan(x)[np.isnan(
                    x)][..., np.newaxis], np.isnan(self.x_poss)[np.newaxis, ...]), axis=-1)
            bint = self.b_poss[indx]
            x_quant = self.x_poss[indx]
        return bint, x_quant

    def bit2float(self, b):
        '''Tansforms a bit array b into its floating point representation x
        INPUT
        b: Bit array b
        OUTPUT
        x: Floating point value
        '''
        bint = mop.bin2int(b, axis=-1, dtype=self.intx)
        x = self.bitint2float(bint)
        return x

    def float2bit(self, x):
        '''Decomposes a floating point value x into its bit array representation b
        INPUT
        x: Floating point value
        OUTPUT
        b: Bit array b
        '''
        bint, _ = self.float2bitint(x)
        b = mop.int2bin(bint, N=self.N_bits)
        return b

    def float_errorcorrection(self, b, p_b):
        '''Error correction of a floating point bit sequence b, only for use if no non-finite floats occur!
        INPUT
        b: Floating point vector represented as bit matrix 
        p_b: Matrix of posterior probabilities of each bit being 0
        OUTPUT
        b_corr: Error-corrected bit matrix
        '''
        # Error correction for -0
        # Note: There are two zeros -0 and +0, but only +0 is used, since quantization always maps on +0
        # - Possible to exclude -0, 2 representatives for one interval/cluster not necessary
        # - No influence on semantic performance, only certain bit errors are the consequence after receiver processing with bits
        # Define (-0) bit sequence and find -0 bit sequence
        b_zneg = np.concatenate([np.ones(1, dtype=b.dtype), np.zeros(
            self.N_bits - 1, dtype=b.dtype)])[np.newaxis, ...]
        iszneg = np.sum((b != b_zneg), axis=-1) == 0
        b_corr0 = b
        if iszneg.any():
            # Find entry with lowest probability, compare first bit probability being 1 with those of others being 0
            arg_p_b_min = np.argmin(np.concatenate(
                [1 - p_b[iszneg, 0][:, np.newaxis], p_b[iszneg, 1:]], axis=-1), axis=-1)
            # Correct error by flipping bits with lowest probability
            b_corr0[iszneg, arg_p_b_min] == 1 - b[iszneg, arg_p_b_min]

        # One error can only occur in exponent,
        # then exponent is NaN/+/-Inf bit sequence, other values are valid
        bexp = b_corr0[:, 1:self.N_exp + 1]
        isnonfin = np.sum((bexp == np.ones(self.N_exp)), axis=-1) == self.N_exp
        b_corr = b_corr0
        if isnonfin.any():
            p_bexp = p_b[:, 1:self.N_exp + 1]
            # Hard error correction according to most probable error
            bexp[isnonfin, np.argmax(p_bexp[isnonfin, :], axis=-1)] = 0
            b_corr[isnonfin, 1:self.N_exp + 1] = bexp[isnonfin, :]
            # Default error correction -> 0
            # b_corr[isnonfin, :] = np.zero((np.sum(isnonfin), self.N_bits))
        return b_corr


# EOF
