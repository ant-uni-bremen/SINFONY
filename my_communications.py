#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import my_math_operations as mop


# Basic communication channel model: Additive White Gaussian Noise

def awgn(x, sigma, compl=0):
    '''Adds Additive White Gaussian Noise (AWGN) with standard deviations [sigma] to signal [x]
    compl: complex or real-valued
    '''
    if compl == 1:
        n = (np.random.normal(0, sigma) + 1j *
             np.random.normal(0, sigma)) / np.sqrt(2)
    else:
        n = np.random.normal(0, sigma)
    y = x + n
    return y

# Modulator------------------------------------------------------------


class modulation():
    '''Modulation class
    '''
    # Class Attribute
    name = 'Modulation object'
    # Initializer / Instance Attributes

    def __init__(self, mod_name):
        # modulation type, e.g., BPSK, QPSK, QAM16, QAM64, ASK4, ASK8
        self.mod_name = mod_name
        self.m = 0                  # modulation vector
        self.M = 0                  # modulation order
        self.compl = 1              # complex?
        self.mod2vec()              # Initialize output variables
        # a-priori probabilities (default: equal prob.)
        self.alpha = 1 / self.M * np.ones((self.M))
    # Instance methods

    def mod2vec(self):
        '''Create symbol alphabet vector m for for complex/real modulation mod:
            mod: Modulation string
            m: Symbol alphabet vector
            compl: Complex (1) or real (0) modulation
        '''
        if self.mod_name == 'BPSK':
            self.m = np.array([1, -1])
            self.compl = 0
        elif self.mod_name == 'QPSK' or self.mod_name == 'QAM4':
            # with Gray coding
            self.m = np.array([1, -1])
            self.compl = 1
        elif self.mod_name == 'QAM16':
            # with Gray coding
            self.m = np.array([-3, -1, 3, 1])
            self.compl = 1
        elif self.mod_name == 'ASK4':
            # with Gray coding
            self.m = np.array([-3, -1, 3, 1])
            self.compl = 0
        elif self.mod_name == 'QAM64':
            self.m = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
            self.compl = 1
        elif self.mod_name == 'ASK8':
            self.m = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
            self.compl = 0
        # falsch, richtig: 16*16=256
        # elif self.mod_name == 'QAM256':
        #     print('Gray coding einfügen!!!')
        #     self.m = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
        #     self.compl = 1
        # elif self.mod_name == 'ASK16':
        #     print('Gray coding einfügen!!!')
        #     self.m = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
        #     self.compl = 0
        else:
            print('Modulation not available.')
        # normalization # a = 1 / np.mean(self.m ** 2)
        a = np.sqrt(3 / (self.m.shape[0] ** 2 - 1))
        self.m = self.m * a
        # * (self.compl + 1) # effective size of symbol alphabet
        self.M = self.m.shape[0]
        return self.m, self.compl, self.M

    def modulate(self, c, axis=-1):
        '''Modulate code words [c] onto symbol alphabet [m] w.r.t. one dimension [axis]
        INPUT
        c: Input bits
        m: Modulation alphabet
        OUTPUT
        x: Symbols
        '''
        x = modulator(c, self.m, axis)
        return x


def modulator(c, m, axis=-1):
    '''Modulate code words [c] onto symbol alphabet [m] w.r.t. one dimension [axis]
    INPUT
    c: Input bits
    m: Modulation alphabet
    OUTPUT
    x: Symbols
    '''
    cl = mop.bin2int(c, axis)
    x = m[cl]
    return x


# Interface Equalizer / Decoder --------------------------------------------------


def mimo_coding(c, Nt, M, arch):
    '''Encode code words c horizontally or vertically into c2
    INPUT
    c: code words of dim (Nbc, n)
    Nt: Number of transmit symbols
    M: Modulation order
    arch: PAC: Per antenna coding (horizontal) / PSC: Per stream coding (vertical)
    OUTPUT
    c2: MIMO encoding of c of dim (Nbc / n * log2(M), Nt, n / log2(M), log2(M))
    '''
    if arch == 'horiz':
        rest = np.mod(c.shape[-1], np.log2(M))
        if rest != 0:
            c_end = np.random.randint(
                2, size=(c.shape[0], int(np.log2(M) - rest)))
            c0 = np.concatenate((c, c_end), axis=-1)
        else:
            c0 = c
        c1 = c0.reshape((-1, int(Nt * np.log2(M)), c0.shape[-1]))
        c2 = c1.reshape((c1.shape[0], c1.shape[1], int(
            c1.shape[-1] / np.log2(M)), int(np.log2(M))))
    elif arch == 'vert':
        fit2x = Nt * np.log2(M) / c.shape[-1]
        if int(fit2x) >= 1:
            c0 = c.reshape((-1, int(c.shape[-1] * int(fit2x))))
            c_end = np.random.randint(
                2, size=(c0.shape[0], int(Nt * np.log2(M) - c0.shape[-1])))
            c1 = np.concatenate((c0, c_end), axis=-1)
            # expand dims by 1 for same processing
            c2 = c1.reshape(
                (c1.shape[0], int(c1.shape[1] / np.log2(M)), 1, int(np.log2(M))))
        else:
            c_end = np.random.randint(2, size=(c.shape[0], int(
                Nt * np.log2(M) * np.ceil(1 / fit2x) - c.shape[-1])))
            c0 = np.concatenate((c, c_end), axis=-1)
            c1 = c0.reshape(
                (-1, int(np.ceil(1 / fit2x)), int(Nt * np.log2(M))))
            c2 = np.transpose(c1.reshape((c1.shape[0], c1.shape[1], int(
                c1.shape[-1] / np.log2(M)), int(np.log2(M)))), (0, 2, 1, 3))
    else:
        print('Architecture not available.')
        c2 = c
    return c2


def mimo_decoding(llr_c, n, Nt, M, arch):
    '''Decode horizontally or vertically encoded code word llrs [llr_c] of equalizer dimensions back into original [llr_c2]
    INPUT
    llr_c: LLRs of code bits from equalizer of dim (Nb, Nt, log2(M))
    n: code word length
    Nt: Number of transmit symbols
    M: Modulation order
    arch: PAC: Per antenna coding (horizontal) / PSC: Per stream coding (vertical)
    OUTPUT
    llr_c2: Original order of llr_c of dim (Nbc, n)
    '''
    if arch == 'horiz':
        llr_c1 = np.transpose(np.transpose(llr_c, (1, 0, 2)).reshape(
            (llr_c.shape[-2], -1, int(n + np.mod(np.log2(M) - n, np.log2(M))))), (1, 0, 2))
        llr_c2 = llr_c1.reshape((-1, llr_c1.shape[-1]))[:, :n]
    elif arch == 'vert':
        fit2x = Nt * np.log2(M) / n
        if int(fit2x) >= 1:
            llr_c0 = llr_c.reshape((llr_c.shape[0], -1))
            llr_c1 = llr_c0[:, :int(
                llr_c0.shape[-1] - np.mod(llr_c0.shape[-1], n))]
            llr_c2 = llr_c1.reshape((-1, n))
        else:
            llr_c1 = llr_c.reshape(
                (-1, int(Nt * np.log2(M) * np.ceil(1 / fit2x))))
            llr_c2 = llr_c1[:, :n]
    else:
        print('Architecture not available.')
        llr_c2 = llr_c
    return llr_c2


class random_interleaver():
    # Class Attribute
    name = 'Random interleaver'
    # Initializer / Instance Attributes

    def __init__(self):
        self.interleaver = 0
    # Instance methods

    def interleave(self, c):
        '''Interleave code words
        c: Code words
        '''
        i = np.indices(self.interleaver.shape)[0]
        c_perm = c[i, self.interleaver]
        return c_perm

    def deinterleave(self, c_perm):
        '''Deinterleave code words
        c_perm: Permutated / interleaved code words
        '''
        i = np.indices(self.interleaver.shape)[0]
        c = np.zeros(c_perm.shape)
        c[i, self.interleaver] = c_perm
        return c

    def shuffle(self, Nb, n):
        '''Compute new random permutation of code bits c
        Nb: Batch size, number of random permutations
        n: Code word length
        '''
        self.interleaver = np.array(
            [np.random.permutation(n) for _ in range(Nb)])
        return self.interleaver


# Coding theory ------------------------------------------------------


def symprob2llr(p_m, M):
    '''Calculate llrs of bits c from symbol probabilities p_m of modulation alphabet of cardinality M
    p_m: symbol probabilities p_m
    M: Modulation order/ number of symbols
    llr_c: llr of code bit c; ln(p(c=0)/p(c=1)) = ln(p(c=0)/(1-p(c=0)))
    p_c0: probability of code bit c being 0; p(c=0)
    '''
    c_poss = mop.int2bin(np.array(range(0, M)), int(np.log2(M)))
    mask = (c_poss == 0)
    p_c0 = np.sum(p_m[:, :, :, np.newaxis] *
                  mask[np.newaxis, np.newaxis, :], axis=-2)
    llr_c = np.log(p_c0 / (1 - p_c0))
    llr_c = np.clip(llr_c, -1e6, 1e6)   # avoid infinity
    return llr_c, p_c0


def encoder(b, G):
    '''Encodes bitvector b with code given in generator matrix G of coding theory dimensions
    b: bit stream or bit vectors
    c: code bits
    '''
    if len(b.shape) == 1:
        b2 = b[:len(b) - np.mod(len(b), G.shape[0])]
        b2 = np.reshape(b2, (-1, G.shape[0]))
    else:
        # b2 = b.copy()
        b2 = b
    c = np.mod(np.dot(b2, G), 2).astype('int')
    return c


def bp_decoder(llr, H, it, mode):
    '''Soft decoding for given code reflected by parity check matrix H by belief propagation
    llr:    Log-likelihood ratios
    H:      Parity check matrix
    it:     Number of iterations
    mode:   Exact (0, default) and approximate (1) calculation of boxplus
    '''
    bp_out = llr
    cv = 0
    for _ in range(0, it):
        vc = bp_out[:, :, np.newaxis] * H.T[np.newaxis, :, :] - cv
        cv = boxplus(vc, H, mode)
        bp_out = np.sum(cv, axis=-1) + llr

    cr = (np.sign(bp_out) < 0) * 1
    k = np.size(H, 1) - np.size(H, 0)
    br = cr[:, 0: k]
    return bp_out, cr, br


def boxplus(llrs, H, mode):
    '''Calculate boxplus operation of llrs with parity check matrix H
    mode: 0: boxplus / 1: boxplus approximation
    '''
    if mode == 1:
        H_uncon = (H == 0) * 1
        sign = np.prod(np.sign(llrs) + np.expand_dims(H_uncon.T, 0), axis=1,
                       keepdims=True) / np.sign(llrs + H_uncon.T[np.newaxis, :, :])
        mask = (np.ones((H.shape[-1], H.shape[-1])) - np.eye(H.shape[-1]))
        masked_llrs = np.transpose(np.abs(
            llrs)[:, :, :, np.newaxis] * mask[np.newaxis, :, np.newaxis, :], (0, 3, 2, 1))
        masked_llrs2 = np.ma.masked_equal(masked_llrs, 0.0, copy=False)
        mini = np.array(np.min(masked_llrs2, axis=-1)) * \
            H.T[np.newaxis, :, :]  # accurate
        res = sign * mini
    # elif mode == 2:
    #     # alternative more compact but slow implementation: at least 4 times slower
    #     mask = (np.ones((H.shape[-1], H.shape[-1])) - np.eye(H.shape[-1]))
    #     masked_llrs = np.transpose(llrs[: , :, :, np.newaxis] * mask[np.newaxis, :, np.newaxis, :], (0, 3, 2, 1))
    #     masked_llrs2 = np.ma.masked_equal(masked_llrs, 0.0, copy = False)
    #     vc_tanh = np.tanh(np.clip(masked_llrs2 / 2, -1e12, 1e12))
    #     cv = np.array(np.prod(vc_tanh, axis = -1)) * H.T[np.newaxis, :, :]
    #     cv = np.clip(np.array(cv), -1 + 1e-12, 1 - 1e-12)
    #     res = 2 * np.arctanh(cv)
    else:
        H_uncon = (H == 0) * 1
        vc_tanh = np.tanh(np.clip(llrs / 2, -1e12, 1e12))
        vc_tanh_prod = vc_tanh + H_uncon.T[np.newaxis, :, :]
        cv = np.prod(vc_tanh_prod, 1, keepdims=True)
        cv = (cv / vc_tanh_prod) * H.T[np.newaxis, :, :]
        cv = np.clip(np.array(cv), -1 + 1e-12, 1 - 1e-12)
        res = 2 * np.arctanh(cv)
    return res


def hard_decoder(cr, H):
    '''Hard decoding for given code reflected by parity check matrix H
    '''
    # syndrome table
    s_table = H.T
    c = np.copy(cr)
    s = np.mod(np.dot(cr, s_table), 2)
    arg_e = (s[:, np.newaxis] == s_table).all(axis=-1)
    c[arg_e] = np.mod(cr[arg_e] + 1, 2)
    k = H.shape[1] - H.shape[0]
    u = c[:, :k]
    return c, u


def mimoequ_decoding(fr, dataobj, sim_par, mimo_arch, H, decoder, it_dec):
    '''MIMO system: LLR computation + decoding after equalization
    fr: Output probabilities from equalizer
    dataobj: Data generation object with included interleaver
    sim_par: Simulation parameter object
    H: Parity Check Matrix
    mimo_arch: MIMO system architecture ('vert': vertical, 'horiz': horizontal)
    decoder: Decoder type ('syn': syndrome, 'bp': belief propagation)
    it_dec: Number of decoding iterations
    '''
    n = H.shape[1]          # Code word length

    # TODO: a-posteriori, but extrinsic information (a-posteriori / a-priori) required?
    llr_c2, _ = symprob2llr(fr, sim_par.mod.M)
    llr_c_perm = mimo_decoding(llr_c2, n, sim_par.Nt, sim_par.mod.M, mimo_arch)
    llr_c = dataobj.intleav.deinterleave(llr_c_perm)

    if decoder == 'syn':
        c0 = (llr_c < 0) * 1
        [cr, ur] = hard_decoder(c0, H)
        bp_out = []         # Placeholder

    if decoder == 'bp':
        [bp_out, cr, ur] = bp_decoder(llr_c, H, it_dec, 0)

    return bp_out, cr, ur


# EOF
