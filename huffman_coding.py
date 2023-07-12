#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2022

@author: beck
"""

import numpy as np


## Definition of Huffman source coding object

class huffman_coder():
    '''Data object for Huffman source encoding and decoding
    '''
    # Class Attribute
    name = 'Huffman source coding object'
    def __init__(self, symbols, probs = None, data = None):
        '''Initialization
        symbols: Data to be Huffman encoded
        probs: Probability of the data symbols
        data: Data from which probabilities are inferred
        '''
        self.symbols = symbols
        if probs.any() == None:
            self.p_s = self.calculate_probability(data)
        else:
            self.p_s = probs
        self.nodes = None
        self.huff_tree = self.huffman_tree_gen(dict(zip(self.symbols, self.p_s)))
        self.huff_code = self.calculate_codes(self.huff_tree)

    class Node:
        '''A Huffman tree node
        '''
        def __init__(self, prob, symbol, left = None, right = None):
            '''Initialization of Node
            INPUT
            prob: Probability of symbol
            symbol: Symbol
            left: Left node
            right: Right node
            code: Tree direction (0/1)
            '''
            self.prob = prob
            self.symbol = symbol
            self.left = left
            self.right = right
            self.code = None # ''

    def huffman_tree_gen(self, symbol_with_probs):
        '''Huffman tree generator
        INPUT
        symbol_with_probs: Dictionary consisting of symbols and their probabilities
        OUTPUT
        huff_tree: Huffman tree
        '''
        
        symbols = symbol_with_probs.keys()
        nodes = []
        
        # Converting symbols and probabilities into huffman tree nodes
        for symbol in symbols:
            nodes.append(self.Node(symbol_with_probs.get(symbol), [symbol]))
        
        while len(nodes) > 1:
            # Sort all the nodes in ascending order based on their probability
            nodes = sorted(nodes, key=lambda x: x.prob)
            # for node in nodes:  
            #      print(node.symbol, node.prob)
        
            # Pick 2 smallest nodes
            right = nodes[0]
            left = nodes[1]
        
            left.code = 0
            right.code = 1
        
            # Combine the 2 smallest nodes to create new node
            newNode = self.Node(left.prob + right.prob, left.symbol + right.symbol, left, right)
        
            nodes.remove(left)
            nodes.remove(right)
            nodes.append(newNode)

        huff_tree = nodes[0]
        return huff_tree

    def encoding(self, data):
        '''A helper function to obtain the encoded output
        INPUT
        data: Data to be Huffman encoded
        huffman_code: Dictionary of Huffman code
        OUTPUT
        encoding_output: Huffman encoded symbols
        encoded: Encoded Huffman sequence provided as one list
        '''
        huffman_code = self.huff_code
        encoding_output = []
        for c in data:
            # print(coding[c], end = '')
            encoding_output.append(huffman_code[c])
            
        #string = ''.join([str(item) for item in encoding_output])
        encoded = [item for sublist in encoding_output for item in sublist]
        return encoded, encoding_output

    def decoding(self, encoded_data):
        '''Huffman decoding
        INPUT
        encoded_data: Huffman encoded data
        huffman_tree: Huffman tree of encoded data
        OUTPUT
        decoded_output: Huffman decoded stream
        '''
        huffman_tree = self.huff_tree
        tree_head = huffman_tree
        decoded_output = []
        for x in encoded_data:
            if x == 1: # x == '1'
                huffman_tree = huffman_tree.right   
            elif x == 0: # x == '0'
                huffman_tree = huffman_tree.left
            try:
                if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                    pass
            except AttributeError:
                decoded_output.append(huffman_tree.symbol[0])
                huffman_tree = tree_head
            
        # string = ''.join([str(item) for item in decoded_output])
        return decoded_output

    def calculate_codes(self, node, val = [], codes = dict()): # val = ''
        '''Huffman code for current node
        INPUT
        node: Huffman tree node
        val: Current value of Huffman sequence
        codes: Huffman subcode
        OUTPUT
        codes: Huffman subcode
        '''
        # newVal = val + str(node.code)
        if node.code == None:
            newVal = val
        else:
            newVal = val + [node.code]

        if (node.left):
            self.calculate_codes(node.left, newVal, codes)
        if (node.right):
            self.calculate_codes(node.right, newVal, codes)

        if (not node.left and not node.right):
            codes[node.symbol[0]] = newVal
    
        return codes

    def total_gain(self):
        '''Calculate the space difference between compressed and non compressed data
        OUTPUT
        length_uncoded: Space usage before compression (in bits)
        avg_length: Space usage after compression (in bits)
        entropy: Entropy (in bits)
        redundancy: Redundancy (in bits)
        code_rate: Code rate
        '''
        p_s = self.p_s
        huff_code = self.huff_code
        length_uncoded = int(np.log2(p_s.shape[0]))
        avg_length = np.sum([len(value) * p_s[key] for key, value in huff_code.items()])
        entropy = np.sum(- p_s[p_s != 0] * np.log2(p_s[p_s != 0]))
        redundancy = avg_length - entropy
        code_rate = length_uncoded / avg_length
        print("Space usage before compression (in bits):", length_uncoded)
        print("Space usage after compression (in bits):",  avg_length)
        print("Entropy (in bits):",  entropy)
        print("Redundancy (in bits):",  redundancy)
        print("Code rate:",  code_rate)
        return length_uncoded, avg_length, entropy, redundancy, code_rate
    
    def calculate_probability(self, data):
        '''A helper function to calculate the probabilities of symbols in given data
        data: Data whose a-priori probabilities have to be calculated from their frequencies
        symprob: Dictionary of symbols and their probabilities
        '''
        symprob = dict()
        for element in data:
            if symprob.get(element) == None:
                symprob[element] = 1 / data.shape[0]
            else: 
                symprob[element] += 1 / data.shape[0]
        return symprob









""" Test """
if __name__ == '__main__':

	import matplotlib.pyplot as plt
	## Own packages
	import mymathops as mop
	import mycom as com
	from myequ import lin_det_soft
	import myfunc as mf
	import myfloat as mfl
	import SemCom.SemFloat as semfl



	# Test Simulation
	sim = 1                                 # Test simulation with communication channel: 1
	prior = 0 			                    # 0: Gaussian, 1: Uniform (continuous), 2: Data based (gradients), 3: Data based (gradients+models), 4: Semcom, 5: Uniform (per class)
	Nb = 1000 			                    # 100, 1000, 10000/100000
	floatx = mfl.float_toolbox('float4')    # Floating point object: Float4/8/16

	# Data statistics generation
	p_ba, p_s = semfl.compute_prior(floatx, mode = prior)
	b_poss = np.arange(0, 2 ** floatx.N_bits, dtype = floatx.intx)
	# p_s = huffman.calculate_probability(data)
	huffman = huffman_coder(symbols = b_poss, probs = p_s)
	huff_stat = huffman.total_gain()

	# Functional test: Huffman encoding and decoding
	b = np.random.choice(b_poss, p = p_s, size = (Nb))
	b_huffseq, b_huff = huffman.encoding(b)
	b_r = np.array(huffman.decoding(b_huffseq))
	ber = np.sum(b != b_r) / b.shape[0]

	# Huffman over AWGN: Performance evaluation
	if sim == 1:
		it_max = 1000                   # Maximum number of iterations per SNR value
		Nerr_min = 100                  # Mininum number of error to be found per SNR value
		step_size = 1                   # SNR step
		EbN0_range = [-2, 10]           # SNR range
		R_c = huff_stat[-1]             # Code rate
		mod = com.modulation('BPSK')    # Modulation object: BPSK
		mod.alpha = 1 / mod.M * np.ones((int(floatx.N_bits / np.log2(mod.M)), mod.M))
		sim_par = mf.simulation_parameters(1, 1, 1, mod, 1, EbN0_range, rho = 0)
		sim_par.snr_gridcalc(step_size)
		float_data = semfl.float_dataset_gen(Nb, floatx, mod, 0, 0, p_s = p_s, huffman = huffman)
		perf_meas = mf.performance_measures(Nerr_min = Nerr_min, it_max = it_max, sel_crit = 'ber') # 'it'

		PER = []
		per = []

		for ii, snr in enumerate(sim_par.SNR):
			if snr not in perf_meas.SNR:
				while perf_meas.stop_crit():    # Simulate until 1000 errors or stop after it_max iterations

					# Float source + channel
					float_data.snr_min = snr
					float_data.snr_max = snr
					[y, sigma, s, b_huffseq] = float_data()
					z = np.concatenate((1 - b_huffseq[..., np.newaxis], b_huffseq[..., np.newaxis]), axis = -1)
					
					# Float receiver
					x_est = y
					Phi_ee = mop.tvec2diag(sigma ** 2)
					if huffman == 0:
						p_x = lin_det_soft(x_est, Phi_ee, mod.m, p_ba) # p_ba = mod.alpha
					else:
						p_x = lin_det_soft(x_est, Phi_ee, mod.m, np.array([0.5, 0.5])[np.newaxis, :]) # p_ba = mod.alpha
					_, p_b0 = com.symprob2llr(p_x, mod.M)
					p_b = np.concatenate((p_b0, 1 - p_b0), axis = -1)
					b_r = (p_b[..., 0] < 0.5) * 1

					# Huffman decoding
					b_huffdec = mop.int2bin(np.array(huffman.decoding((p_b[..., 0] < 0.5) * 1)), N = floatx.N_bits)
					str2 = ''.join(str(e) for e in (b_huffdec.flatten() * 1))
					str1 = ''.join(str(e) for e in floatx.float2bit(s).flatten())
					# s_r = floatx.bit2float(b_huffdec)

					perf_meas.eval(z, p_b, mod)
					perf_meas.levenshtein_dist(str1, str2)
					# Compares representation of bit sequence, but: different lengths!
					# perf_meas.mse_calc(s, s_r, mode = 0)
					# Packet rate of huffman coding
					per.append(str1 != str2)
					# Output
					perf_meas.err_print()

				# Save only if accuracy high enough after it_max iterations
				[print_str, sv_flag] = perf_meas.err_saveit(sim_par.SNR[ii], sim_par.EbN0[ii], sim_par.EbN0[ii] - 10 * np.log10(R_c))
				print('{}, '.format(len(sim_par.SNR) - ii) + print_str)
				if sv_flag:
					# saveobj.save(pathfile, perf_meas.results()) # Save results to file
					PER.append(np.mean(per))
				per = []


		# Plot
		# BER / Levenshtein SER
		plt.figure(1)
		plt.semilogy(perf_meas.EbN0, perf_meas.BER, label = 'Uncoded BER')
		plt.semilogy(perf_meas.CEbN0, perf_meas.LS, label = 'Huffman LS')
		# plt.semilogy(perf_meas.EbN0, perf_meas.LS)
		plt.ylim(10 ** -5, 1)
		plt.grid(visible = True, which = 'major', color = '#666666', linestyle = '-')
		plt.minorticks_on()
		plt.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.3)
		plt.xlabel('EbN0')
		plt.ylabel('BER/Levenshtein SER (LS)')
		plt.legend()

		# Frame error rate
		plt.figure(2)
		plt.semilogy(perf_meas.EbN0, perf_meas.FER, label = 'Uncoded FER')
		plt.semilogy(perf_meas.EbN0, PER, label = 'Huffman FER')
		plt.ylim(10 ** -5, 1.2)
		plt.grid(visible = True, which = 'major', color = '#666666', linestyle = '-')
		plt.minorticks_on()
		plt.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.3)
		plt.xlabel('EbN0')
		plt.ylabel('FER')
		plt.legend()