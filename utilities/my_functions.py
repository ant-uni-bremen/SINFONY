#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import os
import psutil
import json
import numpy as np
from my_math_operations import int2bin
# Only for Levenshtein distance
import jellyfish
# Saving functions
import h5py


# General helpful functions

def get_ram():
    '''Get RAM usage in GB
    '''
    RAM = psutil.Process(os.getpid()).memory_info().rss / 1e9
    return RAM


def print_time(time):
    '''Print time up to day resolution
    INPUT
    time: time in s
    OUTPUT
    time_str: time string
    '''
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    # y, d = divmod(d, 365.25) # problem with 0.25
    print_time = "{}:{:02d}:{:02d}:{:02d}".format(
        int(d), int(h), int(m), int(np.round(s)))
    return print_time


# Saving and filename----------------------------------------------------

class filename_module():
    '''Class responsible for file names
    For CMDNet/MIMO simulations
    '''
    # Class Attribute
    name = 'File name creator'
    # Initializer / Instance Attributes

    def __init__(self, typename, path, algo, fn_ext, sim_set, code_set=0):
        # Inputs
        self.ospath = path
        self.typename = typename
        self.filename = ''
        self.path = ''
        self.pathfile = ''
        self.mod = sim_set['Mod']
        self.Nr = sim_set['Nr']
        self.Nt = sim_set['Nt']
        self.L = sim_set['L']
        self.algoname = algo
        self.fn_ext = fn_ext
        self.code = code_set
        # Initialize
        self.generate_pathfile_MIMO()
    # Instance methods

    def generate_filename_MIMO(self):
        '''Generates file name
        '''
        if self.code:
            self.filename = self.typename + self.algoname + '_' + self.mod + '_{}_{}_{}_'.format(
                self.Nt, self.Nr, self.L) + self.code['code'] + self.code['dec'] + self.code['arch'] + self.fn_ext
        else:
            self.filename = self.typename + self.algoname + '_' + self.mod + \
                '_{}_{}_{}'.format(self.Nt, self.Nr, self.L) + self.fn_ext
        return self.filename

    def generate_path_MIMO(self):
        '''Generates path name
        '''
        self.path = os.path.join(
            self.ospath, self.mod, '{}x{}'.format(self.Nt, self.Nr))  # '/', '\\'
        return self.path

    def generate_pathfile_MIMO(self):
        '''Generates full path and filename
        '''
        self.generate_path_MIMO()
        self.generate_filename_MIMO()
        self.pathfile = os.path.join(self.path, self.filename)
        return self.pathfile


class savemodule():
    '''Class responsible for data saving
    '''
    # Class Attribute
    name = 'Save data'
    # Initializer / Instance Attributes

    def __init__(self, form='npz'):
        # Inputs
        self.json_ending = '.json'
        self.hdf5_ending = '.hdf5'
        self.npz_ending = '.npz'
        self.format = form
        self.list_str = 'el'
    # Instance methods

    def check_path(self, pathfile, verbose=0):
        '''Check for existing path and file, respectively
        '''
        path = os.path.dirname(pathfile)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose == 1:
                print('Created new directory.')
        else:
            if os.path.isfile(pathfile):
                os.remove(pathfile)
                if verbose == 1:
                    print('Deleted existing file.')
        return pathfile

    def save_hdf5(self, pathfile, data, verbose=0):
        '''Save data to hdf5 file
        data: data to be saved in hdf5
        pathfile: path and filename
        '''
        def save_nested2hdf5(f, data, name, list_str='el'):
            '''Save nested data in hdf5 format
            f: hdf5 group
            data: nested data to be saved
            name: name of data to be saved
            '''
            if isinstance(data, dict):
                # Avoid first group at first call?
                # if name:
                #     grp = f.create_group(name)
                # else:
                #     grp = f
                grp = f.create_group(name)
                for key, value in data.items():
                    save_nested2hdf5(grp, value, key, list_str)
            elif isinstance(data, list):
                grp = f.create_group(name)
                if len(data) > 1:
                    # lexicographical ordering requires filling with zeros
                    N_fill = int(np.log10(len(data) - 1) + 1)
                else:
                    N_fill = 1
                for ii, el in enumerate(data):
                    save_nested2hdf5(grp, el, list_str +
                                     str(ii).zfill(N_fill), list_str)
            else:
                f.create_dataset(name, data=data)  # dset =

        pathfile = pathfile + self.hdf5_ending
        self.check_path(pathfile, verbose)
        # Save
        with h5py.File(pathfile, 'w') as f:
            save_nested2hdf5(f, data, 'SavedData', list_str=self.list_str)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_hdf5(self, pathfile):
        '''Save data to hdf5 file
        load_data: data to be loaded from hdf5 file
        pathfile: path and filename
        '''
        def load_nested2hdf5(data, list_str='el'):
            '''Load nested data from hdf5 format
            data: data in hdf5 format
            '''
            if isinstance(data, h5py.Group):
                keylist = list(data.keys())
                # check for list / before check if elements in subgroup
                if keylist and keylist[0][0:len(list_str)] == list_str:
                    dic = []
                    for value in data.values():
                        dic.append(load_nested2hdf5(value, list_str))
                else:
                    dic = {}
                    for key, value in data.items():
                        dic[key] = load_nested2hdf5(value, list_str)
            else:
                # dic[data.name] = data.value
                dic = data.value
            return dic

        pathfile = pathfile + self.hdf5_ending
        load_data = None
        if os.path.isfile(pathfile):
            with h5py.File(pathfile, 'r') as f:
                dic = load_nested2hdf5(f, list_str=self.list_str)
            load_data = dic['SavedData']
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save_json(self, pathfile, data, verbose=0):
        '''Save data to json file
        data: data to be saved in json
        pathfile: path and filename
        '''
        class py2jsonEncoder(json.JSONEncoder):
            '''Encoder for dtype conversion to json
            '''

            def default(self, obj):
                if isinstance(obj, np.float16):
                    return obj.astype('float64')
                if isinstance(obj, np.float32):
                    return obj.astype('float64')
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)

        pathfile = pathfile + self.json_ending
        self.check_path(pathfile, verbose)
        # Save
        with open(pathfile, 'w') as outfile:
            json.dump(data, outfile, indent=4, cls=py2jsonEncoder)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_json(self, pathfile):
        '''Load data from json file
        load_data: data to be loaded from json file
        pathfile: path and filename
        '''
        pathfile = pathfile + self.json_ending
        load_data = None
        if os.path.isfile(pathfile):
            with open(pathfile, 'r') as outfile:
                load_data = json.load(outfile)
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save_npz(self, pathfile, data, verbose=0):
        '''Save data to npz file
        data: data to be saved in npz
        pathfile: path and filename
        '''
        pathfile = pathfile + self.npz_ending
        self.check_path(pathfile, verbose)
        # Save as np.array
        # for key, value in data.items():
        #    if isinstance(value, list):
        #        data[key] = np.array(value)
        np.savez(pathfile, **data)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_npz(self, pathfile):
        '''Load data from npz file
        load_data: data to be loaded from npz file
        pathfile: path and filename
        '''
        pathfile = pathfile + self.npz_ending
        load_data = None
        if os.path.isfile(pathfile):
            load_data = np.load(pathfile)  # , allow_pickle = True)
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save(self, pathfile, data, form=None, verbose=0):
        '''Save data to specified format
        data: data to be saved in specified format
        pathfile: path and filename
        form: Format to be used (npz, hdf5, json)
        '''
        if form == None:
            form = self.format
        if form == 'npz':
            self.save_npz(pathfile, data, verbose=verbose)
        elif form == 'hdf5':
            self.save_hdf5(pathfile, data, verbose=verbose)
        elif form == 'json':
            self.save_json(pathfile, data, verbose=verbose)
        else:
            print('This format is not available.')
        return pathfile

    def load(self, pathfile, form=None):
        '''Load data from file with specified format
        load_data: data to be loaded from file with specified format
        pathfile: path and filename
        form: Format to be used (npz, hdf5, json)
        '''
        if form == None:
            form = self.format
        load_data = None
        if form == 'npz':
            load_data = self.load_npz(pathfile)
        elif form == 'hdf5':
            load_data = self.load_hdf5(pathfile)
        elif form == 'json':
            load_data = self.load_json(pathfile)
        else:
            print('This format is not available.')
        return load_data


# Simulation Tools ---------------------------------------------


class simulation_parameters():
    '''Simulation parameters (for CMDNet)
    EbN0_range: List of [EbN0_min, EbN0_max]
    '''
    # Class Attribute
    name = 'Simulation parameters'
    # Initializer / Instance Attributes

    def __init__(self, Nt, Nr, L, mod, N_batch, EbN0_range, rho=0):
        # Inputs
        self.Nt = Nt    # effective number of transmit antennas, e.g., 4, 8, 16, 32, 64, 128, 256
        self.Nr = Nr    # effective number of receive antennas
        self.L = L      # number of iterations / layer
        self.mod = mod
        self.N_batch = N_batch
        self.EbN0_range = EbN0_range
        self.rho = rho
        # Outputs
        self.SNR_range = 0
        self.EbN0 = 0
        self.SNR = 0
        self.SNRcalc()
    # Instance methods

    def snr_gridcalc(self, step_size):
        self.EbN0 = np.linspace(self.EbN0_range[0], self.EbN0_range[1], int(
            (self.EbN0_range[1] - self.EbN0_range[0]) / step_size) + 1)
        self.SNRcalc()
        return self.EbN0, self.SNR

    def SNRcalc(self):
        snr_shift = self.snr_shiftcalc()
        self.SNR_range = self.EbN0_range + snr_shift
        self.SNR = self.EbN0 + snr_shift
        return self.SNR_range, self.SNR

    def snr_shiftcalc(self):
        # Add factor 2 always...
        snr_shift = 10 * np.log10(2 * np.log2(self.mod.M))
        return snr_shift


class performance_measures():
    # Class Attribute
    name = 'Performance measures'
    # Initializer / Instance Attributes

    def __init__(self, Nerr_min=1000, it_max=-1, sel_crit='ber'):
        self.ber = 0
        self.cber = 0
        self.fer = 0
        self.cfer = 0
        self.ser = 0
        self.ce = 0
        self.mse = 0
        self.ls = 0
        self.SNR = []
        self.EbN0 = []
        self.CEbN0 = []
        self.BER = []
        self.CBER = []
        self.FER = []
        self.CFER = []
        self.SER = []
        self.CE = []
        self.MSE = []
        self.LS = []
        self.N_ber = []
        self.N_cber = []
        self.N_fer = []
        self.N_cfer = []
        self.N_ser = []
        self.cel = []
        self.msel = []
        self.lsl = []
        # error count accuracy
        self.Nerr_min = Nerr_min
        self.it_max = it_max
        self.crit = sel_crit
    # Instance methods

    def err_saveit(self, snr, ebn0, cebn0):
        '''Save or delete currently calculated perf. measures according to save criterion
        '''
        sv_flag = self.save_crit()
        if sv_flag:
            self.err_save(snr, ebn0, cebn0)
            print_str = self.it_print()
        else:
            print_str = self.it_fail_print()
            self.err_del()
        return print_str, sv_flag

    def sel_crit(self):
        '''Select criterion
        '''
        if self.crit == 'ber':
            crit = self.N_ber
        elif self.crit == 'cber':
            crit = self.N_cber
        elif self.crit == 'fer':
            crit = self.N_fer
        elif self.crit == 'cfer':
            crit = self.N_cfer
        elif self.crit == 'ser':
            crit = self.N_ser
        elif self.crit == 'it':
            crit = False
        else:
            crit = False
        return crit

    def save_crit(self):
        '''Compute save criterion
        '''
        crit = self.sel_crit()
        if crit:  # necessary, if maximum number of iterations exceeded since then we have not enough errors counted
            save_crit = np.sum(crit) >= self.Nerr_min
        else:  # if no stopping criterion selected, the number of iterations used as the save criterion -> always true
            save_crit = True
        return save_crit

    def stop_crit(self):
        '''Compute stopping criterion
        '''
        crit = self.sel_crit()
        if crit:  # check if stopping criterion w.r.t. number of errors is fulfilled
            if self.it_max == -1:
                stop_crit = np.sum(crit) < self.Nerr_min  # only # errors
            else:
                stop_crit = np.sum(crit) < self.Nerr_min and (
                    len(crit) < self.it_max)  # errors + maximum number of iterations
        else:  # if no stopping criterion selected, the number of iterations is used as stopping criterion
            stop_crit = (len(self.N_ber) < self.it_max)
        return stop_crit

    def err_save(self, snr, ebn0, cebn0):
        '''Save currently calculated perf. measures + SNR/EbN0/CEbN0
        '''
        self.SNR.append(snr)
        self.EbN0.append(ebn0)
        self.CEbN0.append(cebn0)
        self.BER.append(self.ber)
        self.CBER.append(self.cber)
        self.FER.append(self.fer)
        self.CFER.append(self.cfer)
        self.SER.append(self.ser)
        self.CE.append(self.ce)
        self.MSE.append(self.mse)
        self.LS.append(self.ls)
        # now reset counted errors
        self.ber = 0
        self.cber = 0
        self.fer = 0
        self.cfer = 0
        self.ser = 0
        self.ls = 0
        self.N_ber = []
        self.N_cber = []
        self.N_fer = []
        self.N_cfer = []
        self.N_ser = []
        self.cel = []
        self.msel = []
        self.lsl = []
        return self.BER

    def err_del(self):
        '''Delete currently calculated perf. measures
        '''
        # Reset counted errors
        self.ber = 0
        self.cber = 0
        self.fer = 0
        self.cfer = 0
        self.ser = 0
        self.ls = 0
        self.N_ber = []
        self.N_cber = []
        self.N_fer = []
        self.N_cfer = []
        self.N_ser = []
        self.cel = []
        self.msel = []
        self.lsl = []
        return self.BER

    def eval(self, p_x, q_x, mod):
        '''Calculate all error performance measures and print errors
        p_x: true probability of classes / empirical it is one realization of the classes
        q_x: estimated probility of classes
        '''
        cl_t = np.argmax(p_x, axis=-1)
        cl_r = np.argmax(q_x, axis=-1)
        self.err_calc(cl_t, cl_r, mod)
        self.crossentropy_calc(p_x, q_x)
        # self.mse_calc(s_t, s_r)
        return self.BER

    def err_calc(self, cl_t, cl_r, mod):
        '''Calculate bit, frame and symbol error rate:
        INPUT
        cl_t, cl_r: Class label tensors
        mod: modulation object
        OUTPUT
        N_ber: Number of bit errors
        ber: Bit error rate
        fer: Frame error rate
        ser: Symbol error rate
        '''
        b_t = np.reshape(int2bin(cl_t, int(np.log2(mod.M))),
                         (-1, int(cl_t.shape[-1] * np.log2(mod.M))))
        b_r = np.reshape(int2bin(cl_r, int(np.log2(mod.M))),
                         (-1, int(cl_r.shape[-1] * np.log2(mod.M))))
        self.ber, self.N_ber = self.ber_calc(b_t, b_r)
        self.fer, self.N_fer = self.fer_calc(b_t, b_r)
        self.ser, self.N_ser = self.ser_calc(cl_t, cl_r, mod.compl)
        return self.ber, self.N_ber, self.fer, self.N_fer, self.ser, self.N_ser

    def mse_calc(self, s_t, s_r, mode=0):
        '''Calculate empirical (normalized) mean square error
        s_t: true transmitted symbols
        s_r: estimated received symbols
        mode: normalized (0), unnormalized (1)
        mse: mean square error (also normalized if E[||x||_2^2] = 1 or mode = 0)
        msel: list of calculated mse
        '''
        # normalized w.r.t. #Nt of vector elements
        if mode == 0:
            # normalized (Note: inserting the division into the mean does not attenuate weighting of very high errors since s_r/s_t can be still high...)
            mse_emp = np.mean(np.mean(np.abs(s_t - s_r) ** 2, axis=-1)) / \
                np.mean(np.mean(np.abs(s_t) ** 2, axis=-1))
        else:
            # unnormalized
            mse_emp = np.mean(np.mean(np.abs(s_t - s_r) ** 2, axis=-1))
        self.msel.append(mse_emp)
        self.mse = np.mean(self.msel)
        return self.mse, self.msel

    def levenshtein_dist(self, str_t, str_r):
        '''Calculate Levenshtein distance
        Necessary for comparing sequences of different length, e.g., if there are errors at huffman decoding
        str_t: true transmitted symbols as string
        str_r: estimated received symbols as string
        mse: mean square error (also normalized since E[||x||_2^2] = 1)
        msel: list of calculated mse
        '''
        # normalized w.r.t. str_t
        lsl_rate = jellyfish.levenshtein_distance(str_t, str_r) / len(str_t)
        self.lsl.append(lsl_rate)
        self.ls = np.mean(self.lsl)
        return self.ls, self.lsl

    def crossentropy_calc(self, p_x, q_x, axis=-1):
        '''Calculate cross entropy
        p_x: true probability of classes / empirical it is one realization of the classes
        q_x: estimated probility of classes
        ce: cross entropy
        cel: list of calculated cross entropy
        '''
        # Avoid np.log(0)
        epsilon = 1e-07  # fuzz factor like in keras
        q_x = np.clip(q_x, epsilon, 1)  # 1. - epsilon ?
        # q factors along x -> sum along Nt-dim would be correct
        ce_emp = np.mean(
            np.mean(np.sum(-p_x * np.log(q_x), axis=axis), axis=-1))
        # Calculation with tensorflow
        # ce_emp = np.mean(np.mean(KB.eval(KB.categorical_crossentropy(KB.constant(p_x), KB.constant(q_x), axis = 1)), axis = -1))
        self.cel.append(ce_emp)
        self.ce = np.mean(self.cel)
        return self.ce, self.cel

    def Nber_calc(self, b_t, b_r):
        '''Calculate number of bit errors:
        b_t, b_r: Bit tensors
        Nber: Number of bit errors
        '''
        Nber = np.sum((b_t != b_r).flatten() * 1)
        return Nber

    def Nfer_calc(self, b_t, b_r):
        '''Calculate number of frame errors:
        b_t, b_r: Bit tensors
        Nfer: Number of frame errors
        '''
        Nfer = np.sum((np.sum((b_t != b_r) * 1, axis=-1) != 0) * 1)
        return Nfer

    def ber_calc(self, b_t, b_r):
        '''Calculate uncoded bit error rate:
        b_t, b_r: Bit tensors
        N_ber: Number of uncoded bit errors
        ber: bit error rate
        '''
        N_err = self.Nber_calc(b_t, b_r)
        self.N_ber.append(N_err)
        self.ber = np.mean(self.N_ber) / b_t.size
        return self.ber, self.N_ber

    def cber_calc(self, b_t, b_r):
        '''Calculate coded bit error rate:
        b_t, b_r: Bit tensors
        N_ber: Number of code bit errors
        ber: bit error rate
        '''
        N_err = self.Nber_calc(b_t, b_r)
        self.N_cber.append(N_err)
        self.cber = np.mean(self.N_cber) / b_t.size
        return self.cber, self.N_cber

    def fer_calc(self, b_t, b_r):
        '''Calculate uncoded frame error rate:
        b_t, b_r: Bit tensors
        N_fer: Number of frame errors
        fer: Frame error rate
        '''
        N_err = self.Nfer_calc(b_t, b_r)
        self.N_fer.append(N_err)
        self.fer = np.mean(self.N_fer) / b_r.shape[0]
        return self.fer, self.N_fer

    def cfer_calc(self, b_t, b_r):
        '''Calculate coded frame error rate:
        b_t, b_r: Bit tensors
        N_fer: Number of frame errors
        fer: Frame error rate
        '''
        N_err = self.Nfer_calc(b_t, b_r)
        self.N_cfer.append(N_err)
        self.cfer = np.mean(self.N_cfer) / b_r.shape[0]
        return self.cfer, self.N_cfer

    def ser_calc(self, cl_t, cl_r, compl):
        '''Calculate symbol error rate:
        cl_t, cl_r: Class label tensors
        compl: Complex modulation or not
        N_ser: Number of symbol errors
        ser: Symbol error rate
        '''
        if compl == 1:
            cl_comp = (cl_t != cl_r) * 1
            cl_compl = cl_comp[:, :cl_t.shape[1] // 2] + \
                cl_comp[:, cl_t.shape[1] // 2:]
            N_err = np.sum((cl_compl != 0).flatten())
            self.N_ser.append(N_err)
            self.ser = np.mean(self.N_ser) / cl_compl.size
        else:
            N_err = np.sum(((cl_t != cl_r).flatten() * 1))
            self.N_ser.append(N_err)
            self.ser = np.mean(self.N_ser) / cl_r.size
        return self.ser, self.N_ser

    def results(self, sort=1):
        '''Create dictionary of results
        '''
        # Sort results before saving
        if sort == 0:
            [ebn0, cebn0, snr, ber, cber, ser, fer, cfer, ce, mse, ls] = [self.EbN0, self.CEbN0,
                                                                          self.SNR, self.BER, self.CBER, self.SER, self.FER, self.CFER, self.CE, self.MSE, self.LS]
        else:
            tuples = zip(self.EbN0, self.CEbN0, self.SNR, self.BER, self.CBER,
                         self.SER, self.FER, self.CFER, self.CE, self.MSE, self.LS)
            [ebn0, cebn0, snr, ber, cber, ser, fer, cfer, ce, mse, ls] = map(
                list, zip(*sorted(tuples, reverse=False)))
        # TODO: make strings customizable
        results = {
            "ebn0": ebn0,
            "cebn0": cebn0,
            "snr": snr,
            "ber": ber,
            "cber": cber,
            "ser": ser,
            "fer": fer,
            "cfer": cfer,
            "ce": ce,
            "mse": mse,
            "ls": ls,
        }
        return results

    def load_results(self, results):
        '''Load results from dictionary to object
        results: dictionary of results
        '''
        def npz2dict2(npz_file):
            '''Converts npz_file object with object type arrays to dictionary with lists
            '''
            hdict = {}
            for key, value in npz_file.items():
                if isinstance(value, np.ndarray):
                    hdict[key] = value.tolist()
            return hdict

        if results != None:
            # If .npz-file and array, convert back to list
            if isinstance(results, np.lib.npyio.NpzFile):
                results = npz2dict2(results)
            self.EbN0 = results['ebn0']
            self.CEbN0 = results['cebn0']
            self.SNR = results['snr']
            self.BER = results['ber']
            if 'cber' in results:   # for compability with old saves
                self.CBER = results['cber']
            self.SER = results['ser']
            self.FER = results['fer']
            if 'cfer' in results:
                self.CFER = results['cfer']
            self.CE = results['ce']
            self.MSE = results['mse']
            if 'ls' in results:
                self.LS = results['ls']
        return self

    def err_print(self):
        '''Prints selected error count for current iteration
        '''
        if self.crit == 'it':  # if iteration as stopping criterion, fall back to default output mse
            if self.msel:
                print_str = print('it: {}, error: {}'.format(
                    len(self.msel), self.mse))
            else:  # self.N_ber as alternative
                print_str = print('it: {}, error: {}'.format(
                    len(self.N_ber), np.sum(self.N_ber)))
        else:
            print_str = print('it: {}, error: {}'.format(
                len(self.sel_crit()), np.sum(self.sel_crit())))
        return print_str

    def it_print(self):
        '''Prints selected performance measures for current iteration
        '''
        # iterstr = '{},'.format(iteration)
        # Code rate included in EbN0
        print_str = 'EbN0: {:.1f} (SNR: {:.1f}), CBER: {:.2e}, BER: {:.2e}, SER: {:.2e}, CE: {:.6f}'.format(
            self.CEbN0[-1], self.SNR[-1], self.CBER[-1], self.BER[-1], self.SER[-1], self.CE[-1])
        return print_str

    def it_fail_print(self):
        '''Prints error message if accuracy not high enough for saving
        '''
        print_str = 'Accuracy not high enough: N_err={}<={} after {} iterations'.format(
            np.sum(self.sel_crit()), self.Nerr_min, self.it_max)
        return print_str


# Saving and filename----------------------------------------------------

class filename_module():
    '''Class responsible for file names
    For CMDNet/MIMO simulations
    '''
    # Class Attribute
    name = 'File name creator'
    # Initializer / Instance Attributes

    def __init__(self, typename, path, algo, fn_ext, sim_set, code_set=0):
        # Inputs
        self.ospath = path
        self.typename = typename
        self.filename = ''
        self.path = ''
        self.pathfile = ''
        self.mod = sim_set['Mod']
        self.Nr = sim_set['Nr']
        self.Nt = sim_set['Nt']
        self.L = sim_set['L']
        self.algoname = algo
        self.fn_ext = fn_ext
        self.code = code_set
        # Initialize
        self.generate_pathfile_MIMO()
    # Instance methods

    def generate_filename_MIMO(self):
        '''Generates file name
        '''
        if self.code:
            self.filename = self.typename + self.algoname + '_' + self.mod + '_{}_{}_{}_'.format(
                self.Nt, self.Nr, self.L) + self.code['code'] + self.code['dec'] + self.code['arch'] + self.fn_ext
        else:
            self.filename = self.typename + self.algoname + '_' + self.mod + \
                '_{}_{}_{}'.format(self.Nt, self.Nr, self.L) + self.fn_ext
        return self.filename

    def generate_path_MIMO(self):
        '''Generates path name
        '''
        self.path = os.path.join(
            self.ospath, self.mod, '{}x{}'.format(self.Nt, self.Nr))  # '/', '\\'
        return self.path

    def generate_pathfile_MIMO(self):
        '''Generates full path and filename
        '''
        self.generate_path_MIMO()
        self.generate_filename_MIMO()
        self.pathfile = os.path.join(self.path, self.filename)
        return self.pathfile


class savemodule():
    '''Class responsible for data saving
    '''
    # Class Attribute
    name = 'Save data'
    # Initializer / Instance Attributes

    def __init__(self, form='npz'):
        # Inputs
        self.json_ending = '.json'
        self.hdf5_ending = '.hdf5'
        self.npz_ending = '.npz'
        self.format = form
        self.list_str = 'el'
    # Instance methods

    def check_path(self, pathfile, verbose=0):
        '''Check for existing path and file, respectively
        '''
        path = os.path.dirname(pathfile)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose == 1:
                print('Created new directory.')
        else:
            if os.path.isfile(pathfile):
                os.remove(pathfile)
                if verbose == 1:
                    print('Deleted existing file.')
        return pathfile

    def save_hdf5(self, pathfile, data, verbose=0):
        '''Save data to hdf5 file
        data: data to be saved in hdf5
        pathfile: path and filename
        '''
        def save_nested2hdf5(f, data, name, list_str='el'):
            '''Save nested data in hdf5 format
            f: hdf5 group
            data: nested data to be saved
            name: name of data to be saved
            '''
            if isinstance(data, dict):
                # Avoid first group at first call?
                # if name:
                #     grp = f.create_group(name)
                # else:
                #     grp = f
                grp = f.create_group(name)
                for key, value in data.items():
                    save_nested2hdf5(grp, value, key, list_str)
            elif isinstance(data, list):
                grp = f.create_group(name)
                if len(data) > 1:
                    # lexicographical ordering requires filling with zeros
                    N_fill = int(np.log10(len(data) - 1) + 1)
                else:
                    N_fill = 1
                for ii, el in enumerate(data):
                    save_nested2hdf5(grp, el, list_str +
                                     str(ii).zfill(N_fill), list_str)
            else:
                f.create_dataset(name, data=data)  # dset =

        pathfile = pathfile + self.hdf5_ending
        self.check_path(pathfile, verbose)
        # Save
        with h5py.File(pathfile, 'w') as f:
            save_nested2hdf5(f, data, 'SavedData', list_str=self.list_str)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_hdf5(self, pathfile):
        '''Save data to hdf5 file
        load_data: data to be loaded from hdf5 file
        pathfile: path and filename
        '''
        def load_nested2hdf5(data, list_str='el'):
            '''Load nested data from hdf5 format
            data: data in hdf5 format
            '''
            if isinstance(data, h5py.Group):
                keylist = list(data.keys())
                # check for list / before check if elements in subgroup
                if keylist and keylist[0][0:len(list_str)] == list_str:
                    dic = []
                    for value in data.values():
                        dic.append(load_nested2hdf5(value, list_str))
                else:
                    dic = {}
                    for key, value in data.items():
                        dic[key] = load_nested2hdf5(value, list_str)
            else:
                # dic[data.name] = data.value
                dic = data.value
            return dic

        pathfile = pathfile + self.hdf5_ending
        load_data = None
        if os.path.isfile(pathfile):
            with h5py.File(pathfile, 'r') as f:
                dic = load_nested2hdf5(f, list_str=self.list_str)
            load_data = dic['SavedData']
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save_json(self, pathfile, data, verbose=0):
        '''Save data to json file
        data: data to be saved in json
        pathfile: path and filename
        '''
        class py2jsonEncoder(json.JSONEncoder):
            '''Encoder for dtype conversion to json
            '''

            def default(self, obj):
                if isinstance(obj, np.float16):
                    return obj.astype('float64')
                if isinstance(obj, np.float32):
                    return obj.astype('float64')
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)

        pathfile = pathfile + self.json_ending
        self.check_path(pathfile, verbose)
        # Save
        with open(pathfile, 'w') as outfile:
            json.dump(data, outfile, indent=4, cls=py2jsonEncoder)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_json(self, pathfile):
        '''Load data from json file
        load_data: data to be loaded from json file
        pathfile: path and filename
        '''
        pathfile = pathfile + self.json_ending
        load_data = None
        if os.path.isfile(pathfile):
            with open(pathfile, 'r') as outfile:
                load_data = json.load(outfile)
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save_npz(self, pathfile, data, verbose=0):
        '''Save data to npz file
        data: data to be saved in npz
        pathfile: path and filename
        '''
        pathfile = pathfile + self.npz_ending
        self.check_path(pathfile, verbose)
        # Save as np.array
        # for key, value in data.items():
        #    if isinstance(value, list):
        #        data[key] = np.array(value)
        np.savez(pathfile, **data)
        if verbose == 1:
            print('Saved into "' + pathfile + '".')
        return pathfile

    def load_npz(self, pathfile):
        '''Load data from npz file
        load_data: data to be loaded from npz file
        pathfile: path and filename
        '''
        pathfile = pathfile + self.npz_ending
        load_data = None
        if os.path.isfile(pathfile):
            load_data = np.load(pathfile)  # , allow_pickle = True)
            print('Loaded from "' + pathfile + '".')
        else:
            print('File not found.')
        return load_data

    def save(self, pathfile, data, form=None, verbose=0):
        '''Save data to specified format
        data: data to be saved in specified format
        pathfile: path and filename
        form: Format to be used (npz, hdf5, json)
        '''
        if form == None:
            form = self.format
        if form == 'npz':
            self.save_npz(pathfile, data, verbose=verbose)
        elif form == 'hdf5':
            self.save_hdf5(pathfile, data, verbose=verbose)
        elif form == 'json':
            self.save_json(pathfile, data, verbose=verbose)
        else:
            print('This format is not available.')
        return pathfile

    def load(self, pathfile, form=None):
        '''Load data from file with specified format
        load_data: data to be loaded from file with specified format
        pathfile: path and filename
        form: Format to be used (npz, hdf5, json)
        '''
        if form == None:
            form = self.format
        load_data = None
        if form == 'npz':
            load_data = self.load_npz(pathfile)
        elif form == 'hdf5':
            load_data = self.load_hdf5(pathfile)
        elif form == 'json':
            load_data = self.load_json(pathfile)
        else:
            print('This format is not available.')
        return load_data

# EOF
