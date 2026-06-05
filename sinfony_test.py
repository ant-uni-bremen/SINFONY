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
import os
import shutil
import numpy as np
import yaml
# from matplotlib import pyplot as plt

# Tensorflow 2 packages
os.environ['KERAS_BACKEND'] = 'tensorflow'
if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from utilities.gpu_select_tf import gpu_select
    backend_tf = True
else:
    backend_tf = False
import keras


# Own packages
import utilities.my_math_operations as mop
from sinfony_io import find_unique_results_path
from sinfony import simulation_sinfony
from sinfony_classic import simulation_sinfony_classic
from sinfony_classic_features_autoencoder import simulation_sinfony_classic_features_autoencoder

if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Initialization
    if backend_tf:
        gpu_select(number=-2, memory_growth=True)

    move_old_model = True
    snr_range = [0, 0]          # Default: [-30,20]
    snr_step_size = 1           # 1
    load = True                 # True
    test_mode = True            # True

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # Default: 'mnist'
    SETTINGS_FILE = 'classic_ae'
    # Change from 'settings' to 'models' to reload simulations settings
    SETTINGS_FOLDER = 'models'
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)
    entries = os.listdir(settings_path)
    settings_files = [os.path.splitext(entry)[0] for entry in entries if entry.endswith(
        '.yaml') and os.path.isfile(os.path.join(settings_path, entry))]
    snrs = mop.snr_range2snrlist(snr_range, snr_step_size)

    # dataset_loaded = False

    for settings_file in settings_files:
        print('Testing: ' + settings_file)
        try:
            pathfile = os.path.join(settings_path, settings_file)
            if SETTINGS_FILE == 'classic':
                results, pathfile_results, save_object = simulation_sinfony_classic(pathfile, test_mode=test_mode, snrs=snrs)
            elif SETTINGS_FILE == 'classic_ae':
                results, pathfile_results, save_object = simulation_sinfony_classic_features_autoencoder(
                    pathfile, test_mode=test_mode, load=load, snrs=snrs)
            else:
                results, pathfile_results, save_object = simulation_sinfony(
                    pathfile, test_mode=test_mode, load=load, snrs=snrs)

            pathfile_results, counter, accuracy_equal = find_unique_results_path(
                pathfile_results, save_object, results['val_acc'], results['snr'], tolerance=2e-2)
            if counter == 1:
                print("Simulation results file not found, jump to next file...")

        except Exception as e:
            print(f"Error: {e}")
            accuracy_equal = False
            # raise
            # pass  # Keep default if parsing fails
        finally:
            # Backup the folder
            if move_old_model and accuracy_equal:
                # Define all files that should be backed up
                backup_files = [
                    pathfile + '.yaml',
                    pathfile + '_results.npz',
                    pathfile + '_weights.h5',
                    pathfile + '.weights.h5',
                    pathfile + '.npz',
                ]

                try:
                    backup_path = os.path.join(
                        os.path.dirname(pathfile + '.yaml'), 'backup')
                    os.makedirs(backup_path, exist_ok=True)

                    # Move all backup files
                    for backup_file in backup_files:
                        if os.path.exists(backup_file):
                            shutil.move(backup_file, backup_path)
                            print(
                                f"Successfully moved {backup_file} to {backup_path}")
                        else:
                            print(
                                f"File {backup_file} does not exist, skipping...")

                except Exception as move_error:
                    print(
                        f"Failed to move files to backup: {move_error}")

            keras.backend.clear_session()


# EOF
