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
import glob
import shutil
import time


os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras  # NOQA

# Own packages
from utilities.my_functions import print_time  # NOQA
from sinfony import simulation_sinfony  # NOQA
from sinfony_io import find_unique_results_path  # NOQA
import utilities.my_math_operations as mop  # NOQA
from sinfony_classic import simulation_sinfony_classic  # NOQA                                            # NOQA
from sinfony_classic_features_autoencoder import simulation_sinfony_classic_features_autoencoder    # NOQA


def find_duplicate_weight_files(search_dir):
    """Find files that exist in both _weights.h5 and .weights.h5 format"""
    duplicates = []

    for filepath in glob.glob(os.path.join(search_dir, '**/*_weights.h5'), recursive=True):
        # Construct the alternative path
        alt_filepath = filepath.replace('_weights.h5', '.weights.h5')
        if os.path.isfile(alt_filepath):
            duplicates.append((filepath, alt_filepath))

    if duplicates:
        print(f'Found {len(duplicates)} duplicate(s):')
        for f1, f2 in duplicates:
            print(f'  {f1}')
            print(f'  {f2}')
    else:
        print('No duplicates found.')

    return duplicates


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Initialization
    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        from utilities.gpu_select_tf import gpu_select
        gpu_select(number=-2, memory_growth=True)

    move_old_model = False
    snr_range = [0, 0]          # Default: [-30,20]
    snr_step_size = 1           # 1
    load = True                 # True
    test_mode = True            # True
    find_duplicates = False
    test_tolerance = 2e-2

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # Default: 'mnist'
    SETTINGS_FILE = 'mnist'
    # Change from 'settings' to 'models' to reload simulations settings
    SETTINGS_FOLDER = 'models'
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)

    if find_duplicates:
        duplicates_0 = find_duplicate_weight_files(settings_path)
        settings_files = list(set(
            f1.replace('_weights.h5', '').replace('.weights.h5', '')
            for f1, f2 in duplicates_0
        ))
        # print('Corresponding yaml files:')
        # for f in settings_files:
        #     print(f'  {f}')
    else:
        entries = os.listdir(settings_path)
        settings_files = [os.path.splitext(entry)[0] for entry in entries if entry.endswith(
            '.yaml') and os.path.isfile(os.path.join(settings_path, entry))]

    # dataset_loaded = False

    test_iteration = 0
    successful_tests = 0
    successful_runs = 0
    failed_tests = 0
    number_files = len(settings_files)

    start_time = time.time()
    for settings_file in settings_files:
        start_time2 = time.time()
        print('Testing: ' + settings_file)
        if any(s in settings_file for s in ['CIFAR2_nosnr', 'CIFAR5', 'CIFAR7', 'MNIST2_nosnr', 'MNIST5', 'MNIST7', 'nonoise']) or settings_file == 'ResNet14_MNIST2_Ne20':
            # Set SNR=0 effectively to account for correct evaluation settings of the models
            snrs = mop.snr_range2snrlist([1e4, 1e4], snr_step_size)
        else:
            snrs = mop.snr_range2snrlist(snr_range, snr_step_size)
        try:
            pathfile = os.path.join(settings_path, settings_file)
            if SETTINGS_FILE == 'classic':
                results, pathfile_results, save_object = simulation_sinfony_classic(
                    pathfile, test_mode=test_mode, snrs=snrs)
            elif SETTINGS_FILE == 'classic_ae':
                results, pathfile_results, save_object = simulation_sinfony_classic_features_autoencoder(
                    pathfile, test_mode=test_mode, load=load, snrs=snrs)
            else:
                results, pathfile_results, save_object = simulation_sinfony(
                    pathfile, test_mode=test_mode, load=load, snrs=snrs)

            successful_runs = successful_runs + 1

            if settings_file == 'ResNet14_MNIST2_Ne20':
                # Special fix for loading correct results
                results['snr'] = mop.snr_range2snrlist(
                    snr_range, snr_step_size)

            _, counter, accuracy_equal = find_unique_results_path(
                pathfile_results, save_object, results['val_acc'], results['snr'], tolerance=test_tolerance)
            if counter == 1:
                print("Simulation results file not found, jump to next file...")
                accuracy_equal = False
            elif counter >= 2 and not accuracy_equal:
                failed_tests = failed_tests + 1

        except Exception as e:
            print(f"Error: {e}")
            accuracy_equal = False
            # raise
            # pass  # Keep default if parsing fails
        finally:
            if accuracy_equal:
                successful_tests = successful_tests + 1
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

            # Print current progress
            test_iteration = test_iteration + 1
            print(
                f"SINFONY test iteration: {test_iteration}/{number_files}, Successful runs: {successful_runs}/{test_iteration}, Successful tests: {successful_tests}/{successful_runs}, Failed tests: {failed_tests}/{successful_runs}, Total Time: {print_time(time.time() - start_time2)}/{print_time(time.time() - start_time)}\n")
            keras.backend.clear_session()


# EOF
