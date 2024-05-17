#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:04:21 2022

@author: beck
Simulation framework for numerical results of classical digital communication in the article:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA


# LOADED PACKAGES
# Python packages
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import yaml

# Tensorflow 2 packages
import tensorflow as tf
import sionna as sn


# Own packages
import huffman_coding as hc
import datasets
import my_float as mfl
# Note: Important to load models from old files, there a reference to mf including layers is hardcoded
import my_training as mf
import my_training as mt
from my_functions import print_time, savemodule
import my_math_operations as mop


# Only necessary for Windows, otherwise kernel crashes
if os.name.lower() == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def classic_digital_communication(source_signal, huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=None, probability_bit=None):
    '''Classical digital bit transmission with Huffman source coding and LDPC code over AWGN channel
    source_signal: Float values of application / bit input as integer values
    huffman: Huffman encoder/decoder object
    information_word_length: Information word length
    encoder: Channel encoder
    mapper: Modulation mapper
    channel: Communication channel
    demapper: Modulation demapper
    decoder: Channel decoder 
    snr: SNR in dB
    probability_bit: Probabilities of floating bits
    floatx: Floating point conversion object (optional, required for float input)
    reconstructed_source_signal: Reconstructed float values for application / bit output as integer values
    '''
    # Check if input is integer
    integer_check = np.issubdtype(source_signal.dtype, np.integer)
    if integer_check is True:
        # Integers/bits are directly fed into the communication system
        # Bit number is hard coded here
        bit_integer = source_signal
        integer_type = bit_integer.dtype
        if bit_integer.dtype == 'uint8':
            number_bits = 8
        else:
            print('Not implemented!')
    else:
        # Transform source signal into floating point bits
        bit_integer, _ = floatx.float2bitint(source_signal)
        integer_type = floatx.intx
        number_bits = floatx.N_bits

    # Huffman encoding
    bit_huffman_sequence, _ = huffman.encoding(bit_integer)
    # If the number of bits required to be a code block, add random bits
    bits_random_fill = np.random.randint(
        0, 2, size=information_word_length - len(bit_huffman_sequence) % information_word_length)
    source_bits = np.concatenate(
        (np.array(bit_huffman_sequence, dtype='float32'), bits_random_fill))

    # Channel coding
    code_bits = encoder(source_bits.reshape((-1, information_word_length)))

    code_bits_int = interleaver(code_bits)
    transmit_signal = mapper(code_bits_int)
    noise_standard_deviation = mop.snr2standard_deviation(np.random.uniform(
        snr, snr, transmit_signal.shape[0]))[..., np.newaxis].astype('float32')
    if mapper.constellation._constellation_type == 'pam':
        # Complex channel with half the variance in real- and imaginary part
        received_signal = channel(
            [transmit_signal,  2 * noise_standard_deviation ** 2])
        # Also consider here doubled variance
        llr_channel = demapper(
            [received_signal, 2 * noise_standard_deviation ** 2])
    else:
        received_signal = channel(
            [transmit_signal,  noise_standard_deviation ** 2])
        llr_channel = demapper(
            [received_signal, noise_standard_deviation ** 2])
    llr_deinterleaved = deinterleaver(llr_channel)
    # Soft information of code_bits_received cannot pass through Huffman decoding
    code_bits_received = decoder(llr_deinterleaved)

    # Remove added random bits and Huffman decoding
    bit_integer_reconstructed = np.array(huffman.decoding(code_bits_received.numpy().flatten()[
        0:source_bits.shape[0] - bits_random_fill.shape[0]].astype(integer_type).tolist()))
    # If the number of bits after Huffman decoding has changed compared to transmit signals:
    if bit_integer.shape[0] < bit_integer_reconstructed.shape[0]:
        # More than before: Take only as much bits
        bit_integer_reconstructed = bit_integer_reconstructed[0:bit_integer.shape[0]]
    elif bit_integer.shape[0] > bit_integer_reconstructed.shape[0]:
        # Less than before: Add random bits
        bit_integer_reconstructed = np.concatenate((bit_integer_reconstructed, np.random.randint(
            0, 2 ** number_bits, size=bit_integer.shape[0] - bit_integer_reconstructed.shape[0], dtype=integer_type)))
    if integer_check is True:
        reconstructed_source = bit_integer_reconstructed
    else:
        # Transform Huffman integers back to bit stream
        bits_received = mop.int2bin(bit_integer_reconstructed, N=number_bits)
        # Float error correction based on a priori probabilities
        bits_received_corrected = floatx.float_errorcorrection(
            bits_received, probability_bit[..., 0][np.newaxis].repeat(bits_received.shape[0], axis=0))
        # Transform bit sequence into float value
        reconstructed_source = floatx.bit2float(bits_received_corrected)
    return reconstructed_source


def sequence_prior_data_int(bit_integer, bit_integer_maximum=-1, show=False):
    '''Probabilities p(source_signal) of bit sequences or source_signal computed from data set
    INPUT
    bit_integer: Bit sequence as integer
    bit_integer_maximum: Highest integer of bit sequence
    show: Show data distribution
    OUTPUT
    probability_integer: probabilities for each possible integer value
    '''
    # Probabilities of data set discretized to integer values
    # Flatten and discretize data set to floatx precision
    bits_count = np.bincount(bit_integer)
    if bit_integer_maximum == -1:
        bit_integer_maximum = bits_count.shape[0]
    probability_integer = np.zeros(bit_integer_maximum)
    probability_integer[0:bits_count.shape[0]] = bits_count
    probability_integer = probability_integer / np.sum(probability_integer)

    # Plot data distribution before and after quantization
    if show is True:
        plt.figure()
        plt.hist(bit_integer, bins=bit_integer_maximum)
        plt.figure()
        plt.plot(np.arange(0, bit_integer_maximum),
                 probability_integer, 'r-o', label='p(source_signal)')
    return probability_integer


def image_transmission(images, blocks, huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=None, probability_bit=None):
    '''Test image data enters classic communications as source source_signal
    '''
    source_signal = images.reshape([images.shape[0], -1])
    # Evaluate classical digital transmission of images
    # Huffman encode [blocks] feature vectors into one block + split across agents
    reconstructed_source = np.zeros(source_signal.shape)
    number_blocks = int(source_signal.shape[0] / blocks)
    for index_block in range(0, number_blocks):
        reconstructed_source[index_block * blocks:(index_block + 1) * blocks, :] = classic_digital_communication(source_signal[index_block * blocks:(index_block + 1) * blocks, :].flatten(
        ), huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=floatx, probability_bit=probability_bit).reshape((blocks, -1))
    # Rest not included in former blocks
    number_remaining_blocks = reconstructed_source.shape[0] - \
        number_blocks * blocks
    if number_remaining_blocks >= 1:
        reconstructed_source[number_blocks * blocks:, :] = classic_digital_communication(source_signal[number_blocks * blocks:, :].flatten(
        ), huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=floatx, probability_bit=probability_bit).reshape((number_remaining_blocks, -1))
    reconstructed_source = reconstructed_source.reshape(images.shape)
    # reconstructed_source = datasets.preprocess_pixels_image(
    #     reconstructed_source)
    # number_classes = model.predict(reconstructed_source)
    return reconstructed_source  # number_classes


def feature_transmission(source_signal, blocks, huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=None, probability_bit=None):
    '''Test data features enter classic communications as source source_signal.
    '''
    # Evaluate classical digital transmission
    # Huffman encode [blocks] feature vectors into one block + split across agents
    reconstructed_source = np.zeros(source_signal.shape)
    for index_block in range(0, int(source_signal.shape[0] / blocks)):
        # Consider transmission of each agent separately
        for index_x in range(0, source_signal.shape[1]):
            for index_y in range(0, source_signal.shape[2]):
                reconstructed_source[index_block * blocks:(index_block + 1) * blocks, index_x, index_y, :] = classic_digital_communication(source_signal[index_block * blocks:(index_block + 1) * blocks, index_x, index_y, :].flatten(
                ), huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=floatx, probability_bit=probability_bit).reshape((blocks, -1))
    # Extract semantics based on received signal received_signal = r_r = reconstructed_source
    # number_classes = model.layers[-1].predict(reconstructed_source)
    return reconstructed_source  # number_classes


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Load parameters from configuration file
    # Get the script's directory
    path_script = os.path.dirname(os.path.abspath(__file__))
    # Default: 'classic/config_classic.yaml'
    SETTINGS_FILE = 'classic/config_classic.yaml'
    # Load the provided configuration file or the default one
    # python SINFONY.py semantic_config.yaml
    # Workaround for interactive sessions: Only allow config file names starting 'semantic_config'
    SETTINGS_FILE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][0:15].lower(
    ) == 'semantic_config' else SETTINGS_FILE
    # Change to 'settings_saved' to reload simulations settings
    SETTINGS_FOLDER = 'settings'
    settings_path = os.path.join(path_script, SETTINGS_FOLDER, SETTINGS_FILE)
    with open(settings_path, 'r', encoding='UTF8') as file:
        params = yaml.safe_load(file)
    load_settings = params['load_settings']
    dataset_settings = params['dataset']
    communication_settings = params['communication']
    evaluation_settings = params['evaluation']

    # Initialization
    mt.gpu_select(number=load_settings['gpu'], memory_growth=True)
    tf.keras.backend.clear_session()                # Clearing graphs
    tf.keras.backend.set_floatx(load_settings['numerical_precision'])
    # Random seed in every run, predictable random numbers for debugging with np.random.seed(0)
    np.random.seed()

    # Simulation parameters
    classic = evaluation_settings['classic_mode']
    filename_extension = load_settings['simulation_filename_suffix']
    filename_prefix = load_settings['simulation_filename_prefix']
    saveobj = savemodule(form=load_settings['save_format'])

    # Loaded dataset
    # mnist, cifar10, fashion_mnist, hirise64, hirisecrater, fraeser64
    dataset = dataset_settings['dataset']
    # Show first dataset examples, just for demonstration
    show_dataset = dataset_settings['show_dataset']
    train_input, train_labels, test_input, test_labels = datasets.load_dataset(
        dataset)

    # Automatic decision for model with dataset
    if dataset == 'mnist':
        subpath = 'mnist'
        if classic == 2:
            # NOTE: Only simulated for 'ResNet14_MNIST'
            # File for central image classification
            filename = 'ResNet14_MNIST'
        else:
            filename = 'ResNet14_MNIST2_Ne20'
    elif dataset == 'cifar10':
        subpath = 'cifar10'
        if classic == 2:
            filename = 'ResNet20_CIFAR'
        else:
            filename = 'ResNet20_CIFAR2'
    elif dataset == 'fraeser':
        subpath = 'fraeser'
        if classic == 2:
            filename = 'ResNet18_fraeser'               # ResNet18_fraeser_test
        else:
            # NOTE: Should be SINFONY version without noise and transceiver layers, not, e.g., sinfony18_fraeser_lr1e-3_3
            filename = 'ResNet18_fraeser'
    else:
        print('Dataset not implemented into script.')

    # Path for SINFONY model
    path_sinfony = os.path.join(load_settings['path_models'], subpath)
    # Path for classic results
    path_classic = load_settings['path_classic']

    # Parameters
    # Classic communications parameters
    # Code rate: 0.25, 0.5, 0.75
    rate_channel_code = communication_settings['rate_channel_code']
    # Code length: 1000 (default), 16000 (rate_channel_code = 0.5), 15360 (rate_channel_code = 0.25), 11264 (rate_channel_code = 0.75)
    code_word_length = communication_settings['code_word_length']
    # Huffman encode [blocks] feature vectors into one block + channel encoding: 100, 1000 is practical
    blocks = communication_settings['blocks']
    # Huffman code is computational bottleneck: But the smaller the blocks, the less severe errors are
    modulation = communication_settings['modulation']            # Modulation
    num_bits_per_symbol = communication_settings['num_bits_per_symbol']
    float_name = communication_settings['float_compression_to']  # float16
    if classic == 1:
        algorithm = 'classic'
    elif classic == 2:
        algorithm = 'classic_image'

    # Evaluation parameters
    if modulation == 'qam':
        modulation_order = num_bits_per_symbol / 2
    else:
        modulation_order = num_bits_per_symbol
    # 10, rounds through validation data with different channel noise realizations
    validation_rounds = evaluation_settings['validation_rounds']
    # [-1, 5] + 10 * np.log10(2 * rate_channel_code * modulation_order)
    snr_range = evaluation_settings['snr_range'] + 10 * \
        np.log10(2 * rate_channel_code * modulation_order)
    step_size = evaluation_settings['snr_step_size']

    # Evaluation script

    # Load the SINFONY model
    # Path of script being executed
    pathfile = os.path.join(path_script, path_sinfony, filename)
    print('Loading model ' + filename + '...')
    model = tf.keras.models.load_model(pathfile)
    print('Model loaded.')
    if show_dataset is True:
        model.summary()

    # Preprocess Data set
    [train_input_normalized, test_input_normalized] = datasets.preprocess_pixels(
        train_input, test_input)
    # test_input_normalized = datasets.preprocess_pixels_image(test_input)
    if show_dataset is True:
        datasets.summarize_dataset(train_input, train_labels,
                                   test_input, test_labels)

    if classic != 2:
        if len(train_input_normalized) == 1:
            # Models with internal image split
            data_train = [model.layers[1].predict(train_input_normalized)]
            data_validation = [model.layers[1].predict(test_input_normalized)]
        else:
            # Models with input split
            data_train = []
            for index_model, model_layer in enumerate(model.layers[len(train_input_normalized)].layers[-len(train_input_normalized)-1:-1]):
                data_train.append(model_layer.predict(
                    train_input_normalized[index_model]))
            data_validation = []
            for index_model, model_layer in enumerate(model.layers[len(test_input_normalized)].layers[-len(test_input_normalized)-1:-1]):
                data_validation.append(model_layer.predict(
                    test_input_normalized[index_model]))

    # Initialize classic and AE communications
    start_time = time.time()
    rng = np.random.default_rng()

    # Classic communication for whole test data set
    if classic == 2:
        # Images have RGB color entries -> 256 Bit
        # Each entry is one symbol for huffman encoding
        number_bits = evaluation_settings['bits_per_image_value']
        number_categorical = 2 ** number_bits
        bits_poss = np.arange(0, number_categorical)
        # Flatten dataset for digital transmission -> Huffman encoding over all distributed agent data
        test_input_flattened = []
        for test_input_item in test_input:
            test_input_flattened.append(test_input_item.flatten())
        test_input_flattened = np.concatenate(test_input_flattened)
        probability_sequence = sequence_prior_data_int(
            test_input_flattened, bit_integer_maximum=number_categorical)
    elif classic == 1:
        # Features to be transmitted are floating point values
        # Each value is one symbol for huffman encoding
        floatx = mfl.float_toolbox(float_name)
        # Note: Compression from, e.g., float32 to float16 possible!
        number_bits = floatx.N_bits
        bits_poss = floatx.b_poss
        # Flatten dataset for digital transmission -> Huffman encoding over all distributed agent data
        data_validation_flattened = []
        for data_validation_item in data_validation:
            data_validation_flattened.append(data_validation_item.flatten())
        data_validation_flattened = np.concatenate(data_validation_flattened)
        probability_sequence = mfl.sequence_prior_data(
            floatx, data=data_validation_flattened)
        probability_bit = mfl.compute_single_bitprob(
            floatx, probability_sequence)
    # Definition of communication blocks
    huffman = hc.huffman_coder(symbols=bits_poss, probs=probability_sequence)
    # Compute total gain of the Huffman encoding
    _, _, _, _, rate_huffman_code = huffman.total_gain()
    # Information word length
    information_word_length = int(code_word_length * rate_channel_code)
    # Communications components from Sionna
    encoder = sn.fec.ldpc.LDPC5GEncoder(
        information_word_length, code_word_length)
    interleaver = sn.fec.interleaving.RowColumnInterleaver(
        row_depth=num_bits_per_symbol)
    deinterleaver = sn.fec.interleaving.Deinterleaver(interleaver)
    constellation = sn.mapping.Constellation(
        modulation, num_bits_per_symbol=num_bits_per_symbol)
    mapper = sn.mapping.Mapper(constellation=constellation)
    channel = sn.channel.AWGN()
    demapper = sn.mapping.Demapper('app', constellation=constellation)
    decoder = sn.fec.ldpc.LDPC5GDecoder(
        encoder, cn_type='boxplus')  # , hard_out = False
    # Compute total rate
    rate_total = rate_huffman_code * rate_channel_code
    if classic == 2:
        # Number of image entries
        number_entries = 0
        for test_input_item in test_input:
            number_entries = number_entries + \
                np.prod(test_input_item.shape[1:])
    else:
        # Number of features
        number_entries = 0
        for data_validation_item in data_validation:
            number_entries = number_entries + data_validation_item.shape[-1]
    # Average number of channels uses with digital communication
    mean_number_channel_uses = number_entries * number_bits / rate_total
    print(f'Mean number of channel uses: {mean_number_channel_uses:.2f}')

    # Print initialization time
    print('Initialization Time: ' + print_time(time.time() - start_time))

    # Evaluation of model
    print('Evaluate model...')
    # Evaluate model for different SNRs
    snrs = np.linspace(snr_range[0], snr_range[1], int(
        (snr_range[1] - snr_range[0]) / step_size) + 1)
    # SINFONY/RL-SINFONY evaluated with classic communication
    start_time = time.time()
    evaluation_measures = [[], []]
    for snr_index, snr in enumerate(snrs):
        loss_i = 0
        accuracy_i = 0
        for validation_round in range(0, validation_rounds):
            start_time2 = time.time()
            if classic == 1:
                # Test data features enter classic communications as source source_signal.
                reconstructed_sources = []
                for data_validation_item in data_validation:
                    source_signal = data_validation_item
                    if len(source_signal.shape) == 4:
                        # Models with internal image split
                        reconstructed_source = feature_transmission(source_signal, blocks, huffman, information_word_length, encoder, mapper,
                                                                    channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=floatx, probability_bit=probability_bit)
                    else:
                        # Models with input split
                        reconstructed_source = image_transmission(
                            source_signal, blocks, huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr, floatx=floatx, probability_bit=probability_bit)
                    reconstructed_sources.append(reconstructed_source)
                reconstructed_sources = np.concatenate(
                    reconstructed_sources, axis=-1)
                # Extract semantics based on received signal received_signal = r_r = reconstructed_source
                number_classes = model.layers[-1](reconstructed_sources)
                # number_classes = model.layers[-1].predict(reconstructed_sources)
            elif classic == 2:
                reconstructed_sources = []
                for test_input_item in test_input:
                    reconstructed_source = image_transmission(
                        test_input_item, blocks, huffman, information_word_length, encoder, mapper, channel, demapper, decoder, interleaver, deinterleaver, snr)
                    reconstructed_sources.append(datasets.preprocess_pixels_image(
                        reconstructed_source))
                # Extract semantics based on all received images
                number_classes = model.predict(reconstructed_sources)
            # Calculate average loss and accuracy
            loss_ii = np.mean(model.loss(test_labels, number_classes))
            accuracy_ii = np.mean(np.argmax(number_classes, axis=-1) ==
                                  np.argmax(test_labels, axis=-1))

            # Add current measures to total measures
            loss_i = (validation_round * loss_i + loss_ii) / \
                (validation_round + 1)
            accuracy_i = (validation_round * accuracy_i +
                          accuracy_ii) / (validation_round + 1)
            print(
                f'Validation Round: {validation_round + 1}/{validation_rounds}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time2)}')

        # Append list with evaluation for each SNR value
        evaluation_measures[0].append(loss_i)
        evaluation_measures[1].append(accuracy_i)
        print(
            f'Iteration: {snr_index + 1}/{len(snrs)}, SNR: {snr}, CE: {loss_i:.4f}, Acc: {accuracy_i:.2f}, Time: {print_time(time.time() - start_time)}')

    accuracy = np.array(evaluation_measures[1])
    loss = np.array(evaluation_measures[0])
    plt.figure(1)
    plt.semilogy(snrs, 1 - accuracy)
    plt.figure(2)
    plt.semilogy(snrs, loss)

    # Save evaluation
    print('Save evaluation...')
    results = {
        "snr": snrs,
        "val_loss": loss,
        "val_acc": accuracy,
    }
    pathfile = os.path.join(path_script, path_classic, filename_prefix +
                            algorithm + '_' + filename + filename_extension)
    saveobj.save(pathfile, results)
    print('Evaluation saved.')

    # Save settings when evaluation is done
    SETTINGS_SAVED_FOLDER = 'settings_saved'
    saved_settings_path = os.path.join(path_script, SETTINGS_SAVED_FOLDER)
    with open(os.path.join(saved_settings_path, filename + '.yaml'), 'w', encoding='utf8') as written_file:
        yaml.safe_dump(params, written_file, default_flow_style=False)
    print('Settings saved!')

# EOF
