#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:53:30 2026

@author: beck
Module for stft
"""

import os
import numpy as np
from tqdm import tqdm
import keras
try:
    print(os.environ['KERAS_BACKEND'])
except Exception as e:
    os.environ['KERAS_BACKEND'] = 'tensorflow'

BACKEND_AGNOSTIC_STFT = True


def stft_dataset_in_partitions_keras(data_waveforms, frame_length, frame_step, number_partitions=20, normalization_constant=1, dtype='float32'):
    '''Backend-agnostic STFT for spectrogram
    '''
    data_spectrograms = []
    if number_partitions == 1:
        for waveforms in data_waveforms:
            stft = keras.ops.stft(
                waveforms.astype(dtype) / normalization_constant,
                sequence_length=frame_length,
                sequence_stride=frame_step,
                fft_length=frame_length + 1,
                center=False,
                window="hann"
            )
            # spectrograms = stft
            spectrograms = stft[0] ** 2 + stft[1] ** 2
            data_spectrograms.append(spectrograms)
    else:
        for waveforms in data_waveforms:
            partition_size = waveforms.shape[0] // number_partitions
            spectrograms = []
            for ind in tqdm(range(0, number_partitions)):
                start = ind * partition_size
                end = start + partition_size
                stft = keras.ops.stft(
                    waveforms[start:end, ...].astype(
                        dtype) / normalization_constant,
                    sequence_length=frame_length,
                    sequence_stride=frame_step,
                    fft_length=frame_length + 1,
                    center=False,
                    window="hann"
                )
                # spectrogram = stft
                spectrogram = stft[0] ** 2 + stft[1] ** 2
                spectrograms.append(spectrogram)
            stft = keras.ops.stft(
                waveforms[end:, ...].astype(
                    dtype) / normalization_constant,
                sequence_length=frame_length,
                sequence_stride=frame_step,
                fft_length=frame_length + 1,
                center=False,
                window="hann"
            )
            # spectrogram = stft
            spectrogram = stft[0] ** 2 + stft[1] ** 2
            spectrograms.append(spectrogram)
            spectrograms = np.concatenate(spectrograms, axis=0)
            data_spectrograms.append(spectrograms)
    return data_spectrograms


if not BACKEND_AGNOSTIC_STFT:
    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        import tensorflow as tf

        def stft_dataset_in_partitions_tf(data_waveforms, frame_length, frame_step, number_partitions=20, normalization_constant=1, dtype='float32'):
            '''Tensorflow-based calculation of STFT for spectrogram
            '''
            data_spectrograms = []
            if number_partitions == 1:
                for waveforms in data_waveforms:
                    stft = tf.signal.stft(
                        waveforms.astype(dtype) / normalization_constant,
                        frame_length=frame_length,
                        frame_step=frame_step,
                    )
                    # spectrograms = stft
                    spectrograms = keras.ops.abs(stft) ** 2
                    data_spectrograms.append(spectrograms)
            else:
                for waveforms in data_waveforms:
                    partition_size = waveforms.shape[0] // number_partitions
                    spectrograms = []
                    for ind in tqdm(range(0, number_partitions)):
                        start = ind * partition_size
                        end = start + partition_size
                        stft = tf.signal.stft(
                            waveforms[start:end, ...].astype(
                                dtype) / normalization_constant,
                            frame_length=frame_length,
                            frame_step=frame_step,
                        )
                        # spectrogram = stft
                        spectrogram = keras.ops.abs(stft) ** 2
                        spectrograms.append(spectrogram)
                    stft = tf.signal.stft(
                        waveforms[end:, ...].astype(
                            dtype) / normalization_constant,
                        frame_length=frame_length,
                        frame_step=frame_step,
                    )
                    # spectrogram = stft
                    spectrogram = keras.ops.abs(stft) ** 2
                    spectrograms.append(spectrogram)
                    spectrograms = np.concatenate(spectrograms, axis=0)
                    data_spectrograms.append(spectrograms)
            return data_spectrograms

        stft_dataset_in_partitions = stft_dataset_in_partitions_tf
        print('Selected tensorflow-based stft.')
    else:
        stft_dataset_in_partitions = stft_dataset_in_partitions_keras
        print('Selected backend-agnostic stft.')
else:
    stft_dataset_in_partitions = stft_dataset_in_partitions_keras
    print('Selected backend-agnostic stft.')


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():
    from datasets import load_dataset_speech_commands
    from utilities.gpu_select_tf import gpu_select
    gpu_select(number=-2, memory_growth=True)

    # Speech Commands dataset
    print('Loading speech commands dataset...')
    data_waveforms, data_labels = load_dataset_speech_commands(
        duration=1)
    print('Computing spectrograms by tensorflow stft...')
    # Normalization in speech commands by maximum value 32768.0 to avoid overflow
    data_spectrograms_tf = stft_dataset_in_partitions_tf(
        data_waveforms, frame_length=255, frame_step=128, normalization_constant=32768.0)
    print('Computing spectrograms by keras stft...')
    data_spectrograms_keras = stft_dataset_in_partitions_keras(
        data_waveforms, frame_length=255, frame_step=128, normalization_constant=32768.0)
    mean_error = np.mean(
        np.abs(data_spectrograms_tf[0] - data_spectrograms_keras[0]))
    print(f'Mean normalized absolute error: {mean_error:.2e}')
