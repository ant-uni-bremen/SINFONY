#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40 2025

@author: beck
GCM implemented in tensorflow

Belongs to simulation framework for numerical results of the articles:
1. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. doi: 10.48550/arXiv.2412.05103.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

import os
import gc
import tensorflow as tf
# import tensorflow.keras as keras
import keras
from keras.optimizers import SGD, Adam
import numpy as np
from matplotlib import pyplot as plt
import time

import datasets
import utilities.my_training as mt
import sinfony_wrapper as sw
import model_evaluation
import utilities.my_math_operations as mop
from utilities.my_functions import savemodule, print_time, get_ram
import sinfony_io


def logarithmic_scale(N):
    result = []
    power = 0
    while True:
        base = 10 ** power
        for multiplier in range(1, 10):
            val = multiplier * base
            if val > N:
                return result
            result.append(val)
        power += 1


def logarithmic_scale_inclusive(N):
    scale = logarithmic_scale(N)
    if scale[-1] != N:
        scale.append(N)
    return scale


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class AttentionWeightLayer(keras.layers.Layer):
    '''Layer to normalize attention weights to one and restrict
    to positive values, to implement constraints
    '''

    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.epsilon = 1e-12
        self.attention_weights = self.add_weight(
            name="raw_weights",
            shape=(1, 1, self.feature_dim),
            initializer="ones",
            trainable=True,
        )

    def call(self, inputs=None):
        # Optional: inputs are ignored
        positive_weights = keras.activations.relu(self.attention_weights)
        normalized_weights = positive_weights / (
            tf.reduce_sum(positive_weights, axis=-1,
                          keepdims=True) + self.epsilon
        )
        return normalized_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class WeightedDistanceLayer(keras.layers.Layer):
    def __init__(self, weight_number, **kwargs):
        super().__init__(**kwargs)
        self.weight_number = weight_number
        self.epsilon = 1e-12
        self.attention_weight_layer = AttentionWeightLayer(weight_number)

    def call(self, inputs):
        x, exemplars = inputs  # unpack inputs
        # Normalize the weights so they sum to 1 across the feature dimension
        attention_weights = self.attention_weight_layer(None)
        weighted_diff = tf.square(tf.expand_dims(x, 1) - tf.expand_dims(exemplars, 0)) * \
            attention_weights   # (B, N, D)
        distances = tf.sqrt(tf.reduce_sum(
            weighted_diff, axis=-1) + self.epsilon)  # (B, N)
        return distances

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_number": self.weight_number,
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class SimilarityLayer(keras.layers.Layer):
    def __init__(self, similarity_gradient=1.0, **kwargs):
        super().__init__(**kwargs)
        self.similarity_gradient = similarity_gradient
        self.c = self.add_weight(
            name="similarity_gradient",
            shape=(),
            initializer=keras.initializers.Constant(self.similarity_gradient),
            trainable=True,
        )

    def call(self, inputs):
        distances, labels = inputs
        # Similarity: s = exp(-c * d)
        log_similarities = -keras.activations.relu(self.c) * distances
        # log_similarities = -tf.abs(self.c) * distances  # (B, N)
        # similarities = tf.exp(log_similarities)  # (B, N)
        similarities_norm = tf.nn.softmax(log_similarities)

        # Weighted sum of similarities per class
        probs = tf.matmul(similarities_norm, tf.cast(labels, tf.float32))

        # Weighted sum of similarities per class
        # numerators = tf.matmul(similarities, tf.cast(
        #     self.labels, tf.float32))  # (B, C)
        # denominators = tf.reduce_sum(
        #     similarities, axis=1, keepdims=True)      # (B, 1)
        # epsilon2 = 0
        # probs = numerators / (denominators + epsilon2)  # (B, C)
        return probs

    def get_config(self):
        config = super().get_config()
        config.update({
            "similarity_gradient": self.similarity_gradient
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class ExemplarMemoryLayer(keras.layers.Layer):
    def __init__(self, exemplars=None, labels=None, exemplars_shape=None, labels_shape=None, **kwargs):
        super().__init__(**kwargs)

        if exemplars is not None and labels is not None:
            self.exemplars_shape = exemplars.shape
            self.labels_shape = labels.shape

            self.exemplars = self.add_weight(
                name='exemplars_memory',
                shape=exemplars.shape,
                initializer=tf.constant_initializer(exemplars),
                trainable=False,
            )
            self.labels = self.add_weight(
                name='exemplar_labels',
                shape=labels.shape,
                initializer=tf.constant_initializer(labels),
                trainable=False,
                dtype=tf.int64
            )
        elif exemplars_shape is not None and labels_shape is not None:
            self.exemplars_shape = exemplars_shape
            self.labels_shape = labels_shape

            # Dummy zero-initialized placeholders on deserialization
            self.exemplars = self.add_weight(
                name='exemplars_memory',
                shape=exemplars_shape,
                initializer='zeros',
                trainable=False
            )
            self.labels = self.add_weight(
                name='exemplar_labels',
                shape=labels_shape,
                initializer='zeros',
                trainable=False,
                dtype=tf.int64
            )
        else:
            raise ValueError(
                "Must provide either (exemplars and labels) or (exemplars_shape and labels_shape)")

    def call(self, inputs=None):
        # Simply output stored exemplars and labels
        return self.exemplars, self.labels

    def get_config(self):
        config = super().get_config()
        config.update({
            "exemplars_shape": self.exemplars_shape,
            "labels_shape": self.labels_shape
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class GeneralizedContextModel(keras.Model):
    '''Generalized Context Model implemented in TensorFlow
    '''

    def __init__(self, exemplars=None, labels=None, similarity_gradient=1.0, exemplars_shape=None, labels_shape=None, **kwargs):
        super().__init__(**kwargs)
        if exemplars is not None and labels is not None:
            self.exemplars_shape = exemplars.shape
            self.labels_shape = labels.shape
            self.memory = ExemplarMemoryLayer(exemplars, labels)
        elif exemplars_shape is not None and labels_shape is not None:
            self.exemplars_shape = exemplars_shape
            self.labels_shape = labels_shape
            self.memory = ExemplarMemoryLayer(
                exemplars=None,
                labels=None,
                exemplars_shape=exemplars_shape,
                labels_shape=labels_shape
            )
        else:
            raise ValueError(
                "Need either (exemplars and labels) or (exemplars_shape and labels_shape)")

        self.similarity_gradient = similarity_gradient
        self.distance = WeightedDistanceLayer(self.exemplars_shape[1])
        self.similarity = SimilarityLayer(
            similarity_gradient=similarity_gradient)

    def call(self, x):
        """
        x: Tensor of shape (B, D) - input batch
        Returns: (B, C) - class probabilities
        """
        exemplars, labels = self.memory(None)

        # Weighted Euclidean distance (B, 1, D) - (1, N, D)
        distances = self.distance([x, exemplars])

        # Similarity: s = exp(-c * d)
        # Weighted sum of similarities per class
        probs = self.similarity([distances, labels])
        return probs

    def get_config(self):
        config = super().get_config()
        config.update({
            "exemplars_shape": self.exemplars_shape,
            "labels_shape": self.labels_shape,
            "similarity_gradient": self.similarity_gradient
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class GeneralizedContextModelDifferentiableMemory(keras.Model):
    '''Generalized Context Model implemented in tensorflow
    '''

    def __init__(self, exemplars_dim, labels, similarity_gradient=1.0):
        super().__init__()
        # Trainable weights per feature dimension
        self.exemplars_dim = exemplars_dim
        self.similarity_gradient = similarity_gradient
        self.distance = WeightedDistanceLayer(self.exemplars_dim)
        self.similarity = SimilarityLayer(
            similarity_gradient=self.similarity_gradient)
        self.labels = self.add_weight(
            name='exemplar_labels',
            shape=labels.shape,
            initializer=keras.initializers.Constant(labels),
            trainable=False,
            dtype=tf.int64
        )

    def call(self, x, exemplars):
        """
        x: Tensor of shape (B, D) - input batch
        Returns: (B, C) - class probabilities
        """
        # Weighted Euclidean distance (B, 1, D) - (1, N, D)
        distances = self.distance([x, exemplars])

        # Similarity: s = exp(-c * d)
        # Weighted sum of similarities per class
        probs = self.similarity([distances, self.labels])

        return probs

    def get_config(self):
        config = super().get_config()
        # Convert tensors to numpy arrays to be serializable
        config.update({
            "exemplars_dim": self.exemplars_dim,
            "labels": self.labels.numpy(),
            "similarity_gradient": self.similarity_gradient,
        })
        return config


# @keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class GeneralizedContextModelOld(keras.Model):
    '''Generalized Context Model implemented in tensorflow
    '''

    def __init__(self, exemplars, labels, similarity_gradient=1.0):
        super().__init__()
        self.exemplars_shape = exemplars.shape
        self.labels_shape = labels.shape
        self.similarity_gradient = similarity_gradient
        # Trainable weights per feature dimension
        self.attention_weight_layer = AttentionWeightLayer(exemplars.shape[1])
        self.exemplars = self.add_weight(
            name='exemplars_memory',
            shape=self.exemplars_shape,
            initializer=keras.initializers.Constant(exemplars),
            trainable=False,
        )
        self.labels = self.add_weight(
            name='exemplar_labels',
            shape=self.labels_shape,
            initializer=keras.initializers.Constant(labels),
            trainable=False,
            dtype=tf.int64
        )
        # Learnable scalar
        self.c = self.add_weight(
            name="similarity_gradient",
            shape=(),
            initializer=keras.initializers.Constant(self.similarity_gradient),
            trainable=True,
        )

    def call(self, x):
        """
        x: Tensor of shape (B, D) - input batch
        Returns: (B, C) - class probabilities
        """
        epsilon = 1e-12          # For numerical stability
        # Normalize the weights so they sum to 1 across the feature dimension
        attention_weights_normalized = self.attention_weight_layer(None)

        # Weighted Euclidean distance (B, 1, D) - (1, N, D)
        weighted_diff = tf.square(tf.expand_dims(x, 1) - tf.expand_dims(self.exemplars, 0)) * \
            attention_weights_normalized   # (B, N, D)
        distances = tf.sqrt(tf.reduce_sum(
            weighted_diff, axis=-1) + epsilon)  # (B, N)

        # Similarity: s = exp(-c * d)
        log_similarities = -keras.activations.relu(self.c) * distances
        # log_similarities = -tf.abs(self.c) * distances  # (B, N)
        # similarities = tf.exp(log_similarities)  # (B, N)
        similarities_norm = tf.nn.softmax(log_similarities)

        # Weighted sum of similarities per class
        probs = tf.matmul(similarities_norm, tf.cast(self.labels, tf.float32))

        # Weighted sum of similarities per class
        # numerators = tf.matmul(similarities, tf.cast(
        #     self.labels, tf.float32))  # (B, C)
        # denominators = tf.reduce_sum(
        #     similarities, axis=1, keepdims=True)      # (B, 1)
        # epsilon2 = 0
        # probs = numerators / (denominators + epsilon2)  # (B, C)
        return probs

    def get_config(self):
        config = super().get_config()
        # For serialization, we pass the shapes and initial values
        config.update({
            "exemplars": self.exemplars.numpy(),
            "labels": self.labels.numpy(),
            "similarity_gradient": self.c.numpy().item(),
        })
        return config


def compute_gcm_input_data(data_input_norm, gcm_input, sinfony, transceiver_split, snr, batch_size=32):
    '''Computes GCM input exemplar data from dataset
    '''
    # Flattening for GCM input
    if gcm_input == 2:
        exemplars = np.reshape(
            data_input_norm[0], [data_input_norm[0].shape[0], -1])
    else:
        if transceiver_split == 1:
            sinfony_output_training = sinfony(
                data_input_norm, snr=snr, batch_size=batch_size)
        else:
            sinfony_output_training = sinfony(
                data_input_norm, batch_size=batch_size)
        exemplars = np.reshape(sinfony_output_training, [
                               sinfony_output_training.shape[0], -1])
    return exemplars


def new_optimizer(opt_class=keras.optimizers.SGD, opt_config={"learning_rate": 0.01, "momentum": 0.9}):
    return opt_class(**opt_config)


class MaskingLayer(keras.layers.Layer):
    """
    A Keras layer that applies a (fixed or trainable) mask to the last dimension of the input.

    This is useful for masking certain output dimensions, e.g., suppressing or weighting
    logits or regression outputs in a trainable or static way.

    Args:
        output_dim (int): Number of output dimensions (e.g., number of classes).
        mask_initializer (str or tf.initializer): Initializer for the mask.
            Use "ones" for no initial masking, "zeros" to suppress everything initially, etc.
        trainable (bool): If True, the mask will be learned during training.
        **kwargs: Additional keyword arguments passed to the Layer base class.
    """

    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.mask_initializer = mask
        self.mask = self.add_weight(
            name="mask",
            shape=mask.shape,
            initializer=tf.constant_initializer(self.mask_initializer),
            trainable=False,
        )

    def call(self, inputs):
        return inputs * self.mask  # Broadcasted over batch

    def get_config(self):
        return {"mask": self.mask_initializer, **super().get_config()}


class OutputSelector(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def get_config(self):
        return {"indices": self.indices, **super().get_config()}


def wrap_with_output_masking(base_model, mask):
    """
    Wraps an existing model by applying a masking vector to its output.

    Args:
        base_model (keras.Model): the model whose output will be wrapped
        mask: masks the output tensor

    Returns:
        keras.Model: a new model with same input as base_model, but masked output
    """
    input_tensor = sw.clone_model_inputs(base_model)
    # input_tensor = base_model.input
    output_tensor = base_model(input_tensor)

    # Apply masking or transformation
    masked_output = OutputSelector(mask)(output_tensor)
    # masked_output = MaskingLayer(mask)(output_tensor)

    return keras.Model(inputs=input_tensor, outputs=masked_output, name=base_model.name + "_masked")


def epoch2iterationboundaries(epoch_bound, dataset_size, batch_size):
    ''' Calculates SGD iteration boundaries from epoch boundaries, dataset size and batch size
    '''
    iterations_per_epoch = dataset_size / batch_size
    boundaries = list(np.round(np.array(epoch_bound)
                               * iterations_per_epoch).astype('int'))
    return boundaries


if __name__ == '__main__':

    mt.gpu_select(number=-2, memory_growth=True, cpus=64)
    use_weights = True

    # Choose project/dataset
    # Possible data sets: mnist, cifar10, fraeser, hise, speechcommands, urbansound8k
    wrapper = 'mnist'
    simulation = 'snr'          # snr, memory, working_memory

    load = True
    filename_gcm = 'GCM_Ne1_Na20'
    gcm_input = 0               # 0: sinfony output, 1: last layer input, 2: image input
    # Number of last layer input features used by GCM [only active for gcm_input==1]
    number_features = -1        # -1: all features are used
    # Probabilistic GCM decisions + Optimal policy: True, Optimal policy: False
    gcm_decision_policy = True

    # SINFONY version
    transceiver_split = 1       # image recognition: 0, sinfony: 1
    sinfony_version = 0         # high bandwidth: 0, low bandwidth: 1
    # image_split: One image (False) or multi-image classification based on different views (true)
    image_split = True

    # Training parameters
    snr_training = [-4, 6]      # Default training SNR [-4, 6]
    training_batch_size = 64    # Default: 64
    training_epochs = 1         # Default: 1
    # GCM learning rate schedule
    epoch_bound = [2, 10]
    values = [1.0e-0, 1.0e-1, 1.0e-2]
    learning_rate_schedule = 1.0e-0  # 1.0e-0, None
    opt_class = SGD             # SGD, Adam

    # Joint training
    joint_training = False
    # Parameters updated in alternation or jointly, default: True
    alternating = True
    differentiable_memory = False
    joint_gcm_training_epochs = 1
    joint_sinfony_training_epochs = 1
    alternating_training_iterations = 20
    epoch_bound2 = []
    values2 = [1.0e-3]
    learning_rate_schedule2 = 1.0e-3  # 1.0e-3, None
    opt_class2 = SGD             # SGD, Adam
    # optimizer2 = Adam()

    # Evaluation settings
    validation_rounds = 10      # Default: 10 (100 for fraeser)
    validation_batch_size = 64  # Default: 64
    # SNR Simulation
    snr_range = [10, 20]       # Default: [-30,20]
    snr_step_size = 1           # 1
    # Memory simulation
    # SNR for memory_size simulation, evaluate for training SNR?
    snr_evaluation = [20, 20]   # Default: [20, 20]
    fixed_dataset_per_memory_size = True
    memory_sizes = []           # Automatically logarithmic spacing
    # Working memory simulation
    working_memory_sizes = []
    print('Simulation starts...')

    # Load dataset
    dataset_name = wrapper
    if dataset_name == 'urbansound8k':
        # validation_split needs to be set manually for urbansound8k
        train_input_norm, train_labels, test_input_norm, test_labels = datasets.load_dataset(
            dataset_name, image_split=image_split, validation_split=0.90)
    else:
        train_input_norm, train_labels, test_input_norm, test_labels = datasets.load_dataset(
            dataset_name, image_split=image_split)
    datasets.summarize_dataset(
        train_input_norm, train_labels, test_input_norm, test_labels)
    number_classes, image_shapes = datasets.get_data_properties(
        test_labels, test_input_norm)

    # --------------- TRAINING + EVALUATION -------------------

    if gcm_input == 1:
        last_layer_input = True
    elif gcm_input == 0:
        last_layer_input = False
    else:
        last_layer_input = True
    # Load sinfony wrapper
    path_script = os.path.dirname(os.path.abspath(__file__))
    subpath_results = os.path.join(path_script, 'models', wrapper)
    if gcm_input != 2:
        template_files, _ = sw.template_models(wrapper)
        filename_sinfony = sw.select_sinfony(template_files=template_files, image_split=image_split,
                                             transceiver_split=transceiver_split, sinfony_version=sinfony_version)
        pathfile_sinfony = os.path.join(subpath_results, filename_sinfony)
        sinfony = sw.select_wrapper(transceiver_split, number_classes,
                                    image_shapes, path=pathfile_sinfony, last_layer_input=last_layer_input)
        sinfony.set_snr(snr_training)
        if gcm_input == 1 and number_features != -1:
            # Select k most important features and set rest to zero
            weights_last_layer = sinfony.model_full.get_weights()[-2]
            feature_importance = np.sum(np.abs(weights_last_layer), axis=1)
            top_k_indices = np.argpartition(
                feature_importance, -number_features)[-number_features:]
            # weights_k_important = np.zeros(
            #     weights_last_layer.shape[0], dtype='float32')
            # weights_k_important[top_k_indices] = 1
            sinfony.model = wrap_with_output_masking(
                sinfony.model, top_k_indices)
            # sinfony_weights = sinfony.model_full.get_weights()
            # sinfony_weights[-2] = weights_k_important
            # sinfony.model.set_weights(sinfony_weights)
    else:
        sinfony = []
        filename_sinfony = ''

    # Calculate iteration boundaries for learning rate schedule
    if not learning_rate_schedule:
        boundaries = epoch2iterationboundaries(
            epoch_bound, train_input_norm[0].shape[0], training_batch_size)
        learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
    if not learning_rate_schedule2:
        boundaries2 = epoch2iterationboundaries(
            epoch_bound2, train_input_norm[0].shape[0], training_batch_size)
        learning_rate_schedule2 = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries2, values2)
    opt_config = {"learning_rate": learning_rate_schedule, "momentum": 0.9}
    optimizer = new_optimizer(opt_class=opt_class, opt_config=opt_config)
    optimizer2 = opt_class2(
        learning_rate=learning_rate_schedule2, momentum=0.9)

    # Create filenames
    if gcm_input != 2 and differentiable_memory is True:
        filename_gcm = filename_gcm + '_differentiable'
    if gcm_input == 0:
        filename = filename_gcm + '+' + filename_sinfony
    elif gcm_input == 1:
        filename = filename_gcm + '_Nfeatures' + \
            str(number_features) + '+' + filename_sinfony
    else:
        filename = filename_gcm + '_' + dataset_name
    if joint_training is True and gcm_input != 2:
        filename = filename + '+joint_training'
    pathfile = os.path.join(path_script, subpath_results, filename)

    results = {}
    saveobj = savemodule(form='npz')
    if simulation == 'memory':
        fn_simulation = '_memory'
    elif simulation == 'working_memory':
        fn_simulation = '_working_memory'
    else:
        fn_simulation = ''
    filename_prefix = ''
    filename_suffix = '_results'
    pathfile_results = os.path.join(path_script, subpath_results,
                                    filename_prefix + filename + filename_suffix + fn_simulation)
    pathfile_results_opt = pathfile_results + '_opt'

    # Main simulation
    accuracy_opt = 0
    if simulation == 'memory':
        print('Memory Size Simulation')

        accuracy, loss, memory_sizes = model_evaluation.evaluate_gcm_memory(gcm_input, sinfony, transceiver_split, train_input_norm, train_labels, snr_training, test_input_norm, test_labels, snr_evaluation, training_epochs=training_epochs, batch_size=training_batch_size,
                                                                            validation_batch_size=validation_batch_size, opt_class=opt_class, opt_config=opt_config, validation_rounds=validation_rounds, gcm_decision_policy=gcm_decision_policy, fixed_dataset_per_memory_size=fixed_dataset_per_memory_size, memory_sizes=memory_sizes)
        if gcm_decision_policy is True:
            accuracy_opt = accuracy[1]
            accuracy = accuracy[0]

    elif simulation == 'snr' or simulation == 'working_memory':
        if simulation == 'snr':
            print('SNR Simulation')
        else:
            print('Working Memory Capacity Simulation')

        # GCM input computation
        exemplars = compute_gcm_input_data(
            train_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
        exemplar_labels = train_labels
        if load is False or gcm_input == 2:
            test_exemplars = compute_gcm_input_data(
                test_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
            test_exemplars_labels = test_labels
        else:
            test_exemplars = 0
            test_exemplars_labels = 0
        # Easily OOM if not enough RAM! Lines are for debugging
        # number_exemplars = 10000  # 60000
        # exemplars = exemplars[0:number_exemplars, ...]
        # exemplar_labels = exemplar_labels[0:number_exemplars, ...]

        # GCM model and training - Only GCM training first
        gcm = GeneralizedContextModel(
            exemplars, exemplar_labels, similarity_gradient=1.0)
        # tf.debugging.check_numerics(
        #     gcm.attention_weights, message="input contains NaNs or infs")
        gcm.compile(optimizer=optimizer,
                    loss='categorical_crossentropy', metrics=['accuracy'])

        if load is False:
            # GCM training
            print('GCM training starts...')
            gcm.fit(exemplars, exemplar_labels, validation_data=(
                test_exemplars, test_exemplars_labels), batch_size=training_batch_size, epochs=training_epochs)
            print('GCM training complete!')
            # Save the GCM model
            print('Saving model...')
            sinfony_io.try_save(gcm, pathfile, to_weights=use_weights)
            print('Model saved.')

        if load is True and joint_training is False:
            # Load existing GCM model:
            print('Loading model...')
            gcm = sinfony_io.try_load(gcm, pathfile, from_weights=use_weights)
            print('Model loaded.')

        if gcm_input != 2:
            # Combine SINFONY + GCM for evaluation or joint training
            # Compose them into one model
            if differentiable_memory is True and joint_training is True:
                # GCM version with differentiable memory -> trained end-to-end
                # NOTE: Too complex for gradient computation with weight sharing
                sinfony.model.trainable = True
                gcm_diff = GeneralizedContextModelDifferentiableMemory(
                    exemplars.shape[1], exemplar_labels, similarity_gradient=1.0)
                gcm_diff.set_weights(
                    gcm.get_weights()[1:][:-2] + gcm.get_weights()[1:][-2:])
                input_image = sw.clone_model_inputs(sinfony.model)
                features = sinfony.model(input_image)

                image_memory = []
                for train_input_norm_item in train_input_norm:
                    image_memory.append(tf.constant(train_input_norm_item))
                features_memory = sinfony.model(image_memory)
                gcm_output = gcm_diff(features, features_memory)
                # Final model
                sinfony_gcm = keras.Model(
                    inputs=input_image, outputs=gcm_output)
            else:
                input_image = sw.clone_model_inputs(sinfony.model)
                # input_image = sinfony.model.input
                features = sinfony.model(input_image)
                # Input reshaper
                # features_flat = keras.layers.Flatten()(features)
                gcm_output = gcm(features)
                # Final model
                sinfony_gcm = keras.Model(
                    inputs=input_image, outputs=gcm_output)
        else:
            sinfony_gcm = gcm

        if load is True and joint_training is True:
            # Load existing model:
            print('Loading model...')
            # sinfony_gcm = keras.models.load_model(pathfile)
            sinfony_gcm = sinfony_io.try_load(
                sinfony_gcm, pathfile, from_weights=use_weights)
            print('Model loaded.')

        sinfony_gcm.compile(optimizer=optimizer2,
                            loss='categorical_crossentropy', metrics=['accuracy'])

        if load is False and joint_training is True:
            if differentiable_memory is True:
                print('Joint training with differentiable memory starts...')
                history_sinfony = sinfony_gcm.fit(train_input_norm, exemplar_labels, validation_data=(
                    test_input_norm, test_exemplars_labels), batch_size=training_batch_size, epochs=alternating_training_iterations)
            else:
                print('Joint alternating training starts...')
                start_time = time.time()
                for idx_alternate in range(0, alternating_training_iterations):
                    # SINFONY + GCM Training
                    # Note: Keeps optimizer state between iterations
                    # Trainable sinfony model weights
                    sinfony.model.trainable = True
                    # Freeze GCM since memory is fixed
                    if alternating is True:
                        gcm.trainable = False
                    sinfony_gcm.compile(optimizer=optimizer2,
                                        loss='categorical_crossentropy', metrics=['accuracy'])
                    history_sinfony = sinfony_gcm.fit(train_input_norm, exemplar_labels, validation_data=(
                        test_input_norm, test_exemplars_labels), batch_size=training_batch_size, epochs=joint_gcm_training_epochs)
                    # Calculate new GCM exemplars for memory
                    exemplars = compute_gcm_input_data(
                        train_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
                    test_exemplars = compute_gcm_input_data(
                        test_input_norm, gcm_input, sinfony, transceiver_split, snr_training, batch_size=validation_batch_size)
                    exemplar_var = [
                        v for v in gcm.weights if "exemplars_memory" in v.name][0]
                    exemplar_var.assign(exemplars)
                    # gcm.get_layer('exemplars_labels').set_weights(
                    #     [exemplar_labels])
                    # Freeze the sinfony model weights
                    sinfony.model.trainable = False
                    # Train GCM based on new exemplars
                    if alternating is True:
                        gcm.trainable = True
                    gcm.compile(optimizer=optimizer,
                                loss='categorical_crossentropy', metrics=['accuracy'])
                    history_gcm = gcm.fit(exemplars, exemplar_labels, validation_data=(
                        test_exemplars, test_exemplars_labels), batch_size=training_batch_size, epochs=joint_sinfony_training_epochs)
                    # Free RAM, otherwise it accumulates
                    gc.collect()
                    print(
                        f"Iteration: {idx_alternate + 1}/{alternating_training_iterations}, CE: {history_gcm.history['val_loss'][-1]:.4f}, Acc: {history_gcm.history['val_accuracy'][-1]:.2f}, Time: {print_time(time.time() - start_time)}, RAM: {get_ram():.2f} GB")

            # Save the model
            print('Saving model...')
            sinfony_io.try_save(sinfony_gcm, pathfile, to_weights=use_weights)
            sinfony_gcm.save(pathfile)
            print('Model saved.')

        if simulation == 'working_memory':

            if gcm_input == 2:
                accuracy, loss, working_memory_sizes = model_evaluation.evaluate_gcm_working_memory(gcm, test_exemplars, test_exemplars_labels, validation_rounds=validation_rounds,
                                                                                                    batch_size=validation_batch_size, gcm_decision_policy=gcm_decision_policy, working_memory_sizes=working_memory_sizes)
            else:
                sinfony.set_snr(snr_evaluation)
                accuracy, loss, working_memory_sizes = model_evaluation.evaluate_gcm_working_memory(sinfony_gcm, test_input_norm, test_labels, validation_rounds=validation_rounds,
                                                                                                    batch_size=validation_batch_size, gcm_decision_policy=gcm_decision_policy, working_memory_sizes=working_memory_sizes)
            if gcm_decision_policy is True:
                accuracy_opt = accuracy[1]
                accuracy = accuracy[0]

        else:
            # Evaluation of model
            snrs = mop.snr_range2snrlist(snr_range, snr_step_size)
            print('Evaluate model...')
            print(filename)
            # Evaluate model for different SNRs
            if transceiver_split == 1:
                # SINFONY/RL-SINFONY
                accuracy, loss = model_evaluation.evaluate_sinfony(sinfony_gcm, test_input_norm, test_labels,
                                                                   snrs=snrs, validation_rounds=validation_rounds, gcm_decision_policy=gcm_decision_policy)
                if gcm_decision_policy is True:
                    accuracy_opt = accuracy[1]
                    accuracy = accuracy[0]
            else:
                if gcm_input == 2:
                    # GCM image recognition
                    accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                        gcm, test_exemplars, test_exemplars_labels, with_gcm=gcm_decision_policy)
                else:
                    # Standard image recognition: Evaluate model accuracy once for test data
                    accuracy_i, loss_i = model_evaluation.evaluate_image_classifier(
                        sinfony_gcm, test_input_norm, test_labels, with_gcm=gcm_decision_policy)
                # Independent from SNR / constant, but plotted over SNR range
                loss = np.array(loss_i) * np.ones(snrs.shape)
                if gcm_decision_policy is True:
                    accuracy = np.array(accuracy_i[0]) * np.ones(snrs.shape)
                    accuracy_opt = np.array(
                        accuracy_i[1]) * np.ones(snrs.shape)
                else:
                    accuracy = np.array(accuracy_i) * np.ones(snrs.shape)
                print(f'> {accuracy_i * 100.0:.3f}')
    else:
        print('Simulation not implemented')
        loss = 0
        accuracy = 0

    # Show performance curve
    if simulation == 'memory':
        x_axis = memory_sizes
    elif simulation == 'working_memory':
        x_axis = working_memory_sizes
    else:
        x_axis = snrs

    if simulation == 'memory':
        xlabel = 'Memory Size'
    elif simulation == 'working_memory':
        xlabel = 'Working Memory Size'
    else:
        xlabel = 'SNR'
    plt.figure(1)
    plt.semilogy(x_axis, 1 - accuracy)
    if gcm_decision_policy is True:
        plt.semilogy(x_axis, 1 - accuracy_opt)
    plt.xlabel(xlabel)
    plt.ylabel(
        'semantic performance measure: classification error rate')
    plt.figure(2)
    plt.semilogy(x_axis, loss)
    plt.xlabel(xlabel)
    plt.ylabel('Crossentropy Loss')

    # Save evaluation
    print('Save evaluation...')
    results['val_loss'] = loss
    results['val_acc'] = accuracy
    if simulation == 'memory':
        results['memory_size'] = memory_sizes
    elif simulation == 'working_memory':
        results['working_memory_size'] = working_memory_sizes
    else:
        results['snr'] = snrs
    saveobj.save(pathfile_results, results)
    if gcm_decision_policy is True:
        results['val_acc'] = accuracy_opt
        saveobj.save(pathfile_results_opt, results)
    print('Evaluation saved.')
