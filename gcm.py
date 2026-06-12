#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40 2025

@author: beck
GCM implemented in tensorflow

Belongs to simulation framework for numerical results of the articles:
1. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework,” IEEE Open Journal of the Communications Society, vol. 7, pp. 748-768, Jan 2026. https://doi.org/10.1109/OJCOMS.2026.3652845
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# import tensorflow as tf
# import tensorflow.keras as keras
import keras
import numpy as np

import sinfony_wrapper as sw


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
            keras.ops.sum(positive_weights, axis=-1,
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
        weighted_diff = keras.ops.square(keras.ops.expand_dims(x, 1) - keras.ops.expand_dims(exemplars, 0)) * \
            attention_weights   # (B, N, D)
        distances = keras.ops.sqrt(keras.ops.sum(
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
        # log_similarities = -keras.ops.abs(self.c) * distances  # (B, N)
        # similarities = keras.ops.exp(log_similarities)  # (B, N)
        similarities_norm = keras.ops.softmax(log_similarities)

        # Weighted sum of similarities per class
        probs = keras.ops.matmul(
            similarities_norm, keras.ops.cast(labels, 'float32'))

        # Weighted sum of similarities per class
        # numerators = keras.ops.matmul(similarities, keras.ops.cast(
        #     self.labels, 'float32'))  # (B, C)
        # denominators = keras.ops.sum(
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
                initializer=keras.initializers.Constant(exemplars),
                trainable=False,
            )
            self.labels = self.add_weight(
                name='exemplar_labels',
                shape=labels.shape,
                initializer=keras.initializers.Constant(labels),
                trainable=False,
                dtype='int64'
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
                dtype='int64'
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
            dtype='int64'
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
            dtype='int64'
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
        weighted_diff = keras.ops.square(keras.ops.expand_dims(x, 1) - keras.ops.expand_dims(self.exemplars, 0)) * \
            attention_weights_normalized   # (B, N, D)
        distances = keras.ops.sqrt(keras.ops.sum(
            weighted_diff, axis=-1) + epsilon)  # (B, N)

        # Similarity: s = exp(-c * d)
        log_similarities = -keras.activations.relu(self.c) * distances
        # log_similarities = -keras.ops.abs(self.c) * distances  # (B, N)
        # similarities = keras.ops.exp(log_similarities)  # (B, N)
        similarities_norm = keras.ops.softmax(log_similarities)

        # Weighted sum of similarities per class
        probs = keras.ops.matmul(
            similarities_norm, keras.ops.cast(self.labels, 'float32'))

        # Weighted sum of similarities per class
        # numerators = keras.ops.matmul(similarities, keras.ops.cast(
        #     self.labels, 'float32'))  # (B, C)
        # denominators = keras.ops.sum(
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


class MaskingLayer(keras.layers.Layer):
    """
    A Keras layer that applies a (fixed or trainable) mask to the last dimension of the input.

    This is useful for masking certain output dimensions, e.g., suppressing or weighting
    logits or regression outputs in a trainable or static way.

    Args:
        output_dim (int): Number of output dimensions (e.g., number of classes).
        mask_initializer (str or keras.initializer): Initializer for the mask.
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
            initializer=keras.initializers.Constant(self.mask_initializer),
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
        return keras.ops.take(inputs, self.indices, axis=-1)

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
