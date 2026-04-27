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
import numpy as np
# import tensorflow.keras as keras
import keras
import utilities.my_math_operations as mop


def ensure_directory_exists(folder_path):
    """
    Checks whether a directory exists, and creates it if it doesn't.

    Args:
        folder_path (str): Path to the target directory.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")


def try_load_weights(model, pathfile):
    """
    Try loading weights into a model from pathfile trying common Keras extensions.
    First tries the new '.weights.h5' format, then falls back to '_weights.h5'.

    Returns:
        The model with loaded weights.
    """
    tried_paths = [pathfile + '.weights.h5', pathfile + '_weights.h5', pathfile +
                   '.h5', os.path.join(pathfile, os.path.basename(os.path.dirname(pathfile)))]

    for weights_path in tried_paths:
        try:
            if os.path.exists(weights_path):
                print(f"Trying to load weights from: {weights_path}")
                model.load_weights(weights_path)
                print(f"✅ Successfully loaded weights from: {weights_path}")
                return model
        except Exception as e:
            print(f"❌ Failed to load weights from {weights_path}: {e}")

    raise FileNotFoundError(
        f"Weights not found at '{pathfile}'")


def try_load_model(pathfile, custom_objects=None, use_tfsm_layer=True):
    """
    Try loading a model from pathfile trying common Keras extensions.
    If loading fails due to Keras 3 / SavedModel incompatibility and the path
    is a SavedModel directory, optionally fall back to keras.layers.TFSMLayer
    (inference-only).

    Returns:
        A Keras model (loaded) OR a TFSMLayer (fallback) if use_tfsm_layer=True.
    """
    tried_paths = [pathfile + '.keras', pathfile +
                   '.hdf5', pathfile + '.h5', pathfile]

    for model_path in tried_paths:
        try:
            if os.path.exists(model_path):
                print(f"Trying to load model from: {model_path}")
                if not custom_objects:
                    model = keras.models.load_model(model_path)
                else:
                    model = keras.models.load_model(
                        model_path, custom_objects=custom_objects)
                print(f"Successfully loaded model from: {model_path}")
                return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            # Fallback: TensorFlow SavedModel directory -> inference-only TFSMLayer
            if use_tfsm_layer and os.path.isdir(model_path):
                sm_pb = os.path.join(model_path, "saved_model.pb")
                if os.path.exists(sm_pb):
                    try:
                        print(
                            f"Falling back to TFSMLayer for SavedModel dir: {model_path}")
                        model = keras.layers.TFSMLayer(
                            model_path, call_endpoint="serving_default")
                        return model
                    except Exception as e2:
                        print(
                            f"Failed to load SavedModel via TFSMLayer from {model_path}: {e2}")

    raise FileNotFoundError(
        f"Model not found at '{pathfile}'")


def try_load(model, pathfile, from_weights=True, custom_objects=None, use_tfsm_layer=True):
    """
    Try loading a model from pathfile.
    First tries to load weights into the provided model (robust, version-independent),
    then falls back to loading the full model (fragile, version-dependent).

    Returns:
        A Keras model with loaded weights, or a TFSMLayer as last resort.
    """
    # 1st: Try loading weights into the provided model (preferred)
    if from_weights is True:
        try:
            model = try_load_weights(model, pathfile)
        except FileNotFoundError:
            pass
    else:
        # 2nd: Try loading full model, ignoring the provided model
        model = try_load_model(
            pathfile, custom_objects=custom_objects, use_tfsm_layer=use_tfsm_layer)
    return model


def try_save(model, pathfile, to_weights=True):
    """
    Try saving model weights and full model to pathfile.
    """
    ensure_directory_exists(os.path.dirname(pathfile))
    if to_weights is True:
        try:
            print('Saving model weights...')
            model.save_weights(pathfile + '.weights.h5')
            print('✅ Model weights saved.')
        except Exception as e:
            print(f'❌ Failed to save model weights: {e}')
    else:
        try:
            print('Saving model...')
            model.save(pathfile + '.keras')
            print('✅ Model saved.')
        except Exception as e:
            print(f'❌ Failed to save model: {e}')


def find_layer_by_name(model, name):
    """Find a (sub)layer by name in a normal Keras Model."""
    for layer in getattr(model, "layers", []):
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):  # nested model
            found = find_layer_by_name(layer, name)
            if found is not None:
                return found
    return None


def find_variable_by_name(obj, name, endswith=True):
    """
    Find a Variable in an object (e.g., TFSMLayer or Keras model) by name.

    endswith=True is recommended for SavedModels because variables often get prefixes.
    """
    vars_ = getattr(obj, "variables", None)
    if vars_ is None:
        return None

    for v in vars_:
        if (endswith and v.name.endswith(name)) or ((not endswith) and v.name == name):
            return v
    return None


def is_assignable_variable(x):
    return hasattr(x, "assign") and hasattr(x, "dtype") and hasattr(x, "shape")


def get_noise_target(model,
                     layer_name="gaussian_noise2",
                     var_suffix="stddev:0"):
    """
    Returns an object that can be passed to set_noise_variance():
      - a Keras noise layer (supports set_weights), or
      - a Variable (from a TFSMLayer / SavedModel), or
      - None if nothing suitable found.
    """
    # Prefer real Keras layer if the model has a layer graph
    if hasattr(model, "layers"):
        layer = find_layer_by_name(model, layer_name)
        if layer is not None and hasattr(layer, "set_weights"):
            return layer
    else:
        # Fallback: look for a variable (works for TFSMLayer too)
        v = find_variable_by_name(model, var_suffix, endswith=True)
        if v is not None and is_assignable_variable(v):
            return v
        else:
            print('Could not extract noise layer!')

    return None


def set_noise_variance(target, snr):
    """
    Set noise level on target;
      - a Keras noise layer via set_weights([..])  (expects stddev vector)
      - or a Variable (e.g., from TFSMLayer) via assign()

    snr: signal-to-noise ratio
    """
    stddev = mop.snr2standard_deviation(snr)
    stddev_vec = np.array([stddev, stddev], dtype='float32')

    # Case A: Keras layer with set_weights
    if hasattr(target, "set_weights"):
        target.set_weights([stddev_vec])
        # evaluated_model.get_layer('gaussian_noise2').set_weights([sigma_test])
        # evaluated_model.layers[-2].set_weights([sigma_test])
        return
    # Case B: Variable
    elif is_assignable_variable(target):
        target.assign(keras.ops.cast(stddev_vec, target.dtype))
        return

    raise TypeError(
        f"Unsupported target type for setting noise variance: {type(target)}")
