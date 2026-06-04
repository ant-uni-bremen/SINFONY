#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 01 17:33 2025

@author: beck
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

import os
import tensorflow as tf
# import tensorflow.keras as keras
# import keras
import utilities.my_training as mf

import sinfony_io

# import gcm
from gcm import GeneralizedContextModel, GeneralizedContextModelOld, AttentionWeightLayer
from keras.optimizers import SGD
import sinfony_architectures.resnet as resnet
# import sinfony_architectures.resnet_sinfony as resnet_sinfony

# print(keras.saving.get_custom_objects())

from collections import defaultdict
import re
import shutil
import traceback
import logging


def shift_layer_index_to_zero(root, prefix="tx_layer"):
    """
    Renames tx_layer{N}_k -> tx_layer{N-1}_k (e.g. tx_layer1_0 -> tx_layer0_0)
    recursively. Uses layer._name (hack).
    """
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)(_.+)?$")

    def walk(obj):
        for layer in getattr(obj, "layers", []):
            m = pat.match(layer.name)
            if m:
                n = int(m.group(1))
                rest = m.group(2) or ""
                if n > 0:
                    new_name = f"{prefix}{n - 1}{rest}"
                    print(f"{layer.name} -> {new_name}")
                    layer._name = new_name
            if hasattr(layer, "layers") and layer.layers:
                walk(layer)

    walk(root)


def rewrite_weight_handle_names(root):
    """Rewrite w._handle_name based on current layer.name and weight order."""
    # collect unique layer objects recursively
    def collect(obj, out):
        for layer in getattr(obj, "layers", []):
            if layer not in out:
                out.append(layer)
            if hasattr(layer, "layers") and layer.layers:
                collect(layer, out)

    layers = []
    collect(root, layers)

    for layer in layers:
        for i, w in enumerate(layer.weights):
            base = w.name.split("/")[-1].split(":")[0]  # 'kernel'/'bias'/...
            w._handle_name = f"{layer.name}/{base}_{i}"


def model_fix(model):
    '''Fix old model saves with nested layers and unique variable names'''

    # 🔹 rekursiv alle Layer sammeln
    def collect_layers(layer, layers):
        if layer not in layers:
            layers.append(layer)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                collect_layers(sub, layers)

    all_layers = []
    collect_layers(model, all_layers)

    # 🔹 ursprüngliche Namen merken
    original_names = {layer: layer.name for layer in all_layers}

    # 🔹 Zählen der Duplikate
    name_total = defaultdict(int)
    for orig_name in original_names.values():
        name_total[orig_name] += 1

    # 🔹 umbenennen nur bei mehrfachen Layern
    name_idx = defaultdict(int)
    for layer in all_layers:
        orig_name = original_names[layer]

        # Layernamen fixen
        if name_total[orig_name] > 1:
            idx = name_idx[orig_name]
            layer._name = f"{orig_name}_{idx}"
            name_idx[orig_name] += 1

        # Variablen eindeutige _handle_name geben
        for i, w in enumerate(layer.weights):
            # z. B. tx_layer1_0/kernel_0
            w._handle_name = f"{layer.name}/{w.name.split('/')[-1].split(':')[0]}_{i}"

    return model


def print_layers(layer, indent=0):
    prefix = "  " * indent
    print(f"{prefix}- {layer.name} ({layer.__class__.__name__})")

    if hasattr(layer, "layers"):
        for sub in layer.layers:
            print_layers(sub, indent + 1)


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():
    # Maximum logging
    # tf.get_logger().setLevel(logging.DEBUG)
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models', 'mnist')

    # Choose the format you want to test
    model_ext = ''  # '.keras' or '.hdf5'
    save_weights = True
    delete_models = False
    keras3 = False

    # Loop through all files in the directory
    for fname in os.listdir(model_dir):
        if fname.endswith(model_ext):
            if not fname.endswith('.npz') and not fname.endswith('.h5') and not fname.endswith('.yaml') and not fname.startswith('backup'):
                # fname = 'ResNet14_MNIST4_Ne20_snr-4_6'
                # fname = 'ResNet14_MNIST4_3layer_snr-4_6'
                # fname = 'sinfony14_MNIST_ntx14_Ne20_snr-4_6_human_test'
                model_path = os.path.join(model_dir, fname)

                if keras3:
                    weight_ending = '.weights.h5'
                else:
                    weight_ending = '_weights.h5'
                weight_path = os.path.join(model_dir, fname + weight_ending)
                try:
                    print(f"Testing model load: {model_path}")
                    # gcm = GeneralizedContextModel(
                    #     exemplars, exemplar_labels, similarity_gradient=1.0)
                    # sinfony, filename_sinfony = sw.select_sinfony(path=subpath_results, template_files=template_files, image_split=image_split,
                    #                                             transceiver_split=transceiver_split, sinfony_version=sinfony_version, last_layer_input=last_layer_input)
                    model = sinfony_io.try_load_model(model_path,
                                                      custom_objects={
                                                          "GeneralizedContextModel": GeneralizedContextModel,
                                                          "AttentionWeightLayer": AttentionWeightLayer,
                                                          "SGD": SGD,
                                                          "Residual": resnet.Residual,
                                                          "ResnetBlock": resnet.ResnetBlock,
                                                          "ResidualBottleneck": resnet.ResidualBottleneck,
                                                          "GaussianNoise2": mf.GaussianNoise2,
                                                      },
                                                      )
                    # Fix model
                    model2 = model_fix(model)
                    # shift_layer_index_to_zero(model2, prefix="tx_layer")
                    # shift_layer_index_to_zero(model2, prefix="rx_layer")
                    # rewrite_weight_handle_names(model2)
                    # print_layers(model2)

                    # Optionally summarize or inspect the model
                    # model.summary()
                    # print_layers(model)
                    # model_weight_names = [w.name for w in model.weights]
                    if save_weights is True:
                        model2.save_weights(weight_path)
                        print("✅ Save successful.\n")
                    else:
                        model2.load_weights(weight_path, by_name=True)
                        print("✅ Load successful.\n")

                    # Delete the folder
                    if delete_models is True:
                        if os.path.exists(model_path):
                            if os.path.isfile(model_path):
                                os.remove(model_path)
                                print("Deleted model file.\n")
                            else:
                                shutil.rmtree(model_path)
                                print("Deleted model folder.\n")
                        else:
                            print("Path does not exist.\n")

                except Exception as e:
                    print(f"❌ Failed to load or save model {fname}: {e}\n")
                    # # Vollständiger Traceback mit allen Zeilen
                    # traceback.print_exc()

                    # # Ursache der Exception
                    # cause = e.__cause__
                    # while cause is not None:
                    #     print(f"\nVerursacht durch: {cause}")
                    #     traceback.print_exception(
                    #         type(cause), cause, cause.__traceback__)
                    #     cause = cause.__cause__
