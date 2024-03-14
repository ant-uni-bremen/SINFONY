#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 16:06:40 2024

@author: beck
Module for loading and preprocessing numerous datasets

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import os
import numpy as np
from matplotlib import pyplot as plt
# Pandas required for dataset import
import pandas as pd

# Tensorflow 2 packages
import tensorflow as tf

# Own packages
import my_training as mt


# Dataset functions
def load_dataset(dataset='mnist', validation_split=0.85, image_split=True):
    '''Load dataset
    mnist: Handwritten digits 0-9
    cifar10: Images of animals, vehicles, ..., with 10 classes
    fasion_mnist: Like mnist but with fashion images
    hirise: Images from Martian surface with crater classes, etc. Only available after download of hirise dataset
    hirisecrater: Like hirise, but combines all classes except for craters in one class
    fraeser: Images from bime (processing/production) with decent and damaged tools
    validation_split: If there is no explicit definition of the validation set, use this split ratio
    image_split: If the dataset images contain more than one image, then we can split them into multiple images
    '''
    dataset = dataset.lower()
    # Load dataset
    if dataset == 'cifar10':
        (train_input, train_labels), (validation_input,
                                      validation_labels) = tf.keras.datasets.cifar10.load_data()
        train_input = [train_input]
        validation_input = [validation_input]
    elif dataset == 'mnist':
        (train_input, train_labels), (validation_input,
                                      validation_labels) = tf.keras.datasets.mnist.load_data()
        # Reshape dataset to have a single color channel
        train_input = [train_input[..., np.newaxis]]
        validation_input = [validation_input[..., np.newaxis]]
    elif dataset == 'fashion_mnist':
        (train_input, train_labels), (validation_input,
                                      validation_labels) = tf.keras.datasets.fashion_mnist.load_data()
        # Reshape dataset to have a single color channel
        train_input = [train_input[..., np.newaxis]]
        validation_input = [validation_input[..., np.newaxis]]
    elif dataset[0:6] == 'hirise':
        # Extract logic from dataset name
        if dataset[:12] == 'hirisecrater':
            hirisecrater = True
            if dataset[12:].isdigit():
                resolution = int(dataset[12:])
            else:
                resolution = None
        else:
            hirisecrater = False
            if dataset[6:].isdigit():
                resolution = int(dataset[6:])
            else:
                resolution = None
        train_input, validation_input, train_labels, validation_labels, _, _ = load_dataset_hirise(
            resolution=resolution, hirisecrater=hirisecrater)
    elif dataset[0:7] == 'fraeser':
        # Extract resolution from dataset name
        if dataset[7:].isdigit():
            resolution = int(dataset[7:])
        else:
            resolution = None
        train_input, validation_input, train_labels, validation_labels = load_dataset_tools(
            resolution=resolution, image_split=image_split, number_augmentations=0, validation_split=validation_split, validation_diverse=False, shuffle=False)
    elif dataset[0:4] == 'hise':
        if dataset[4:].isdigit():
            resolution = int(dataset[4:])
            # TODO: HiSE dataset could be named differently...
            dataset_hise = 'HiSE' + dataset[4:]
        else:
            resolution = None
            dataset_hise = None
        train_input, validation_input, train_labels, validation_labels = load_dataset_hise(
            dataset=dataset_hise, resolution=resolution, validation_split=validation_split, number_augmentations=0)
    else:
        print('Dataset not available.')
    # One hot encode target values
    train_labels = tf.keras.utils.to_categorical(train_labels)
    validation_labels = tf.keras.utils.to_categorical(validation_labels)
    return train_input, train_labels, validation_input, validation_labels


def load_dataset_tools(resolution=None, image_split=True, number_augmentations=0, validation_split=0.9, validation_diverse=False, shuffle=False):
    '''Load and preprocess tool dataset (fraeser)
    resolution: Targeted image resolution
    image_split: Split the image into two original tool images from the top and from the side
    number_augmentations: Number of image augmentations, rotated and flipped
    validation_split: Validation split
    Experimental
    validation_diverse: Activate customized and fixed validation split to diversify the validation data
    shuffle: Shuffle validation dataset
    '''
    # Select subdirectory where images are saved
    data_directory = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'Datasets', 'Fraeser_Aufnahmen_220824_Inkl_Verkippt')
    data_file = os.path.join(data_directory, 'Bildliste.CSV')
    dataset_sheet = pd.read_csv(data_file, sep=";", header=None,
                                engine='python')  # dataset_sheet.loc[:, 0]
    # Expert labeling based on all 8/16 images
    labels_list = dataset_sheet.loc[1:, ...].sort_values(
        by=2, ascending=True).loc[:, 5].to_numpy().astype('int')
    # Expert labeling based on individual images
    dataset_sheet2 = pd.read_csv(os.path.join(data_directory, 'Bildliste_einzeln_bewertet_laden.txt'), sep="\t", header=None,
                                 engine='python')
    labels_list2 = dataset_sheet2.loc[:, ...].sort_values(
        by=2, ascending=True).loc[:, 5].to_numpy().astype('int')
    accuracy_individual = np.sum(
        labels_list == labels_list2) / labels_list.shape[0]
    print(f'Individual image classification accuracy: {accuracy_individual}')
    # First load with full resolution
    full_resolution = (705, 380)
    # Two images are captured as one -> split in first dimension at 218
    split_pixel = 218
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        labels=labels_list.tolist(),
        label_mode='int',
        color_mode='grayscale',
        # validation_split = 0.2,
        shuffle=False,
        image_size=full_resolution,
        crop_to_aspect_ratio=True,
    )
    # Convert into numpy format
    data_input = np.concatenate([x for x, _ in train_dataset], axis=0)
    data_labels = np.concatenate([y for _, y in train_dataset], axis=0)

    if image_split:
        # Split two in images in one image into two images
        data_input_split = [data_input[:, :split_pixel, ...],
                            data_input[:, split_pixel:, ...]]  # data_input
    else:
        # Two images remain one image
        data_input_split = [data_input]

    # Lower resolution for each image
    if resolution is None:
        image_resolution = full_resolution
    else:
        image_resolution = (0, resolution)

    # Set resolution of second dimension but keep aspect ratio, first dimension (large one) is adapted
    if image_resolution != full_resolution:
        if image_split:
            image1_scale_xdim = int(
                split_pixel / full_resolution[1] * image_resolution[1])
            image2_scale_xdim = int(
                (full_resolution[0] - split_pixel) / full_resolution[1] * image_resolution[1])
            data_input_split = [resize_image(data_input_split[0], image1_scale_xdim, image_resolution[1]), resize_image(
                data_input_split[1], image2_scale_xdim, image_resolution[1])]
        else:
            image1_scale_xdim = int(
                full_resolution[0] / full_resolution[1] * image_resolution[1])
            data_input_split = [resize_image(
                data_input_split[0], image1_scale_xdim, image_resolution[1])]

    # Shuffle training and validation dataset -> avoids reproducability
    if shuffle:
        data_input_split, data_labels = mt.shuffle_dataset(
            data_input_split, data_labels)

    # Split into training and validation data

    if validation_diverse is True:
        # Customized and fixed validation split to diversify the validation data
        dataset_size = data_input_split[0].shape[0]
        # Fraeser: 8 images of one tool -> use one image per tool for validation
        # Then: validation_split = 1 - 1/8 = 0.875
        validation_indices = np.arange(0, int(
            dataset_size / 8)) * 8 + np.arange(0, int(dataset_size / 8)) % 8
        train_input, validation_input = indices2validationsplit(
            validation_indices, data_input_split)
        train_labels, validation_labels = indices2validationsplit(
            validation_indices, data_labels)
    else:
        train_input, validation_input = mt.dataset_split(
            data_input_split, validation_split=validation_split)
        train_labels, validation_labels = mt.dataset_split(
            data_labels, validation_split=validation_split)

    # Data augmentation for training set
    if number_augmentations >= 1:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        train_input, train_labels = augment_dataset(
            train_input, train_labels, data_augmentation=data_augmentation, number_augmentations=number_augmentations)
    return train_input, validation_input, train_labels, validation_labels


def load_dataset_hise(dataset=None, resolution=None, validation_split=0.85, number_augmentations=0):
    '''Load and preprocess HiSE dataset
    resolution: Targeted image resolution
    number_augmentations: Number of image augmentations, rotated and flipped
    validation_split: Validation split -> e.g. CIFAR10: 5/6=0.833, MNIST: 6/7=0.857
    shuffle: Shuffle validation dataset
    '''
    # Select subdirectory where images are saved
    default_dataset = 'HiSE64'
    data_directory = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'Datasets', 'HiSE_dataset_generation', 'dataset_')
    if dataset is None or not os.path.isdir(data_directory + dataset):
        data_directory = data_directory + default_dataset
    else:
        data_directory = data_directory + dataset
    data_file = os.path.join(data_directory, 'annotation.txt')
    dataset_sheet = pd.read_csv(
        data_file, sep=",", header=None, engine='python')
    labels_list = dataset_sheet.loc[:, 2].to_numpy().astype('int')
    # NOTE: Possible to read where the box is in the image: [box,5,26,36,60]
    # First load with full resolution
    full_resolution = (dataset_sheet.loc[:, 3].to_numpy()[
                       0], dataset_sheet.loc[:, 4].to_numpy()[0])
    # Lower resolution for each image
    if resolution is None:
        image_resolution = full_resolution
    else:
        # width is defined by resolution
        image_length = int(
            full_resolution[0] / full_resolution[1] * resolution)
        image_resolution = (image_length, resolution)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        labels=labels_list.tolist(),
        label_mode='int',
        color_mode='rgb',
        # validation_split = 0.2,
        shuffle=False,
        image_size=image_resolution,
        crop_to_aspect_ratio=True,
    )
    # Convert into numpy format
    data_input = np.concatenate([x for x, _ in train_dataset], axis=0)
    data_labels = np.concatenate([y for _, y in train_dataset], axis=0)

    # Separating image tuples and combining images of same object into one list of images
    number_images = np.sum(dataset_sheet.loc[:, 1].to_numpy() == 0)
    train_input = [data_input[0::number_images, ...],
                   data_input[1::number_images, ...]]
    train_labels = data_labels[0::number_images, ...]

    # Split into training and validation data
    train_input, validation_input = mt.dataset_split(
        train_input, validation_split=validation_split)
    train_labels, validation_labels = mt.dataset_split(
        train_labels, validation_split=validation_split)

    # Data augmentation
    if number_augmentations >= 1:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        train_input, train_labels = augment_dataset(
            train_input, train_labels, data_augmentation=data_augmentation, number_augmentations=number_augmentations)
    return train_input, validation_input, train_labels, validation_labels


def resize_image(image, resize_length, resize_width, interpolation='bilinear'):
    '''Resize image to length resize_length and width resize_width
    interpolation: interpolation technique is default 'bilinear'
    '''
    image_resized = tf.keras.layers.Resizing(
        resize_length, resize_width, crop_to_aspect_ratio=True, interpolation=interpolation)(image).numpy()
    return image_resized


def indices2validationsplit(validation_indices, datasets):
    '''Customized validation split according to given validation indices
    validation_indices: Indices to be validation dataset
    data_input: Inputs of the dataset
    '''
    not_list = False
    if not isinstance(datasets, list):
        not_list = True
        datasets = [datasets]
    dataset_size = datasets[0].shape[0]
    full_dataset_indices = np.arange(dataset_size)
    training_indices = np.setdiff1d(full_dataset_indices, validation_indices)
    train_input = []
    validation_input = []
    for dataset in datasets:
        validation_input.append(
            dataset[validation_indices, ...])
        train_input.append(dataset[training_indices, ...])
    if not_list:
        validation_input = validation_input[0]
        train_input = train_input[0]
    return train_input, validation_input


def convert_dataset_to_npz(dataset_name):
    '''Load and convert dataset to npz file
    '''
    if dataset_name == 'hirise':
        train_input, validation_input, train_labels, validation_labels, test_input, test_labels = load_dataset_hirise(
            resolution=None, hirisecrater=False, number_augmentations=0)
        data_directory = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'Datasets', 'hirise-map-proj-v3_2')
    elif dataset_name == 'fraeser':
        train_input, validation_input, train_labels, validation_labels = load_dataset_tools(
            resolution=None, number_augmentations=0, validation_split=1, shuffle=False)
        test_input = []  # No test set for fraeser
        test_labels = []
        data_directory = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'Datasets', 'Fraeser_Aufnahmen_220824_Inkl_Verkippt')
    else:
        print('Dataset not available')

    # Save pixel values as integers
    for imageset_index, imageset in enumerate(train_input):
        train_input[imageset_index] = imageset.astype('int32')
    for imageset_index, imageset in enumerate(validation_input):
        validation_input[imageset_index] = imageset.astype('int32')
    for imageset_index, imageset in enumerate(test_input):
        test_input[imageset_index] = imageset.astype('int32')
    dataset = {}
    dataset['images'] = {}
    dataset['labels'] = {}
    dataset['images']['training'] = train_input
    dataset['images']['validation'] = validation_input
    dataset['images']['test'] = test_input
    dataset['labels']['training'] = train_labels
    dataset['labels']['validation'] = validation_labels
    dataset['labels']['test'] = test_labels
    np.savez_compressed(os.path.join(
        data_directory, 'dataset_' + dataset_name), dataset)


def load_dataset_hirise(resolution=None, hirisecrater=False, number_augmentations=0):
    '''Load and preprocess hirise dataset
    resolution: Loaded image resolution
    hirisecrate: Use only labeling of craters, two class problem
    number_augmentations: Number of image augmentations
    '''
    # Select subdirectory where images are saved
    data_directory = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'Datasets', 'hirise-map-proj-v3_2')
    data_file = os.path.join(data_directory, 'labels-map-proj_v3_2.txt')
    data_file2 = os.path.join(
        data_directory, 'labels-map-proj_v3_2_train_val_test.txt')
    dataset_sheet = pd.read_csv(data_file, sep=r"\s", header=None,
                                engine='python')  # dataset_sheet.loc[:, 0]
    dataset_sheet2 = pd.read_csv(
        data_file2, sep=r"\s", header=None, engine='python')
    # Add index of alphanumerically ordered data set to dataset_sheet2
    indr = np.argmax(dataset_sheet.sort_values(by=0, ascending=True).loc[:, 0].to_numpy(
    ) == dataset_sheet2.loc[:, 0].to_numpy()[:, None], axis=-1)
    dataset_sheet2[3] = indr.tolist()
    labels_list = dataset_sheet.sort_values(
        by=0, ascending=True).loc[:, 1].to_numpy()
    class_names = pd.read_csv(
        os.path.join(data_directory, 'landmarks_map-proj-v3_2_classmap.csv'), sep=',', header=None, engine='python')
    print(f'Class names: {class_names.loc[:, 1].to_dict()}')
    # Lower resolution for each image, always quadratic size
    full_resolution = (227, 227)
    if resolution is None:
        # Default is full image resolution
        image_resolution = full_resolution
    else:
        image_resolution = (resolution, resolution)
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        labels=labels_list.tolist(),
        label_mode='int',
        color_mode='grayscale',
        # validation_split = 0.2,
        # subset = "training", # "validation"
        shuffle=False,
        # seed = 123,
        image_size=image_resolution,
        crop_to_aspect_ratio=True,
    )
    # Convert into numpy format
    data_input = np.concatenate([x for x, _ in train_dataset], axis=0)
    data_labels = np.concatenate([y for _, y in train_dataset], axis=0)
    # Combine all classes except for craters in one class
    if hirisecrater:
        data_labels = (data_labels == 1) * 1
    # Training data set
    ind_set = dataset_sheet2.loc[dataset_sheet2.loc[:, 2]
                                 == 'train', 3].to_numpy()
    train_input = [data_input[ind_set, ...]]
    train_labels = [data_labels[ind_set, ...]]
    # Validation data set
    ind_set = dataset_sheet2.loc[dataset_sheet2.loc[:, 2]
                                 == 'val', 3].to_numpy()
    validation_input = [data_input[ind_set, ...]]
    validation_labels = [data_labels[ind_set, ...]]
    # Actual test set (small), not used so far...
    ind_set = dataset_sheet2.loc[dataset_sheet2.loc[:, 2]
                                 == 'test', 3].to_numpy()
    test_input = [data_input[ind_set, ...]]
    test_labels = [data_labels[ind_set, ...]]
    # Data augmentation
    if number_augmentations >= 1:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        train_input, train_labels = augment_dataset(
            train_input, train_labels, data_augmentation=data_augmentation, number_augmentations=number_augmentations)
    return train_input, validation_input, train_labels, validation_labels, test_input, test_labels

# How image resizing with crop aspect ratio works (smart_resize):
# From: https://github.com/keras-team/keras/blob/v2.14.0/keras/utils/image_utils.py
#
# Interpolation: bilinear (default)
#
# Your output images will actually be `(200, 200)`, and will not be distorted.
# Instead, the parts of the image that do not fit within the target size
# get cropped out.

# The resizing process is:

# 1. Take the largest centered crop of the image that has the same aspect
# ratio as the target size. For instance, if `size=(200, 200)` and the input
# image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
# the width.
# 2. Resize the cropped image to the target size. In the example above,
# we resize the `(340, 340)` crop to `(200, 200)`.


def augment_input_data(datasets, data_augmentation, number_augmentations=1):
    '''Augment list of datasets
    data_augmentation: Keras model with augmentation layers
    number_augmentations: Number of added augmentations
    '''
    if number_augmentations >= 1:
        for image_number, image in enumerate(datasets):
            datasets_augmented = [image]
            for _ in range(number_augmentations):
                datasets_augmented.append(data_augmentation(image).numpy())
            datasets[image_number] = np.concatenate(datasets_augmented, axis=0)
    return datasets


def augment_dataset(train_images, train_labels, data_augmentation=None, number_augmentations=1):
    '''Augment list of datasets
    data_augmentation: Keras model with augmentation layers
    number_augmentations: Number of added augmentations
    '''
    if data_augmentation is None:
        # Default data augmentation layers
        data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(
            "horizontal_and_vertical"), tf.keras.layers.RandomRotation(0.2),])
    # Data augmentation
    if number_augmentations >= 1:
        train_images = augment_input_data(train_images, data_augmentation,
                                          number_augmentations=number_augmentations)
        train_labels = np.repeat(train_labels[np.newaxis],
                                 number_augmentations + 1, axis=0).flatten()
    return train_images, train_labels


def preprocess_pixels(train_images, test_images):
    '''Preprocessing: Scale pixels between 0 and 1 for image list
    '''
    train_images_normalized = train_images
    # Preprocess list of images
    for image_number, train_image in enumerate(train_images):
        # Convert RGB values from integers to floats
        train_images_normalized[image_number] = preprocess_pixels_image(
            train_image)
    # Same steps for test/validation set
    test_images_normalized = test_images
    for image_number, test_image in enumerate(test_images):
        test_images_normalized[image_number] = preprocess_pixels_image(
            test_image)
    # Return normalized images
    return train_images_normalized, test_images_normalized


def preprocess_pixels_image(image):
    '''Preprocessing: Scale pixels between 0 and 1 for one image
    '''
    # Convert RGB values from integers to floats
    image_preprocessed = image.astype('float32')
    # Normalize to range 0-1 (avoiding saturation with typical NN activation functions)
    image_preprocessed = image_preprocessed / 255.0
    # Return normalized images
    return image_preprocessed


def summarize_dataset(train_inputs, train_labels, test_inputs, test_labels):
    '''Summarize loaded training dataset (train_input, train_labels)
    and validation/test dataset (test_input, test_labels)
    '''
    for train_listindex, train_input_listelement in enumerate(train_inputs):
        print(
            f'Train: X={train_input_listelement.shape}, y={train_labels.shape}')
        print(
            f'Test: X={test_inputs[train_listindex].shape}, y={test_labels.shape}')
        # Plot first few images
        for image_number in range(9):
            # Define subplot
            plt.subplot(330 + 1 + image_number)
            # Plot raw pixel data
            plt.imshow(
                train_input_listelement[image_number], cmap=plt.get_cmap('gray'))
        # Show the figure
        plt.show()


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():
    train_input, validation_input, train_labels, validation_labels = load_dataset_hise()
