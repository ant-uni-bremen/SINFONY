#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 14:58:13 2024

@author: beck
ResNet for image recognition

Belongs to simulation framework for numerical results of the articles:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347 (First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1, Stockholm, Sweden, May 2024.
"""

# Tensorflow 2 packages
import tensorflow as tf


# ResNet for image recognition
# NOTE: Different results with tf.function decorator -> not necessary
#       since the layers/models are just defined here, it is not a function


def repeat_entry_2_list(number_units, number_blocks):
    '''Repeat number_blocks times an entry number_units, if number_units is not already a list
    '''
    if isinstance(number_units, list) and len(number_units) != number_blocks:
        number_units = number_units[0]
    if not isinstance(number_units, list):
        number_units = [number_units] * number_blocks
    return number_units


def calculate_resnet_layer_number(number_resnet_blocks, number_residual_units, bottleneck=False):
    '''Calculate official ResNet layer number
    Pooling layers not counted / only conv2d + dense softmax
    Bottleneck structure for deeper architectures -> 3 instead of 2
    '''
    if bottleneck is True:
        layer_number_per_residual_unit = 3
    else:
        layer_number_per_residual_unit = 2
    layer_number_at_input_and_output = 2
    number_residual_units = repeat_entry_2_list(
        number_residual_units, number_resnet_blocks)
    resnet_layer_number = layer_number_per_residual_unit * \
        sum(number_residual_units) + layer_number_at_input_and_output
    return resnet_layer_number


class ResnetConfiguration():
    '''ResNet configuration class with CIFAR10 default values
    Configuration of ResNet for CIFAR10 with [6 * number_residual_units + 2] layers
    image_shape: of input image with x/y dimension and channel as last dimension
    number_classes: number of image classes
    number_resnet_blocks: is fixed to 3 for CIFAR10
    number_residual_units: with 3 we arrive at ResNet20
    '''

    def __init__(self, architecture='CIFAR10', image_shape=(32, 32, 3), number_classes=10, number_filters=16, number_residual_units=3, number_resnet_blocks=3, preactivation=True, bottleneck=False, batch_normalization=True, weight_initialization='he_uniform', weight_decay=tf.keras.regularizers.l2(0.0001)):
        self.architecture = architecture
        self.image_shape = image_shape
        self.number_classes = number_classes
        self.number_filters = number_filters
        self.number_residual_units = number_residual_units
        self.number_resnet_blocks = number_resnet_blocks
        self.preactivation = preactivation
        self.bottleneck = bottleneck
        self.batch_normalization = batch_normalization
        self.weight_initialization = weight_initialization
        self.weight_decay = weight_decay


class ResnetConfigurationImageNet(ResnetConfiguration):
    '''ResNet configuration class with ImageNet default values
    Function returns architecture of ResNet18/34/50/101/152 for ImageNet dataset
    See Table 1 in "Deep Residual Learning for Image Recognition" for ImageNet architectures
    number_residual_units and bottleneck vary according to Table 1
    ResNet18: [2, 2, 2, 2], ResNet34: [3, 4, 6, 3], (bottleneck) ResNet50: [3, 4, 6, 3], ResNet101: [3, 4, 23, 3], ResNet152: [3, 8, 36, 3]
    ------------------
    number_resnet_blocks: is fixed to 4 (compared to 3 for CIFAR10)
    '''

    def __init__(self, *args, architecture='ImageNet', image_shape=(224, 224, 3), number_classes=1000, number_filters=64, number_residual_units=2, number_resnet_blocks=4, preactivation=True, bottleneck=False, **kwargs):
        super().__init__(*args, architecture=architecture, image_shape=image_shape, number_classes=number_classes, number_filters=number_filters, number_residual_units=number_residual_units, number_resnet_blocks=number_resnet_blocks,
                         preactivation=preactivation, bottleneck=bottleneck, **kwargs)


class Residual(tf.keras.Model):
    """The Residual block of ResNet for ResNet-18/34 and ResNet-CIFAR10.
    For ReLU activations weight initialization with he_uniform is better than glorot
    Preactivation version for better training
    """

    def __init__(self, number_channels, use_1x1conv=False, strides=1, kernel_initializer=None, kernel_regularizer=None, preactivation=True, batch_normalization=True):
        super().__init__()
        self.preactivation = preactivation
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_initializer2 = tf.keras.initializers.get(
            kernel_initializer)
        self.kernel_initializer3 = tf.keras.initializers.get(
            kernel_initializer)
        self.conv1 = tf.keras.layers.Conv2D(number_channels, padding='same', kernel_size=3, strides=strides,
                                            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(number_channels, padding='same', kernel_size=3,
                                            kernel_initializer=self.kernel_initializer2, kernel_regularizer=self.kernel_regularizer)
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(number_channels, kernel_size=1, strides=strides,
                                                kernel_initializer=self.kernel_initializer3, kernel_regularizer=self.kernel_regularizer)
        else:
            self.conv3 = None
        if batch_normalization:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
        else:
            self.bn1 = tf.keras.layers.Lambda(lambda x: x)
            self.bn2 = tf.keras.layers.Lambda(lambda x: x)

    def call(self, input_tensor):
        '''Run Residual block
        '''
        if self.preactivation is True:
            # New architecture: pre-activation
            output_tensor = self.conv1(
                tf.keras.activations.relu(self.bn1(input_tensor)))
            output_tensor = self.conv2(
                tf.keras.activations.relu(self.bn2(output_tensor)))
            if self.conv3 is not None:
                input_tensor = self.conv3(input_tensor)
            output_tensor = output_tensor + input_tensor
        else:
            # Original architecture
            output_tensor = tf.keras.activations.relu(
                self.bn1(self.conv1(input_tensor)))
            output_tensor = self.bn2(self.conv2(output_tensor))
            if self.conv3 is not None:
                input_tensor = self.conv3(input_tensor)
            output_tensor = output_tensor + input_tensor
            output_tensor = tf.keras.activations.relu(output_tensor)
        return output_tensor


class ResidualBottleneck(tf.keras.Model):
    """The Residual block of ResNet in bottleneck version for ResNet-50/101/152.
    For ReLU activations weight initialization with he_uniform is better than glorot
    Preactivation version for better training
    (Identity shortcuts are essential for efficient training with less parameters, but not used in preactivation paper...)
    """

    def __init__(self, number_channels, use_1x1conv=False, strides=1, kernel_initializer=None, kernel_regularizer=None, preactivation=True, batch_normalization=True):
        super().__init__()
        self.preactivation = preactivation
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_initializer2 = tf.keras.initializers.get(
            kernel_initializer)
        self.kernel_initializer3 = tf.keras.initializers.get(
            kernel_initializer)
        self.kernel_initializer4 = tf.keras.initializers.get(
            kernel_initializer)
        self.conv1 = tf.keras.layers.Conv2D(number_channels, padding='same', kernel_size=1, strides=strides,
                                            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(number_channels, padding='same', kernel_size=3,
                                            kernel_initializer=self.kernel_initializer2, kernel_regularizer=self.kernel_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(4 * number_channels, padding='same', kernel_size=1,
                                            kernel_initializer=self.kernel_initializer3, kernel_regularizer=self.kernel_regularizer)
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(4 * number_channels, kernel_size=1, strides=strides,
                                                kernel_initializer=self.kernel_initializer4, kernel_regularizer=self.kernel_regularizer)
        else:
            self.conv4 = None
        if batch_normalization:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        else:
            self.bn1 = tf.keras.layers.Lambda(lambda x: x)
            self.bn2 = tf.keras.layers.Lambda(lambda x: x)
            self.bn3 = tf.keras.layers.Lambda(lambda x: x)

    def call(self, input_tensor):
        '''Run Residual block with bottleneck
        '''
        if self.preactivation is True:
            # New architecture: pre-activation
            output_tensor = self.conv1(
                tf.keras.activations.relu(self.bn1(input_tensor)))
            output_tensor = self.conv2(
                tf.keras.activations.relu(self.bn2(output_tensor)))
            output_tensor = self.conv3(
                tf.keras.activations.relu(self.bn3(output_tensor)))
            if self.conv4 is not None:
                input_tensor = self.conv4(input_tensor)
            output_tensor = output_tensor + input_tensor
        else:
            # Original architecture
            output_tensor = tf.keras.activations.relu(
                self.bn1(self.conv1(input_tensor)))
            output_tensor = tf.keras.activations.relu(
                self.bn2(self.conv2(output_tensor)))
            output_tensor = self.bn3(self.conv3(output_tensor))
            if self.conv4 is not None:
                input_tensor = self.conv4(input_tensor)
            output_tensor = output_tensor + input_tensor
            output_tensor = tf.keras.activations.relu(output_tensor)
        return output_tensor


class ResnetBlock(tf.keras.Model):  # tf.keras.layers.Layer
    '''ResNet block
    '''

    def __init__(self, number_channels, number_residuals, first_block=False, kernel_initializer=None, kernel_regularizer=None, preactivation=True, bottleneck=False, batch_normalization=True,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(number_residuals):
            if bottleneck:
                if i == 0:
                    self.residual_layers.append(ResidualBottleneck(number_channels, use_1x1conv=True, strides=2,
                                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, preactivation=preactivation, batch_normalization=batch_normalization))
                else:
                    self.residual_layers.append(ResidualBottleneck(
                        number_channels, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, preactivation=preactivation, batch_normalization=batch_normalization))
            else:
                if i == 0 and not first_block:
                    self.residual_layers.append(Residual(number_channels, use_1x1conv=True, strides=2,
                                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, preactivation=preactivation, batch_normalization=batch_normalization))
                else:
                    self.residual_layers.append(Residual(
                        number_channels, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, preactivation=preactivation, batch_normalization=batch_normalization))

    def call(self, input_tensor):
        '''Execute ResNet block
        '''
        for layer in self.residual_layers:  # TODO: Test for self.residual_layers instead of self.residual_layers.layers
            input_tensor = layer(input_tensor)
        return input_tensor


def list2first_element(input_list):
    '''Converts list with one element to this element
    '''
    if isinstance(input_list, list):
        first_element = input_list[0]
    else:
        first_element = input_list
    if not isinstance(input_list, int) and not isinstance(input_list, tuple) and len(input_list) > 1:
        print('Warning: More than one element in provided list!')
    return first_element


def resnet_feature_extractor(resnet_config=ResnetConfiguration()):
    '''Function returns ResNet feature extractor for both ImageNet and CIFAR10 datasets
    image_shape: of input image with x/y dimension and channel as last dimension
    number_resnet_blocks: is fixed to 3 for CIFAR10
    number_residual_units: with 3 we arrive at ResNet20
    weight_init: default is he_uniform, he_normal in ResNet paper
    '''
    # Input handling
    image_shape = list2first_element(resnet_config.image_shape)
    number_filters = list2first_element(resnet_config.number_filters)
    # number_residual_units is list by default -> conversion
    number_residual_units = repeat_entry_2_list(
        resnet_config.number_residual_units, resnet_config.number_resnet_blocks)

    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(image_shape)
    if resnet_config.architecture.lower() == 'cifar10':
        # CIFAR10 dataset
        x_tensor = tf.keras.layers.Conv2D(number_filters, kernel_size=3, strides=1, padding='same',
                                          kernel_initializer=resnet_config.weight_initialization, kernel_regularizer=resnet_config.weight_decay)(x_input)
    elif resnet_config.architecture.lower() == 'imagenet':
        # ImageNet dataset
        x_tensor = tf.keras.layers.Conv2D(number_filters, kernel_size=7, strides=2, padding='same',
                                          kernel_initializer=resnet_config.weight_initialization, kernel_regularizer=resnet_config.weight_decay)(x_input)
    else:
        print('Architecture not implemented!')
    if resnet_config.preactivation is False:
        # In original preactivation implementation no activation is used
        # But the paper says this: "For the first Residual Unit (that follows a stand-alone convolutional layer, conv1),
        # we adopt the first activation right after conv1 and before splitting into two paths"
        # No MaxPooling is mentioned
        if resnet_config.batch_normalization:
            x_tensor = tf.keras.layers.BatchNormalization()(x_tensor)
        x_tensor = tf.keras.layers.Activation('relu')(x_tensor)
    # MaxPooling Layer in preactivation version???
    if resnet_config.architecture.lower() != 'cifar10':
        # MaxPooling Layer not in CIFAR10 version
        x_tensor = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding='same')(x_tensor)
    # Step 2 (ResNet Layers)
    for index_block in range(0, resnet_config.number_resnet_blocks):
        if index_block == 0:
            x_tensor = ResnetBlock(2 ** index_block * number_filters, number_residual_units[index_block], first_block=True, kernel_initializer=resnet_config.weight_initialization,
                                   kernel_regularizer=resnet_config.weight_decay, preactivation=resnet_config.preactivation, bottleneck=resnet_config.bottleneck, batch_normalization=resnet_config.batch_normalization)(x_tensor)
        else:
            x_tensor = ResnetBlock(2 ** index_block * number_filters, number_residual_units[index_block], kernel_initializer=resnet_config.weight_initialization,
                                   kernel_regularizer=resnet_config.weight_decay, preactivation=resnet_config.preactivation, bottleneck=resnet_config.bottleneck, batch_normalization=resnet_config.batch_normalization)(x_tensor)
    # Step 3 (Final Layers)
    if resnet_config.preactivation is True:
        if resnet_config.batch_normalization:
            x_tensor = tf.keras.layers.BatchNormalization()(x_tensor)
        x_tensor = tf.keras.layers.Activation('relu')(x_tensor)
    x_tensor = tf.keras.layers.GlobalAvgPool2D()(x_tensor)
    model = tf.keras.models.Model(
        inputs=x_input, outputs=x_tensor)  # , name='ResNetFeatureExtractor'
    return model


def resnet(resnet_config=ResnetConfiguration()):
    '''Function returns architecture of ResNet for general dataset
    ------------------
    architecture: Choose architecture tailored to specific dataset
    image_shape: of input image with x/y dimension and channel as last dimension
    number_classes: number of image classes
    '''
    # Input handling
    image_shape = list2first_element(resnet_config.image_shape)
    # Model definition
    x_input = tf.keras.layers.Input(image_shape)
    feature_extractor = resnet_feature_extractor(resnet_config=resnet_config)
    x_tensor = feature_extractor(x_input)
    # TODO: Also regularize last softmax layer?
    x_tensor = tf.keras.layers.Dense(
        units=resnet_config.number_classes, activation='softmax')(x_tensor)
    resnet_layer_number = calculate_resnet_layer_number(
        resnet_config.number_resnet_blocks, resnet_config.number_residual_units, resnet_config.bottleneck)
    model = tf.keras.models.Model(
        inputs=x_input, outputs=x_tensor, name=f'ResNet{resnet_layer_number}_{resnet_config.architecture.upper()}')
    return model
