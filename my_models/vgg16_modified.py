# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Flatten, Dense, Input, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape

from my_utils.dictionary_convolution import DictConv2D

def index_generator(index_list):
    for index in index_list:
        yield index

def ModifiedVGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000,
                  rate=4,
                  index_list=None):

    index_gen = index_generator(index_list)

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = DictConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', comp_rate=rate, dict_index=index_gen.next())(img_input)
    x = DictConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', comp_rate=rate, dict_index=index_gen.next())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = DictConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', comp_rate=rate, dict_index=index_gen.next())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = DictConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', comp_rate=rate, dict_index=index_gen.next())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', comp_rate=rate, dict_index=index_gen.next())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', comp_rate=rate, dict_index=index_gen.next())(x)
    x = DictConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', comp_rate=rate, dict_index=index_gen.next())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')


    return model

def VGG16_lite(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000,
               rate=4):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    import my_models

    # Block 1
    x = Conv2D(max(my_models.LEAST_ATOMS, int(64/rate)), (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(64/rate)), (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(max(my_models.LEAST_ATOMS, int(128/rate)), (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(128/rate)), (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(max(my_models.LEAST_ATOMS, int(256/rate)), (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(256/rate)), (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(256/rate)), (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(max(my_models.LEAST_ATOMS, int(512/rate)), (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    return model
