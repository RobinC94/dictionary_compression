# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _obtain_input_shape

from my_utils.dictionary_convolution import DictConv2D

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              rate=4,
              index_dict=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    if num_col * num_row > 2:
        x = DictConv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        comp_rate=rate,
        dict_index=index_dict)(x)
    else:
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def ModifiedInceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                        rate=4,
                        index_dict=None):

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid', rate=rate, index_dict=index_dict)
    x = conv2d_bn(x, 32, 3, 3, padding='valid', rate=rate, index_dict=index_dict)
    x = conv2d_bn(x, 64, 3, 3, rate=rate, index_dict=index_dict)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid', rate=rate, index_dict=index_dict)
    x = conv2d_bn(x, 192, 3, 3, padding='valid', rate=rate, index_dict=index_dict)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)

    branch5x5 = conv2d_bn(x, 48, 1, 1, rate=rate, index_dict=index_dict)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, rate=rate, index_dict=index_dict)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, rate=rate, index_dict=index_dict)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)

    branch5x5 = conv2d_bn(x, 48, 1, 1, rate=rate, index_dict=index_dict)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, rate=rate, index_dict=index_dict)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, rate=rate, index_dict=index_dict)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)

    branch5x5 = conv2d_bn(x, 48, 1, 1, rate=rate, index_dict=index_dict)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, rate=rate, index_dict=index_dict)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, rate=rate, index_dict=index_dict)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid', rate=rate, index_dict=index_dict)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, rate=rate, index_dict=index_dict)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid', rate=rate, index_dict=index_dict)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)

    branch7x7 = conv2d_bn(x, 128, 1, 1, rate=rate, index_dict=index_dict)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, rate=rate, index_dict=index_dict)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, rate=rate, index_dict=index_dict)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, rate=rate, index_dict=index_dict)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, rate=rate, index_dict=index_dict)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)

        branch7x7 = conv2d_bn(x, 160, 1, 1, rate=rate, index_dict=index_dict)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, rate=rate, index_dict=index_dict)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, rate=rate, index_dict=index_dict)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, rate=rate, index_dict=index_dict)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, rate=rate, index_dict=index_dict)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, rate=rate, index_dict=index_dict)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, rate=rate, index_dict=index_dict)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, rate=rate, index_dict=index_dict)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, rate=rate, index_dict=index_dict)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)

    branch7x7 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, rate=rate, index_dict=index_dict)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, rate=rate, index_dict=index_dict)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, rate=rate, index_dict=index_dict)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, rate=rate, index_dict=index_dict)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, rate=rate, index_dict=index_dict)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid', rate=rate, index_dict=index_dict)

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, rate=rate, index_dict=index_dict)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, rate=rate, index_dict=index_dict)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, rate=rate, index_dict=index_dict)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid', rate=rate, index_dict=index_dict)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, rate=rate, index_dict=index_dict)

        branch3x3 = conv2d_bn(x, 384, 1, 1, rate=rate, index_dict=index_dict)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, rate=rate, index_dict=index_dict)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, rate=rate, index_dict=index_dict)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, rate=rate, index_dict=index_dict)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, rate=rate, index_dict=index_dict)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, rate=rate, index_dict=index_dict)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, rate=rate, index_dict=index_dict)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, rate=rate, index_dict=index_dict)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
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
    model = Model(inputs, x, name='inception_v3')

    return model


def conv2d_bn_lite(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
                   rate=4):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    if num_col * num_row > 2:
        import my_models
        filters = max(my_models.LEAST_ATOMS, int(filters/rate))
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3_lite(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                     rate=4):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn_lite(img_input, 32, 3, 3, strides=(2, 2), padding='valid', rate=rate)
    x = conv2d_bn_lite(x, 32, 3, 3, padding='valid', rate=rate)
    x = conv2d_bn_lite(x, 64, 3, 3, rate=rate)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn_lite(x, 80, 1, 1, padding='valid', rate=rate)
    x = conv2d_bn_lite(x, 192, 3, 3, padding='valid', rate=rate)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn_lite(x, 64, 1, 1, rate=rate)

    branch5x5 = conv2d_bn_lite(x, 48, 1, 1, rate=rate)
    branch5x5 = conv2d_bn_lite(branch5x5, 64, 5, 5, rate=rate)

    branch3x3dbl = conv2d_bn_lite(x, 64, 1, 1, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_lite(branch_pool, 32, 1, 1, rate=rate)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn_lite(x, 64, 1, 1, rate=rate)

    branch5x5 = conv2d_bn_lite(x, 48, 1, 1, rate=rate)
    branch5x5 = conv2d_bn_lite(branch5x5, 64, 5, 5, rate=rate)

    branch3x3dbl = conv2d_bn_lite(x, 64, 1, 1, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_lite(branch_pool, 64, 1, 1, rate=rate)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn_lite(x, 64, 1, 1, rate=rate)

    branch5x5 = conv2d_bn_lite(x, 48, 1, 1, rate=rate)
    branch5x5 = conv2d_bn_lite(branch5x5, 64, 5, 5, rate=rate)

    branch3x3dbl = conv2d_bn_lite(x, 64, 1, 1, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_lite(branch_pool, 64, 1, 1, rate=rate)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn_lite(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn_lite(x, 64, 1, 1, rate=rate)
    branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 96, 3, 3, rate=rate)
    branch3x3dbl = conv2d_bn_lite(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid', rate=rate)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)

    branch7x7 = conv2d_bn_lite(x, 128, 1, 1, rate=rate)
    branch7x7 = conv2d_bn_lite(branch7x7, 128, 1, 7, rate=rate)
    branch7x7 = conv2d_bn_lite(branch7x7, 192, 7, 1, rate=rate)

    branch7x7dbl = conv2d_bn_lite(x, 128, 1, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 128, 7, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 128, 1, 7, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 128, 7, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 1, 7, rate=rate)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_lite(branch_pool, 192, 1, 1, rate=rate)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)

        branch7x7 = conv2d_bn_lite(x, 160, 1, 1, rate=rate)
        branch7x7 = conv2d_bn_lite(branch7x7, 160, 1, 7, rate=rate)
        branch7x7 = conv2d_bn_lite(branch7x7, 192, 7, 1, rate=rate)

        branch7x7dbl = conv2d_bn_lite(x, 160, 1, 1, rate=rate)
        branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 160, 7, 1, rate=rate)
        branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 160, 1, 7, rate=rate)
        branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 160, 7, 1, rate=rate)
        branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 1, 7, rate=rate)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_lite(branch_pool, 192, 1, 1, rate=rate)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)

    branch7x7 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)
    branch7x7 = conv2d_bn_lite(branch7x7, 192, 1, 7, rate=rate)
    branch7x7 = conv2d_bn_lite(branch7x7, 192, 7, 1, rate=rate)

    branch7x7dbl = conv2d_bn_lite(x, 192, 1, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 7, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 1, 7, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 7, 1, rate=rate)
    branch7x7dbl = conv2d_bn_lite(branch7x7dbl, 192, 1, 7, rate=rate)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_lite(branch_pool, 192, 1, 1, rate=rate)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)
    branch3x3 = conv2d_bn_lite(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid', rate=rate)

    branch7x7x3 = conv2d_bn_lite(x, 192, 1, 1, rate=rate)
    branch7x7x3 = conv2d_bn_lite(branch7x7x3, 192, 1, 7, rate=rate)
    branch7x7x3 = conv2d_bn_lite(branch7x7x3, 192, 7, 1, rate=rate)
    branch7x7x3 = conv2d_bn_lite(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid', rate=rate)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn_lite(x, 320, 1, 1, rate=rate)

        branch3x3 = conv2d_bn_lite(x, 384, 1, 1, rate=rate)
        branch3x3_1 = conv2d_bn_lite(branch3x3, 384, 1, 3, rate=rate)
        branch3x3_2 = conv2d_bn_lite(branch3x3, 384, 3, 1, rate=rate)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn_lite(x, 448, 1, 1, rate=rate)
        branch3x3dbl = conv2d_bn_lite(branch3x3dbl, 384, 3, 3, rate=rate)
        branch3x3dbl_1 = conv2d_bn_lite(branch3x3dbl, 384, 1, 3, rate=rate)
        branch3x3dbl_2 = conv2d_bn_lite(branch3x3dbl, 384, 3, 1, rate=rate)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_lite(branch_pool, 192, 1, 1, rate=rate)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
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
    model = Model(inputs, x, name='inception_v3')
    return model


def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = InceptionV3_lite(rate=4)
