from __future__ import print_function
from __future__ import absolute_import

import os

from keras.layers import Input
from keras import layers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,  \
                            GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape

from my_utils import DictConv2D


def identity_block(input_tensor, kernel_size, filters, stage, block, rate=4):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = DictConv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b', comp_rate=rate)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), rate=4):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = DictConv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', comp_rate=rate)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ModifiedResNet50(include_top=True,
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000,
                     rate=4):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      weights='imagenet',
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), rate=rate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', rate=rate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', rate=rate)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', rate=rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', rate=rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', rate=rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', rate=rate)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', rate=rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', rate=rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', rate=rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', rate=rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', rate=rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', rate=rate)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', rate=rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', rate=rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', rate=rate)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
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
    model = Model(inputs, x, name='resnet50')


    return model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ModifiedResNet50(rate=4)
    model.summary()
