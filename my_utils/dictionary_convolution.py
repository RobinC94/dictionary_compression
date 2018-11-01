import sys,os

from keras import backend as K
from keras import initializers,regularizers,constraints,activations
from keras.applications import resnet50

from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.regularizers import l2

import tensorflow as tf
import numpy as np

import my_models

class DictConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 comp_rate=None,
                 dict_index = None,
                 **kwargs
                 ):
        super(DictConv2D, self).__init__(**kwargs)
        self.rank=2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size,2,'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides,2,'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format=conv_utils.normalize_data_format(data_format)
        self.dilation_rate=conv_utils.normalize_tuple(dilation_rate,2,'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.comp_rate = comp_rate
        self.dict_index = dict_index

    def build(self, input_shape):
        if self.data_format=='channels_first':
            channel_axis=1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.input_dim = input_shape[channel_axis]

        self.kernels_shape = self.kernel_size + (self.input_dim, self.filters)

        self.dict_num = max(my_models.LEAST_ATOMS, self.filters/self.comp_rate)

        self.dict_shape = self.kernel_size + (self.input_dim, self.dict_num,)

        self.dict = self.add_weight(shape=(self.dict_shape),
                                    initializer=self.kernel_initializer,
                                    name='dict',
                                    trainable=True)

        self.coef=self.add_weight(shape=(self.filters*2,),
                               initializer=self.kernel_initializer,
                               name='coef',
                               trainable=True)

        if isinstance(self.dict_index, dict):
            self.dict_index = self.dict_index[self.name]

        self.index=np.array(self.dict_index, dtype='int32')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        '''
        self.matrix=tf.SparseTensor(indices=tf.to_int64(self.index),
                                    values=self.coef,
                                    dense_shape=(self.dict_num, self.filters))

        #self.kernel = tf.sparse_tensor_dense_matmul(self.matrix, self.dict)
        '''
        #self.index.eval(K.get_session())
        #print(self.index)
        '''
        self.matrix = tf.transpose(tf.sparse_to_dense(sparse_indices=self.index,
                                         sparse_values=self.coef,
                                         output_shape=(self.filters, self.dict_num)))
        '''
        self._input_shape = inputs.get_shape().as_list()

        self.matrix = tf.SparseTensor(indices=self.index,
                                      values=self.coef,
                                      dense_shape=(self.filters, self.dict_num))


        self.kernel = self._matmul(self.dict)

        
        outputs=K.conv2d(inputs,
                         self.kernel,
                         strides=self.strides,
                         padding=self.padding,
                         data_format=self.data_format,
                         dilation_rate=self.dilation_rate)
        '''
        outputs = K.conv2d(inputs,
                           self.dict,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        outputs = self._matmul(outputs)
        '''


        if self.use_bias:
            outputs=K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)


    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config=super(DictConv2D, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def _matmul(self, tensor):
        sh = tensor.get_shape().as_list()
        #prod = 64
        #for z in sh[1:-1]:
        #    prod *= z
        reshape_tensor = tf.reshape(tensor, (-1, sh[-1]))
        result = tf.reshape(tf.transpose(tf.sparse_tensor_dense_matmul(self.matrix, tf.transpose(reshape_tensor))),
                            [-1,]+sh[1:-1]+[self.filters,])
        return result

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _IMAGE_DATA_FORMAT='channels_last'

    input_shape=(32,32,8)
    kernel = np.array(range(3 * 3 * 8 * 32)).reshape((3, 3, 8, 32))
    from my_utils import comp_kernel
    dic, index, a_list, b_list, e = comp_kernel(kernel, n_components=10)

    layer1=DictConv2D(filters=32,
                      kernel_size=(3,3),
                        strides=(2,2),
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4),
                          padding="same",
                          data_format=_IMAGE_DATA_FORMAT,
                      use_bias=False,
                      comp_rate=4,
                      dict_index=index)
    layer1.build(input_shape)
    #print (layer1.compute_output_shape(input_shape))
    print (layer1.get_config())
    print ("########")
    weights = layer1.get_weights()
    print(np.array(weights[0]).shape)
    print(np.array(weights[1]).shape)
    #print(np.array(weights[2]).shape)

    for i in range(layer1.dict_num):
        x = dic[i]
        weights[0][:, :, :, i] = np.array(x).reshape((3,3,8))

    for i in range(layer1.filters):  ##kernel num
        a = a_list[i]
        b = b_list[i]
        weights[1][i * 2] = a
        weights[1][i * 2 + 1] = b

    layer1.set_weights(weights)

    inin = tf.constant(value=0, shape=(2,32,32,8), dtype="float32")
    res=layer1.call(inin)
    print(res)
