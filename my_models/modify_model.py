import os

import keras
import numpy as np
import tensorflow as tf

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from termcolor import cprint
from array import array

from my_models import ModifiedResNet50,  ModifiedVGG16, ModifiedInceptionV3
from my_utils import comp_kernel, DictConv2D

import my_models

####################################
##config params
#filter_size = 3

####################################
## public API

def modify_model(model, name='resnet50', rate=4, save=False):
    model_new = get_modified_model(name, rate=rate)

    conv_layers_list = get_conv_layers_list(model_new, name)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")
    modify_layer_num = len(conv_layers_list)
    avg_error = 0

    weighted_layers_list = get_weighted_layers_list(model_new, name)

    for l in conv_layers_list:
        print(model.layers[l].name)

        # original weights 0: conv, 1: bias
        weights = model.layers[l].get_weights()
        # new weights 0: A, 1: B, 2: bias, 3: X, 4: Y
        weights_new = model_new.layers[l].get_weights()

        use_bias = model.layers[l].use_bias
        if use_bias:
            weights_new[2] = weights[1]

        # kernels HWCN 3*3*c*n
        kernels = np.array(weights[0])
        kernels_num = np.shape(kernels)[-1]
        filter_size = np.shape(kernels)[:2]
        comp_num = max(my_models.LEAST_ATOMS, kernels_num/rate)

        #print(kernels)
        #print(np.shape(kernels))

        dic, index, a_list, b_list, e = comp_kernel(kernels, n_components=comp_num)
        #print(index)

        model.layers[l].index = tf.constant(value=index, dtype='int32')

        for i in range(model_new.layers[l].dict_num):
            x = dic[i]
            weights_new[0][:,:,:,i] = np.array(x).reshape(filter_size + (model.layers[l].input_shape[-1],))

        for i in range(model.layers[l].filters):  ##kernel num
            a = a_list[i]
            b = b_list[i]
            ind1 = index[i*2]
            ind2 = index[i*2+1]
            weights_new[1][i*2] = a
            weights_new[1][i*2+1] = b
            weights_new[3][i*2,:] = ind1
            weights_new[3][i*2+1,:] = ind2

        avg_error += e
        model_new.layers[l].set_weights(weights_new)

    for l in weighted_layers_list:
        weights = model.layers[l].get_weights()
        model_new.layers[l].set_weights(weights)

    if save:
        model_new.save_weights("/home/crb/PycharmProjects/keras/dictionary/weights/%s_modified_weights_%d.h5" % (name, rate))

    print("total avg error: ", avg_error/modify_layer_num)
    return model_new

########################################
## private API
def get_conv_layers_list(model, name):
    '''
        only  choose layers which is conv layer, and its filter_size must be same as param "filter_size"
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if isinstance(l, DictConv2D):
            res+= [i]
    return res

def get_weighted_layers_list(model, name):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if isinstance(l, Conv2D) or isinstance(l, BatchNormalization) or isinstance(l, Dense):
            res += [i]
    return res

def get_modified_model(name='resnet50', rate=4):
    if name == 'resnet50':
        return ModifiedResNet50(rate=rate)
    elif name == 'vgg16':
        return ModifiedVGG16()
    elif name == 'inceptionv3':
        return ModifiedInceptionV3()
    else:
        raise ValueError("model name wrong")



#####################################
## for debug
if __name__ == "__main__":
    from my_train_and_eval import evaluate_model

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = keras.applications.ResNet50()
    model.load_weights("../weights/resnet50_weights_75_16.h5")
    model_new = modify_model(model, name='resnet50', rate=4)

    evaluate_model(model_new,name='resnet50', image_size=224)
