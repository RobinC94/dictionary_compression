import os

import keras
import numpy as np

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from termcolor import cprint
from array import array

from my_models.resnet50_modified import ModifiedResNet50
from my_models.vgg16_modified import ModifiedVGG16
from my_models.inceptionv3_modified import ModifiedInceptionV3
from my_utils.dic_learn import comp_kernel
from my_utils.dictionary_convolution import DictConv2D

####################################
##config params
#filter_size = 3
least_atoms = 16

####################################
## public API

def modify_model(model, name='resnet50', rate=4, save=False):
    model_new = get_modified_model(name)

    conv_layers_list = get_conv_layers_list(model_new, name)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")
    modify_layer_num = len(conv_layers_list)
    avg_error = 0

    weighted_layers_list = get_weighted_layers_list(model_new, name)

    for l in conv_layers_list:

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
        comp_num = max(least_atoms, kernels_num/rate)
        #print(kernels)
        #print(np.shape(kernels))

        dic, index, a_list, b_list, e = comp_kernel(kernels, n_components=comp_num)

        for i in range(model.layers[l].filters):  ##kernel num
            a = a_list[i]
            b = b_list[i]
            weights_new[0][i] = a
            weights_new[1][i] = b
            i1 = index[i][0]
            i2 = index[i][1]
            x = np.array(dic[i1]).reshape((filter_size[0], filter_size[1], model.layers[l].input_shape[-1]))
            y = np.array(dic[i2]).reshape((filter_size[0], filter_size[1], model.layers[l].input_shape[-1]))
            if use_bias:
                weights_new[3][:, :, :, i] = x
                weights_new[4][:, :, :, i] = y
            else:
                weights_new[2][:, :, :, i] = x
                weights_new[3][:, :, :, i] = y
            weights[0][:, :, :, i] = a * x + b * y

        avg_error += e
        model_new.layers[l].set_weights(weights_new)
        model.layers[l].set_weights(weights)

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

def get_modified_model(name='resnet50'):
    if name == 'resnet50':
        return ModifiedResNet50()
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
    model_new = modify_model(model, name='resnet50', rate=6)

    evaluate_model(model_new,name='resnet50', image_size=224)
    evaluate_model(model, name='resnet50', image_size=224)
