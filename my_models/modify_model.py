import os

import keras
import numpy as np
import tensorflow as tf
import pickle

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from termcolor import cprint
from array import array

from my_models import ModifiedResNet50,  ModifiedVGG16, ModifiedInceptionV3
from my_models.resnet20_modified import ModifiedResnet20, Resnet20
from my_utils import comp_kernel, DictConv2D

import my_models

####################################
##config params
#filter_size = 3

####################################
## public API

def modify_model(model, name='resnet50', rate=4, save=False):

    conv_layers_list = get_conv_layers_list(model, name)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")
    modify_layer_num = len(conv_layers_list)
    avg_error = 0

    weighted_layers_list = get_weighted_layers_list(model, name)
    cprint("weighted layers is:" + str(weighted_layers_list), "yellow")

    dict_list = []
    index_list = []
    x_list_list = []
    index_dict = {}

    for c in range(len(conv_layers_list)):
        l = conv_layers_list[c]
        print(model.layers[l].name)
        # original weights 0: conv, 1: bias
        weights = model.layers[l].get_weights()

        # kernels HWCN 3*3*c*n
        kernels = np.array(weights[0])
        kernels_num = np.shape(kernels)[-1]
        comp_num = max(my_models.LEAST_ATOMS, kernels_num/rate)

        #print(kernels)
        #print(np.shape(kernels))
        nor = 0
        for i in range(kernels_num):
            k = kernels[:, :, :, i]
            nor += np.linalg.norm(k) / kernels_num

        dic, index, x_list, e = comp_kernel(kernels, n_components=comp_num)
        #print(index)
        dic *= nor
        x_list = [i/nor for i in x_list]

        dict_list.append(dic)
        index_list.append(index)
        x_list_list.append(x_list)
        index_dict[l] = index

        avg_error += e

    if save:
        pk_file_name = "/home/crb/PycharmProjects/keras/dictionary/tmp/%s_index_list_%d.pk" % (name, rate)
        with open(pk_file_name, 'w') as f:
            if name == 'inceptionv3':
                pickle.dump(index_dict, f)
            else:
                pickle.dump(index_list, f)

    model_new = get_modified_model(name, rate=rate, index_list=index_list, index_dict=index_dict)

    for c in range(len(conv_layers_list)):
        l = conv_layers_list[c]
        weights = model.layers[l].get_weights()
        weights_new = model_new.layers[l].get_weights()

        use_bias = model.layers[l].use_bias
        if use_bias:
            weights_new[2] = weights[1]

        kernels = np.array(weights[0])
        filter_size = np.shape(kernels)[:2]

        dic = dict_list[c]
        x_list = x_list_list[c]

        for i in range(model_new.layers[l].dict_num):
            d = dic[i]
            weights_new[0][:,:,:,i] = np.array(d).reshape(filter_size + (model.layers[l].input_shape[-1],))

        weights_new[1] = np.array(x_list)

        model_new.layers[l].set_weights(weights_new)

    for l in weighted_layers_list:
        weights = model.layers[l].get_weights()
        model_new.layers[l].set_weights(weights)

    if save:
        model_new.save_weights("/home/crb/PycharmProjects/keras/dictionary/weights/%s_modified_weights_%d.h5" % (name, rate))

    print("total avg error: ", avg_error/modify_layer_num)
    return model_new

def load_modified_model(name='resnet50', include_top=True, rate=4, weights=None):
    pk_file_name = "/home/crb/PycharmProjects/keras/dictionary/tmp/%s_index_list_%d.pk" % (name, rate)
    with open(pk_file_name, 'r') as f:
        index_list = pickle.load(f)

    if name == 'inceptionv3':
        model_new = get_modified_model(name, include_top=include_top, rate=rate, index_dict=index_list)
    else:
        model_new = get_modified_model(name, include_top=include_top, rate=rate, index_list=index_list)

    if weights is not None:
        model_new.load_weights(weights)
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
        #if name == 'resnet50' or name == 'vgg16':
            if isinstance(l, Conv2D) and l.kernel.shape.as_list()[0]*l.kernel.shape.as_list()[1] >= 3:
                res+= [i]
    return res

def get_weighted_layers_list(model, name):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        #if name == 'resnet50' or name == 'vgg16':
            if (isinstance(l, Conv2D) and l.kernel.shape.as_list()[0]*l.kernel.shape.as_list()[1] < 3) \
                    or isinstance(l, BatchNormalization) or isinstance(l, Dense):
                res += [i]
    return res

def get_modified_model(name='resnet50', include_top=True, rate=4, index_list=None, index_dict=None):
    if name == 'resnet50':
        return ModifiedResNet50(include_top=include_top, input_shape=(224,224,3), rate=rate, index_list=index_list)
    elif name == 'vgg16':
        return ModifiedVGG16(include_top=include_top, input_shape=(224,224,3), rate=rate, index_list=index_list)
    elif name == 'resnet20':
        return ModifiedResnet20(rate=rate, index_list=index_list)
    elif name == 'inceptionv3':
        model_tmp = ModifiedInceptionV3(include_top=include_top, input_shape=(299,299,3), rate=rate, index_dict=None)
        dict_res = {}
        layers = model_tmp.layers
        for i, l in enumerate(layers):
            if (isinstance(l, DictConv2D)):
                #print(i, l.name)
                name_id = int(l.name[12:])
                new_name = l.name[:12] + str(name_id + 54)
                dict_res[new_name] = index_dict[i]
        del model_tmp
        return ModifiedInceptionV3(include_top=include_top, input_shape=(299,299,3), rate=rate, index_dict=dict_res)
    else:
        raise ValueError("model name wrong")



#####################################
## for debug
if __name__ == "__main__":
    from my_train_and_eval import evaluate_model
    from cifar10_train_and_eval import model_test

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    #model = keras.applications.ResNet50()
    #model.load_weights("../weights/resnet50_weights_75_16.h5")
    #model = keras.applications.VGG16()
    #model.load_weights("../weights/vgg16_weights_71_42.h5")
    #model = keras.applications.InceptionV3()
    model = Resnet20()
    model.load_weights("../weights/resnet20_cifar10_weights.183.h5")
    name='resnet20'
    rate=50

    model_new = modify_model(model, name=name, rate=rate)
    model_test(model)
    model_test(model_new)


    
    #'''











