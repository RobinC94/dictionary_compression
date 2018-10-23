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

####################################
##config params
#filter_size = 3
least_atoms = 16

####################################
## public API

def modify_model(model, name='resnet50', rate=4, save=False):
    conv_layers_list = get_conv_layers_list(model, name)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    weighted_layers_list = get_weighted_layers_list(model, name)

    model_new = get_modified_model(name)

    for l in conv_layers_list:

        # original weights 0: conv, 1: bias
        weights = model.layers[l].get_weights()
        # new weights 0: A, 1: B, 2: bias, 3: X, 4: Y
        weights_new = model_new.layers[l].get_weights()

        use_bias = model.layers[l].use_bias
        if use_bias:
            weights_new[2] = weights[1]

        # kernels HWCN 3*3*c*n
        kernels = weights[0]
        kernels_num = np.shape(kernels)[-1]
        filter_size = np.shape(kernels)[:2]
        comp_num = max(least_atoms, kernels_num/rate)
        #print(kernels)
        #print(np.shape(kernels))

        dic_list, index_list, a_list, b_list = comp_kernel(kernels, n_components=comp_num)
        #print(np.shape(dic_list))
        #print(np.shape(index_list))
        #print(np.shape(a_list))
        #print(np.shape(b_list))

        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                i1 = index_list[s][i][0]
                i2 = index_list[s][i][1]
                x = np.array(dic_list[s][i1]).reshape(filter_size)
                y = np.array(dic_list[s][i2]).reshape(filter_size)
                a = a_list[s][i]
                b = b_list[s][i]
                weights_new[0][s, i] = a
                weights_new[1][s, i] = b
                if use_bias:
                    weights_new[3][:, :, s, i] = x
                    weights_new[4][:, :, s, i] = y
                else:
                    weights_new[2][:, :, s, i] = x
                    weights_new[3][:, :, s, i] = y
                weights[0][:,:,s,i] = a*x+b*y

        model_new.layers[l].set_weights(weights_new)
        model.layers[l].set_weights(weights)

    for l in weighted_layers_list:
        weights = model.layers[l].get_weights()
        model_new.layers[l].set_weights(weights)

    if save:
        model_new.save_weights("/home/crb/PycharmProjects/keras/dictionary/weights/%s_modified_weights_%d.h5" % (name, rate))

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
        if name == "resnet50" or name == "vgg16":
            if isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] == [3, 3]:
                res+= [i]
        elif name == "inceptionv3":
            if isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2][0] * l.kernel.shape.as_list()[:2][1] > 4:
                res+= [i]
    return res

def get_weighted_layers_list(model, name):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if name == "resnet50" or name == 'vgg16':
            if (isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] != [3, 3]) \
                    or isinstance(l, BatchNormalization) or isinstance(l, Dense):
                res += [i]
        elif name == 'inceptionv3':
            if (isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2][0] * l.kernel.shape.as_list()[:2][1] <= 4) \
                    or isinstance(l, BatchNormalization) or isinstance(l, Dense):
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = keras.applications.ResNet50()
    model.load_weights("../weights/resnet50_weights_75_16.h5")
    model_new = modify_model(model, name='resnet50', rate=8)

    evaluate_model(model_new,name='resnet50', image_size=224)
    evaluate_model(model, name='resnet50', image_size=224)
