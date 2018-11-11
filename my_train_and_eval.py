# !/usr/bin/python
import sys, os
from termcolor import cprint
from random import sample
import itertools
import threading
import keras
import time

from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
#from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, \
    LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy

import numpy as np
import json
from math import sqrt
import xml.etree.ElementTree

##configuration parameters
img_size = 224
img_parent_dir = "/home/crb/datasets/imageNet/ILSVRC2012/"  # sub_dir: train, val, test

nb_epoch = 20
batch_size = 64
evaluating_batch_size = 96

##used in 3rd-party model function
class_parse_file = "/home/crb/.keras/models/imagenet_class_index.json"
imagenet_utils.CLASS_INDEX = json.load(open(class_parse_file))
# used internally
debug_flag = False


## public API
def training_model(model,
                   name='resnet50',
                   image_size=img_size,
                   epoches=nb_epoch,
                   batch_size=batch_size,
                   modified=False,
                   rate=4
                   ):
    if modified:
        weight_path = "./weights/%s_modified_weights_%d.{epoch:02d}.h5" % (name, rate)
        csv_path = './result/train_%s_modified_imagenet_%d.csv' % (name, rate)
        lr_func = lr_fine_tune_schedule
    else:
        weight_path = "./weights/%s_weights.{epoch:02d}.h5" % name
        csv_path = './result/train_%s_imagenet.csv' % name
        lr_func = lr_train_schedule
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_func(0), momentum=0.9, decay=0.0001),
                  metrics=['accuracy', acc_top5])
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    lr_scheduler = LearningRateScheduler(lr_func)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger(csv_path)
    ckpt = ModelCheckpoint(filepath=weight_path, monitor='loss',
                           save_best_only=True,
                           save_weights_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=False)
    model.fit_generator(generator=training_data_gen(image_size, name=name),
                        steps_per_epoch=1281167 / batch_size,  # 1281167 is the number of training data we have
                        validation_data=evaluating_data_gen(image_size, name=name),
                        validation_steps=50000 / evaluating_batch_size,
                        epochs=epoches, verbose=1, max_q_size=32,
                        workers=16,
                        callbacks=[lr_reducer, lr_scheduler, early_stopper, csv_logger, ckpt])
    cprint("training is done\n", "yellow")

def evaluate_model(model, name='resnet50', image_size=img_size):
    nb_eval = 50000
    data_gen = evaluating_data_gen(image_size, name=name)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0), metrics=['accuracy', acc_top5])
    model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy', acc_top5])
    res = model.evaluate_generator(generator=data_gen,
                                   #steps=nb_eval / evaluating_batch_size,
                                   workers=16,
                                   max_q_size=16)
    cprint("top1 acc:" + str(res[1]), "red")
    cprint("top5 acc:" + str(res[2]), "red")

def test_speed(model, name='resnet50', image_size=img_size):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0), metrics=['accuracy'])
    img_path = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"

    if name == 'resnet50':
        preprocessing_function = resnet_preprocess_input
    elif name == 'vgg16':
        preprocessing_function = vgg_preprocess_input
    elif name == 'inceptionv3':
        preprocessing_function = inception_preprocess_input
    else:
        preprocessing_function = imagenet_utils.preprocess_input

    img = image.load_img(img_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessing_function(x)
    model.predict(x)
    model.predict(x)
    start_time = time.time()
    preds = model.predict(x)
    end_time = time.time()
    print('Predicted:', decode_predictions(preds))
    print('Time used:', end_time-start_time)



##private API
def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def training_data_gen(image_size=img_size, name='resnet50'):
    if name == 'resnet50':
        preprocessing_function = resnet_preprocess_input
    elif name == 'vgg16':
        preprocessing_function = vgg_preprocess_input
    elif name == 'inceptionv3':
        preprocessing_function = inception_preprocess_input
    else:
        preprocessing_function = imagenet_utils.preprocess_input

    datagen = ImageDataGenerator(
        channel_shift_range=10,
        horizontal_flip=True,  # randomly flip images

        preprocessing_function=preprocessing_function)

    img_dir = os.path.join(img_parent_dir, "train")
    img_generator = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(image_size, image_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True)

    return img_generator


def evaluating_data_gen(image_size=img_size, name='resnet50'):
    if name == 'resnet50':
        preprocessing_function = resnet_preprocess_input
    elif name == 'vgg16':
        preprocessing_function = vgg_preprocess_input
    elif name == 'inceptionv3':
        preprocessing_function = inception_preprocess_input
    else:
        preprocessing_function = imagenet_utils.preprocess_input

    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function)

    img_dir = os.path.join(img_parent_dir, "val")
    img_generator = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(image_size, image_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=evaluating_batch_size,
        shuffle=True)

    return img_generator


def generate_digit_indice_dict():
    digit_indice_dict = {value[0]: int(key) for key, value in imagenet_utils.CLASS_INDEX.items()}
    return digit_indice_dict


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_end(self, logs={}):
        numpy_loss_history = np.array(self.losses)
        np.savetxt("./result/fine_tune_loss_history.1024.txt", numpy_loss_history, delimiter=",")


def lr_fine_tune_schedule(epoch):
    if epoch < 3:
        lr = 1e-3
    elif epoch < 6:
        lr = 1e-3*sqrt(0.1)
    elif epoch < 9:
        lr = 1e-4
    elif epoch < 12:
        lr = 1e-4*sqrt(0.1)
    #lr*=sqrt(0.1)
    print('Learning rate: ', lr)
    return lr


def lr_train_schedule(epoch):
    lr = 1e-4
    if epoch >= 4:
        lr *= sqrt(0.1)
    if epoch >= 8:
        lr *= sqrt(0.1)
    print('Learning rate: ', lr)
    return lr


# private data member
digit_indice_dict = generate_digit_indice_dict()

##for debug:
if __name__ == "__main__":
    pass
