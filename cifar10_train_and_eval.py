import os

import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.datasets import cifar10
from termcolor import cprint

####################################
##config params
cifar10_path = "/data1/datasets/cifar/cifar-10-batches-py/"

####################################
## public API
def model_train(model, batch_size=32, epochs=200, data_path = None, data_augmentation = True, subtract_pixel_mean = True):
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data(data_path)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./weights/resnet20_cifar10_weights.{epoch:03d}.h5',
                                 monitor='loss',
                                 save_best_only=True,
                                 save_weights_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    csv_logger = CSVLogger('./results/training_resnet20_cifar10.csv')

    callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(featurewise_center=False,# set input mean to 0 over the dataset
                                     samplewise_center=False,# set each sample mean to 0
                                     featurewise_std_normalization=False,# divide inputs by std of dataset
                                     samplewise_std_normalization=False,# divide each input by its std
                                     zca_whitening=False,# apply ZCA whitening
                                     rotation_range=0,# randomly rotate images in the range (deg 0 to 180)
                                     width_shift_range=0.1,# randomly shift images horizontally
                                     height_shift_range=0.1,# randomly shift images vertically
                                     horizontal_flip=True,# randomly flip images
                                     vertical_flip=False)# randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs,
                            verbose=1,
                            workers=8,
                            use_multiprocessing=True,
                            callbacks=callbacks)

def model_test(model, data_path=None, subtract_pixel_mean = True):
    batch_size = 64
    num_classes = 10

    cprint('start testing model', 'red')

    (x_train, y_train), (x_test, y_test) = load_data(data_path)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_test -= x_train_mean

    y_test = np_utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    scores = model.evaluate(x_test, y_test, batch_size = batch_size, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def fine_tune(model, epochs=80, data_path = None, rate=4):
    batch_size = 32
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data(data_path)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_fine_tune_schedule(0)),
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./weights/resnet20_cifar10_fine_tune_weights_%s.{epoch:02d}.h5' % rate,
                                 monitor='loss',
                                 save_best_only=False,
                                 save_weights_only=True)
    lr_scheduler = LearningRateScheduler(lr_fine_tune_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    csv_logger = CSVLogger('./result/training_resnet20_cifar10_fine_tune_%s.csv' % rate)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
                                 width_shift_range=0.1,  # randomly shift images horizontally
                                 height_shift_range=0.1,  # randomly shift images vertically
                                 horizontal_flip=True,  # randomly flip images
                                 vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        verbose=1,
                        workers=16,
                        use_multiprocessing=True,
                        callbacks=callbacks)




########################################
## private API
def load_data(data_path=None):
    if data_path == None:
        path = cifar10_path
    else:
        path = data_path

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = cifar10.load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = cifar10.load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    print("loading data done.")

    return (x_train, y_train), (x_test, y_test)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_fine_tune_schedule(epoch):
    lr = 1e-3
    if epoch > 60:
        lr *= 5e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

#####################################
## for debug
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''