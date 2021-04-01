#!/usr/bin/python
#coding:utf-8
# ==============================================================================
# Copyright 2020 The ICME-2021 ID:18 Authors. All Rights Reserved.
# This open source code is uploaded only for reviewing process of ICME-2021,
# support the reliability of the computational data within paper ID:18.
# Without permission from the ICME-2021 ID:18 Authors , no one shall 
# #disseminate, copy or modify this code for purposes other than 
# reviewing manuscripts.
# ==============================================================================
"""
This code is the calculation program of the paper `` Blind Adversarial Pruning: Towards the comprehensive robust models with gradually pruning against blind Adversarial Attack ''(BAP)
The ICME-2021 ID:18.

Dependency libraries:
This code is writing in Python, 
dependence on tensorflow: https://github.com/tensorflow/tensorflow
and cleverhans: https://github.com/tensorflow/cleverhans.
and tensorflow/model-optimization:  https://github.com/tensorflow/model-optimization
"""

import tensorflow.compat.v1 as tf
#import tensorflow as tf
#from tensorflow.compat.v1 import keras
to_categorical = tf.keras.utils

dataset_ = tf.keras.datasets
mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
cifar10 = tf.keras.datasets.cifar10
cifar100 = tf.keras.datasets.cifar100

from Dataset import mnist_load_data, fashion_load_data, cifar10_load_data, cifar100_load_data
from Dataset import load_svhn, load_stl10, load_tcc

class DataSet:
    def __init__(self, name=None, label_smoothing=0):
        if name is not None:
            self.load(name, label_smoothing)
            self.name = name
            """
            self.X_train
            self.Y_train
            self.X_test
            self.Y_test
            self.img_rows
            self.img_cols
            self.img_channels
            self.num_classes
            self.input_shape
            self.Name
            """

    def load(self, name, label_smoothing=0):
        if name not in ['mnist', 'fashion', 'cifar10', 'cifar100', 'svhn', 'stl10', 'tcc']:
            raise ValueError('dataset must be mnist, fashion, or cifar10, cifar100, svhn, stl10, tcc')
        """
        on going
        stl-10
        train_start=0, train_end=60000, test_start=0, test_end=10000
        :param train_start: index of first training set example
        :param train_end: index of last training set example
        :param test_start: index of first test set example
        :param test_end: index of last test set example
        :param label_smoothing: float, amount of label smoothing for cross entropy        
        """
        if name in ['mnist', 'fashion', 'cifar10', 'cifar100']:
            # load by keras
            if name == 'mnist':
                (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
                num_classes = 10
                img_rows, img_cols, img_channels = 28, 28, 1
            elif name == 'fashion':
                (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
                num_classes = 10
                img_rows, img_cols, img_channels = 28, 28, 1
            elif name == 'cifar10':
                (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
                num_classes = 10
                img_rows, img_cols, img_channels = 32, 32, 3
            elif name == 'cifar100':
                (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
                num_classes = 100
                img_rows, img_cols, img_channels = 32, 32, 3

            # reshape dataset
            if tf.keras.backend.image_data_format() == 'channels_first':
                X_train = X_train.reshape(img_channels, X_train.shape[0], img_rows, img_cols)
                X_test = X_test.reshape(img_channels, X_test.shape[0], img_rows, img_cols)
                input_shape = (img_channels, img_rows, img_cols)
            else:
                X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
                X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
                input_shape = (img_rows, img_cols, img_channels)

        else:
            # load by mycode
            if name == 'svhn':
                (X_train, Y_train), (X_test, Y_test) = load_svhn()
                num_classes = 10
                img_rows, img_cols, img_channels = 32, 32, 3
                # TO DO: permute 
                input_shape = (img_rows, img_cols, img_channels)
            elif name == 'stl10':
                print('dataset stl10 un-done')
                (X_train, Y_train), (X_test, Y_test) = load_stl10()

        # scale to 0,1
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        if name == "tcc":
            (X_train, Y_train), (X_test, Y_test) = load_tcc()
            num_classes = 2
            img_rows, img_cols, img_channels = 2, 1, 1
            
        # convert class vectors to binary class matrices
        Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
        Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

        # Label smoothing
        Y_train = Y_train - label_smoothing * (Y_train - 1. / num_classes)

        # save data
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.Name = name

    """
    import numpy as np
    def data_augmentation(self, batch_size = 5000):
        X_train = np.zeros_like(self.X_train)
        batches = int(np.ceil(float(len(X_train)) / batch_size))
        datagen = ImageDataGenerator(featurewise_center=False,  # 均值为0 set input mean to 0 over the dataset
                                    samplewise_center=False,  # 均值为0 set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False,  # apply ZCA whitening
                                    rotation_range=0,  # 旋转 randomly rotate images in the range (degrees, 0 to 180)
                                    width_shift_range=0.1,  # 平移变换 randomly shift images horizontally (fraction of total width)
                                    height_shift_range=0.1,  # 平移变换 randomly shift images vertically (fraction of total height)
                                    horizontal_flip=True,  # 翻转 randomly flip images
                                    vertical_flip=False)  # 翻转 randomly flip images

        datagen.fit(X_train, augment=True)
        # get transformed images
        randidx = np.random.randint(X_train.shape[0], size= batch_size)

        x_augmented = X_train[randidx].copy()
        y_augmented = Y_train[randidx].copy()
        x_augmented = datagen.flow(x_augmented, np.zeros(batch_size),
                                batch_size= batch_size, shuffle=False).next()[0]
        # append augmented data to trainset
        x_train = np.concatenate((X_train, x_augmented))
        y_train = np.concatenate((Y_train, y_augmented))

        randidx = np.random.randint(X_train.shape[0], size=batch_size)
        aug_x = x_train[randidx]
        aug_y = y_train[randidx]

        return aug_x, aug_y, X_test, Y_test, img_rows, img_cols, img_channels, num_classes


        #     # datagen.fit(X_train)
        #     x_augmented = datagen.flow(X_train, batch_size= batch_size)
        #     for i in range(batches):
        #         start = i * batch_size
        #         end = min(len(X_train), start + batch_size)
        #         batch = x_augmented.next()
        #         x_train[start: end] = batch
        #
        #     print('----Data Augmentation completed----')
        #     return x_train, Y_train, X_test, Y_test, img_rows, img_cols, img_channels, num_classes
    """

if __name__ == '__main__':
    for name in ['mnist', 'fashion', 'cifar10', 'cifar100', 'svhn']: #, 'stl10', 'tcc'
        print('test load dataset: ', name)
        dataset = DataSet(name=name, label_smoothing=0.1)
        print('Name: ', dataset.Name)
        print('input_shape: ', dataset.input_shape)
        print('num_classes: ', dataset.num_classes)
        print('X_train: ', dataset.X_train.shape)
        print('Y_train: ', dataset.Y_train.shape)
        print('X_test: ', dataset.X_test.shape)
        print('Y_test: ', dataset.Y_test.shape)
