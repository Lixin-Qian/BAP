#!/usr/bin/python
#coding:utf-8
# ==============================================================================
# Copyright 2021 - Qian Xuesen Laboratory, China Academy of Space Technology, 
# and all authors of the paper {arxiv.org/abs/2004.05913}.
# This code is the calculation program of the paper 
# `` Blind Adversarial Pruning: Towards the comprehensive robust models with 
# gradually pruning against blind Adversarial Attack ''(BAP)
# ==============================================================================

# For more guidance, see README.md for detail.

#import tensorflow as tf
#from tensorflow import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
MaxPooling2D = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout

# cnn_model: cnn + cnn + cnn + dense     
# test:      cnn + dense + dense
# lenet5:
# vgg16:
# FitNet


def cnn_model(input_shape=(28, 28, 1), nb_classes=10, nb_filters = 32):
    layers = [Conv2D(filters=nb_filters, kernel_size=[8, 8], strides=(2, 2),
                    input_shape=input_shape,
                    padding='same', activation ='relu'),
                Conv2D(filters=nb_filters * 2, kernel_size=[6, 6], strides=(2, 2),
                    input_shape=input_shape,
                    padding='valid', activation ='relu'),
                Conv2D(filters=nb_filters * 2, kernel_size=[5, 5], strides=(1, 1),
                    input_shape=input_shape,
                    padding='valid', activation ='relu'),
                Flatten(),
                Dense(nb_classes)]
    return layers

def test(input_shape=(28, 28, 1), nb_classes=10, dropout_rate=0.5):
    layers = [Conv2D(filters=32, kernel_size=[5, 5], strides=(2, 2),
                     input_shape=input_shape,
                     padding='same', activation ='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                #Dropout(rate=dropout_rate),
                ##############加上dropout报错？input_tensor
                #Dense(200, activation='relu'),
                Dense(nb_classes)]
    return layers

def lenet5(input_shape=(28, 28, 1), nb_classes=10, dropout_rate=0.5):
    layers = [Conv2D(filters=32, kernel_size=[5, 5], strides=(1, 1),
                     input_shape=input_shape,
                     padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2, 2)),
              Conv2D(filters=64, kernel_size=[5, 5], strides=(1, 1),
                     padding='same', activation='relu'),
              MaxPooling2D(pool_size=(2, 2)),
              Flatten(),
              # Dropout(rate=dropout_rate),
              # Dense(200, activation='relu'),
              # Dropout(rate=dropout_rate),
              Dense(84, activation='relu'),
              Dense(nb_classes)]
    return layers

def vgg16(input_shape=(28, 28, 1), nb_classes=10, nb_filters=64):
    # vgg16
    kernel_initializer = 'glorot_uniform'  # glorot_uniform  he_normal
    kernel_regularizer = None  # keras.regularizers.l2(5e-2)  #None
    layers = [Conv2D(filters=nb_filters, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     input_shape=input_shape),
              Conv2D(filters=nb_filters, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=(2, 2)),
              Conv2D(filters=nb_filters * 2, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 2, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=(2, 2)),
              # Dropout(rate=0.5),
              Conv2D(filters=nb_filters * 4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=(2, 2)),
              # Dropout(rate=0.5),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=(2, 2)),
              # Dropout(rate=0.5),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              #### mnist要去掉一个maxpool
              MaxPooling2D(pool_size=[2, 2], strides=(2, 2)),
              # keras.layers.GlobalMaxPooling2D(),
              # Dropout(rate=0.5),
              Flatten(),
              Dense(4096),
              Dense(4096),
              Dense(1024),
              # Dropout(rate=0.5),
              Dense(nb_classes)]
    return layers

def FitNet(input_shape=(28, 28, 1), nb_classes=10, nb_filters=32,):
    # fitnet-4
    #from http://nooverfit.com/wp/%E7%94%A8keras%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E5%87%86%E7%A1%AE%E7%8E%8790%E7%9A%84cifar-10%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/
    #Keras train Cifar-10 with acc 90%+     All you need is a good init
    kernel_initializer='glorot_uniform'  #glorot_uniform  he_normal
    kernel_regularizer=None #keras.regularizers.l2(1e-2)  #None
    layers = [Conv2D(filters=nb_filters, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     input_shape=input_shape),
              Conv2D(filters=nb_filters, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*3/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*3/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=2),
              # Dropout(rate=0.25),
              Conv2D(filters=int(nb_filters*5/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*5/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*5/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*5/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=int(nb_filters*5/2), kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              MaxPooling2D(pool_size=[2, 2], strides=2),
              # Dropout(rate=0.25),
              Conv2D(filters=nb_filters*4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters*4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters*4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters*4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              Conv2D(filters=nb_filters*4, kernel_size=(3, 3), strides=(1, 1),
                     padding="same", activation='relu', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer),
              keras.layers.GlobalMaxPooling2D(),
              # Dropout(rate=0.25),
              Flatten(),
              Dense(500),
              # Dropout(rate=0.5),
              Dense(nb_classes)]
    return layers

if __name__ == '__main__':
    name = ['cnn_model','test','lenet5','vgg16','FitNet']
    layer1 = cnn_model()
    print(layer1)
    layer2 = test()
    print(layer2)    
    layer3 = lenet5()
    print(layer3)
    layer4 = vgg16()
    print(layer4)
    layer5 = FitNet()
    print(layer5)