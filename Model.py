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

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout

from layers import cnn_model,test,lenet5,vgg16,FitNet

def model(name, input_ph = None, input_shape= None,
          nb_classes= 10, logits= False, prun_para=None):
    """
    Define CNN model using Keras

    :param name: Choose which model to use
    :param load_model: Whether to load the model in keras
    :param input_ph: The TensorFlow tensor for the input
    :param input_shape: The input shape
    :param nb_classes: The number of output classes
    :param logits: If set to False, returns a Keras model, otherwise will also return logits tensor
    :return: model / logits and model
    """

    """python
    pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=0.2,
            final_sparsity=0.8, begin_step=1000, end_step=2000),
        'block_size': (2, 3),
        'block_pooling_type': 'MAX'
    }

    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(100,)),
        prune_low_magnitude(layers.Dense(2, activation='tanh'), **pruning_params)
    ])
    """

    model = Sequential()

    if name == 'test':
        layers = test(input_shape= input_shape, nb_classes= nb_classes)
    elif name == 'cnn_model':
        layers = cnn_model(input_shape=input_shape, nb_classes=nb_classes)
    elif name == 'lenet5':
        layers = lenet5(input_shape= input_shape, nb_classes= nb_classes)
    elif name == 'vgg16':
        layers = vgg16(input_shape= input_shape, nb_classes= nb_classes)
    elif name == 'fitnet':
        layers = FitNet(input_shape= input_shape, nb_classes= nb_classes)
    else:
        raise ValueError('No Model')

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    if prun_para is not None:
        model = model_prune(model, prun_para, N_data=60000, batch_size=128)
    else:
        pass

    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

def model_load(load_model, input_ph = None, input_shape= None,
          nb_classes= 10, logits= False, prun_para=None):
    if load_model == 'VGG16':
        model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                                input_shape=input_shape, classes=nb_classes)
        # 'imagenet'
    elif load_model == 'VGG19':
        model = keras.applications.vgg19.VGG19(include_top=True, weights=None,
                                                input_shape=input_shape, classes=nb_classes)
    elif load_model == 'densenet':
        model = keras.applications.densenet.DenseNet121(include_top=True, weights=None,
                                                        input_shape=input_shape, classes=nb_classes)
        # model = keras.applications.densenet.DenseNet169()
        # model = keras.applications.densenet.DenseNet201()
    elif load_model == 'inception':
        model = keras.applications.inception_resnet_v2.InceptionResNetV2()
        # model = keras.applications.inception_v3.InceptionV3()
    elif load_model == 'resnet50':
        model = keras.applications.resnet50.ResNet50()
    elif load_model == 'mobilenet':
        model = keras.applications.mobilenet.MobileNet(include_top=True, weights=None,
                                                        input_shape=input_shape, classes=nb_classes)
    elif load_model == 'nasnet':
        model = keras.applications.nasnet.NASNetMobile()
        #model = keras.applications.nasnet.NASNetLarge()
    elif load_model == 'xception':
        model = keras.applications.xception.Xception()
    else:
        raise ValueError('No Model')
    return model

import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude
def model_prune(model, prun_para=None, N_data=60000, batch_size=128):
    every_epoch_step = np.ceil(1.0 * N_data / batch_size).astype(np.int32)
    pruning_params = {'pruning_schedule': PolynomialDecay(initial_sparsity= prun_para['initial_sparsity'],
                                                            final_sparsity= prun_para['final_sparsity'],
                                                            begin_step= prun_para['begin_epoch'] * every_epoch_step,
                                                            end_step= prun_para['end_epoch'] * every_epoch_step,
                                                            frequency= prun_para['frequency'] #* every_epoch_step
                                                            )}
    model = prune_low_magnitude(model, **pruning_params)
    return model

if __name__ == '__main__':
    name = ['cnn_model','test','lenet5']
    for nm in name:
        print(nm)
        mod = model(nm, input_shape= (28, 28, 1), nb_classes= 10)
        mod.summary()

    prun_para = {'initial_sparsity': 0,
                'final_sparsity': 0.8,
                'begin_epoch': 0,
                'end_epoch': 1,
                'frequency': 200}
    for nm in name:
        print(nm)
        mod = model(nm, input_shape= (28, 28, 1), nb_classes= 10, prun_para=prun_para)
        mod.summary()            

    name = ['vgg16','FitNet']
    for nm in name:
        print(nm)
        mod = model(nm, input_shape= (32, 32, 3), nb_classes= 10)
        mod.summary()

    name = ['VGG16','VGG19','densenet','inception','resnet50','mobilenet','nasnet','xception']
    for nm in name:
        print(nm)
        mod = model_load(load_model=nm ,input_shape= (32, 32, 3), nb_classes= 10)
        mod.summary()