<<<<<<< HEAD:bap/adversarial/Adversarial_loss.py
#!/usr/bin/python
#coding:utf-8
# ==============================================================================
# Copyright 2021 - Qian Xuesen Laboratory, China Academy of Space Technology, 
# and all authors of the paper {arxiv.org/abs/2004.05913}.
# This code is the calculation program of the paper 
# `` Blind Adversarial Pruning: Towards the comprehensive robust models with 
# gradually pruning against blind Adversarial Attack ''(BAP)
# ==============================================================================
"""
Dependency libraries:
This code is writing in Python, 
dependence on tensorflow: https://github.com/tensorflow/tensorflow
and cleverhans: https://github.com/tensorflow/cleverhans.
and tensorflow/model-optimization:  https://github.com/tensorflow/model-optimization
"""

import tensorflow.compat.v1 as tf
#import tensorflow as tf
from tensorflow.compat.v1 import keras
#from tensorflow import keras

from .CoS import cut, scale, norm

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)
    return adv_acc

def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)
        return 0.5 * cross_ent + 0.5 * cross_ent_adv
    return adv_loss

    
#########################
def get_DATAA_acc_metric(model, fgsm, fgsm_params, 
                    DATAA_para = {'adv_ord': 2,
                                  'cal_delta': False,
                                  'delta': 1.0,
                                  'rho': 1.0}):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        x_adv = cut(x_adv, model.input, 
                    delta = DATAA_para['delta'], 
                    ord=DATAA_para['adv_ord'], 
                    cal_delta=DATAA_para['cal_delta'])
        x_adv = scale(x_adv, model.input, 
                    rho = DATAA_para['rho'])

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc

def get_DATAA_loss(model, fgsm, fgsm_params, 
                    DATAA_para = {'adv_ord': 2,
                                  'cal_delta': False,
                                  'delta': 1.0,
                                  'rho': 1.0}):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        x_adv = cut(x_adv, model.input, 
                    delta = DATAA_para['delta'], 
                    ord=DATAA_para['adv_ord'], 
                    cal_delta=DATAA_para['cal_delta'])
        x_adv = scale(x_adv, model.input, 
                    rho = DATAA_para['rho'])

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        # Loss
        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss

def get_adv_norm(model, fgsm, fgsm_params, ord=2):
    def norm_all(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        
        return norm(x_adv, model.input, ord=ord)

    return norm_all
=======
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

import tensorflow.compat.v1 as tf
#import tensorflow as tf
from tensorflow.compat.v1 import keras
#from tensorflow import keras

from CoS import cut, scale, norm

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)
    return adv_acc

def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)
        return 0.5 * cross_ent + 0.5 * cross_ent_adv
    return adv_loss

    
#########################
def get_DATAA_acc_metric(model, fgsm, fgsm_params, 
                    DATAA_para = {'adv_ord': 2,
                                  'cal_delta': False,
                                  'delta': 1.0,
                                  'rho': 1.0}):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        x_adv = cut(x_adv, model.input, 
                    delta = DATAA_para['delta'], 
                    ord=DATAA_para['adv_ord'], 
                    cal_delta=DATAA_para['cal_delta'])
        x_adv = scale(x_adv, model.input, 
                    rho = DATAA_para['rho'])

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc

def get_DATAA_loss(model, fgsm, fgsm_params, 
                    DATAA_para = {'adv_ord': 2,
                                  'cal_delta': False,
                                  'delta': 1.0,
                                  'rho': 1.0}):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        x_adv = cut(x_adv, model.input, 
                    delta = DATAA_para['delta'], 
                    ord=DATAA_para['adv_ord'], 
                    cal_delta=DATAA_para['cal_delta'])
        x_adv = scale(x_adv, model.input, 
                    rho = DATAA_para['rho'])

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        # Loss
        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss

def get_adv_norm(model, fgsm, fgsm_params, ord=2):
    def norm_all(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        
        return norm(x_adv, model.input, ord=ord)

    return norm_all
>>>>>>> 5952b40853b4ca9143ae5469167c8065a5523a59:Adversarial_loss.py
