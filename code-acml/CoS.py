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

import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf
#from tensorflow.compat.v1 import keras

#All function in tensorflow mode

def cut(x_adv, x, delta = 0.1, ord=2, cal_delta=False):
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')

    eta = x_adv-x
    if ord == np.inf:
        eta = tf.clip_by_value(eta, -delta, delta)
        x_adv = eta + x
    else:
        norm = get_norm(eta, ord=ord)
        #cal_delta = True
        if cal_delta is True:
            delta = get_cut(norm) # reduce_mean
        # factor = min(1, delta/norm)
        factor = tf.minimum(1., tf.div(delta, norm))
        x_adv = eta * factor + x
    return x_adv

def fill(x_adv, x, delta = 1.0, ord=2, flag = None):
    # 目前还没用，还没通
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    
    eta = x_adv-x
    #ord = np.inf
    if ord == np.inf:
        pass
    else:
        norm = get_norm(eta, ord=ord)
        factor = tf.maximum(1., tf.div(delta, norm))
        
        if flag is not None:
            print(factor.shape)
            # 有bug， size不对
            tf.reshape(flag, [-1,1,1,1])
            print(flag.shape)
            factor = tf.maximum(1., tf.multiply(factor, flag))  #元素乘法
            print(factor.shape)
            x_adv = factor * eta + x
            print(x_adv.shape)
        else:
            x_adv = factor * eta + x
    return x_adv

def scale(x_adv, x, rho = 0.9):
    x_adv = (x_adv-x) * rho + x
    return x_adv

def norm(x_adv, x, ord=2):
    norm = get_norm(x_adv-x, ord=ord)
    return get_cut(norm)

def get_cut(norm):
    return tf.reduce_mean(norm)

def get_norm(eta, ord=2):

    reduc_ind = list(range(1, len(eta.shape)))
    avoid_zero_div = 1e-12
    if ord == 1:
        norm = tf.maximum(avoid_zero_div,
                        tf.reduce_sum(tf.abs(eta),
                                reduc_ind, keepdims=True))
    elif ord == 2:
        # avoid_zero_div must go inside sqrt to avoid a divide by zero
        # in the gradient through this operation

        if len(eta.shape) == 4:
            img_rows, img_cols, nchannels = eta.shape[1:4]
            pro_all = img_rows * img_cols * nchannels
        elif len(eta.shape) == 2:
            pro_all = eta.shape[1]
        else:
            print("error")

        norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                tf.reduce_sum(tf.square(eta),
                                                reduc_ind,
                                                keepdims=True) 
                                    / (tf.to_float(pro_all))
                        ))
    #return tf.reshape(norm, [-1])
    return norm
    

if __name__ == '__main__':
    print('test')
    #cut

    #fill

    #scale

    #norm

    #get_cut

    #get_norm