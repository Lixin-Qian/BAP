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
import os
import tensorflow.compat.v1 as tf
#from tensorflow import keras

from data_load import DataSet
from Model import model as Model
from Adversarial import Adversarial_Model
from Adversarial_loss import get_adversarial_acc_metric, get_adversarial_loss
from Train import Train
import warnings

warnings.filterwarnings("ignore")

config = tf.ConfigProto(intra_op_parallelism_threads= 2,
                        inter_op_parallelism_threads= 2)
sess = tf.Session(config = config)
# sess = tf.Session()
tf.keras.backend.set_session(sess)

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

# Load Dataset
dataset = DataSet(name= 'mnist', label_smoothing= 0.1)

# model = Model(name='lenet5', input_shape=dataset.input_shape, nb_classes=dataset.num_classes,
#                   prun_para=None)
#
# Train(sess, model, dataset, nb_epochs= 70, learning_rate=1e-3, batch_size=128,
#           load=None, attack=None, adv_train='bat', eps= 5, iterations= 5,
#           prun_para= None)
sparsity = [#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            #0.91, 0.93, 0.95, 0.97, 0.99,
            0.995]
for spar in sparsity:
    prun_para = {'initial_sparsity': 0,
                 'final_sparsity': spar,
                 'begin_epoch': 0,
                 'end_epoch': 30,
                 'frequency': 100}

    model = Model(name='lenet5', input_shape=dataset.input_shape, nb_classes=dataset.num_classes,
                  prun_para=prun_para)

    Train(sess, model, dataset, nb_epochs= 70, learning_rate=1e-3, batch_size=128,
          load=None, attack=None, adv_train='deepfool', eps= 5, iterations= 5,
          prun_para= prun_para)


# FGSM
# sparsity = np.linspace(0, 0.9, num=10).tolist()
# # [0, 0.1, 0.2, 0.3, 0.4...]
# sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# for index, spar in enumerate(sparsity):
#     prun_para = {'initial_sparsity': 0,
#                  'final_sparsity': sparsity[index + 1],
#                  'begin_epoch': 0,
#                  'end_epoch': 20,
#                  'frequency': 100}
#
#     model = Model(name='lenet5', input_shape=dataset.input_shape, nb_classes=dataset.num_classes,
#                   prun_para= prun_para)
#
#
#     Train(sess, model, dataset, nb_epochs= 50, learning_rate= 1e-3, batch_size= 128,
#           load= None, attack= None, adv_train= 'fgsm', eps = 0.3,
#           prun_para=prun_para)

