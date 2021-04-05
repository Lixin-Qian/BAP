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

from data_load import DataSet
from Model import model as Model
from Train import Train

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def adversarial_pruning_training(data, model_name, nb_epochs, learning_rate, batch_size,
                                 attack, adv_train, prune):
    if prune:
        prun_para = {'initial_sparsity': 0,
                 'final_sparsity': 0.8,
                 'begin_epoch': 0,
                 'end_epoch': 1,
                 'frequency': 200}
    else:
        prun_para = None

    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    # tf.keras.backend.clear_session()

    # Load Dataset
    dataset = DataSet(name= data, label_smoothing=0.1)

    model = Model(name= model_name, input_shape= dataset.input_shape, nb_classes=dataset.num_classes,
                    prun_para= prun_para)

    # To be able to call the model in the custom loss, we need to call it once
    # before, see https://github.com/tensorflow/tensorflow/issues/23769
    model(model.input)

    Train(sess, model, dataset, nb_epochs=nb_epochs, learning_rate=learning_rate, batch_size=batch_size,
            load=None, attack=attack, adv_train=adv_train,
            prun_para=prun_para)

def main(argv = None):
    # from flag import print_configuration_op
    # print_configuration_op(FLAGS)

    adversarial_pruning_training(data = FLAGS.data,
                                 model_name = FLAGS.model_name,
                                 nb_epochs = FLAGS.epochs,
                                 learning_rate = FLAGS.learning_rate,
                                 batch_size = FLAGS.batch_size,
                                 attack = FLAGS.attack,
                                 adv_train = FLAGS.adv_train,
                                 prune = FLAGS.prune)

if __name__ == '__main__':

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('data', 'mnist', 'Which DATASET to Train')
    flags.DEFINE_string('model_name', 'lenet5', 'Which Model to Train')
    flags.DEFINE_integer('epochs', 1, 'Training Epochs')
    flags.DEFINE_float('learning_rate', 1e-3, 'Learning Rate')
    flags.DEFINE_integer('batch_size', 128, 'Training Batch_Size')
    flags.DEFINE_string('attack', 'deepfool', 'Which method to attack model')
    flags.DEFINE_string('adv_train', 'deepfool', 'Adversarial training')
    flags.DEFINE_bool('prune', True, 'Prune the model or not')

    tf.app.run()
