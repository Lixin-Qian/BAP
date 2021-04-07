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
import numpy as np
import os
import tensorflow.compat.v1 as tf
#from tensorflow import keras
from tensorflow.compat.v1 import keras
import warnings

from data_load import DataSet
from Adversarial import Adversarial_Model
from Adversarial_loss import get_adversarial_acc_metric, get_adversarial_loss
from CoS import cut

# import cleverhans
from cleverhans.utils_keras import KerasModelWrapper


warnings.filterwarnings("ignore")

sess = tf.Session()
tf.keras.backend.set_session(sess)

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def gen_adv(x_test, batch_size= 128, adv=None, Adv_params=None, sess=None):
    x_input_batch = tf.placeholder(tf.float32, shape= x_test[0: batch_size].shape)
    x_adv_batch = adv.generate(x_input_batch, **Adv_params)
    x_adv = np.copy(x_test)
    n_batch = int((x_test.shape[0] + batch_size - 1) / batch_size)
    for batch in range(n_batch):
        # if windows
        print('Train batch for adv: {0} in {1}'.format(batch, n_batch), end='\r')
        end = min(x_test.shape[0], (batch + 1) * batch_size)
        start = max(end - batch_size, 0) #keep each batch = batch_size, include the last batch
        x_adv[start: end] = sess.run(x_adv_batch,
                                     feed_dict={x_input_batch: x_test[start: end]})
    return x_adv

# 两种方法，结果相近 相差小于0.1%
def estimate_acc(dataset, model,
                 attack, eps, iterations):
    if attack is not None:
        print('####Attack Model with {}####'.format(attack))
        wrap_model = KerasModelWrapper(model)

        at_model = Adversarial_Model(adv= attack, eps= iterations, iterations = iterations).adv_model
        attack_params = Adversarial_Model(adv= attack, eps= iterations, iterations = iterations).adv_params
        attack_model = at_model(wrap_model, sess= sess)

        x_adv = gen_adv(dataset.X_test, batch_size= 128,
                        adv= attack_model, Adv_params= attack_params, sess=sess)
    else:
        # pass
        x_adv = dataset.X_test

    # adv_acc = get_adversarial_acc_metric(model, attack_model, attack_params)
    # adv_loss = get_adversarial_loss(model, attack_model, attack_params)

    _, adv_accuracy = model.evaluate(x_adv, dataset.Y_test, verbose=0)
    # _, _, adv_accuracy = model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
    print(adv_accuracy)

    return adv_accuracy


def estimate_acc_df(dataset, model,
                 attack, eps, iterations, delta):
                 # eps = delta, iter = eps
    if attack is not None:
        print('####Attack Model with {}####'.format(attack))
        wrap_model = KerasModelWrapper(model)

        at_model = Adversarial_Model(adv= attack, eps= eps, iterations = iterations).adv_model
        attack_params = Adversarial_Model(adv= attack, eps= eps, iterations = iterations).adv_params
        attack_model = at_model(wrap_model, sess= sess)

        x_adv = gen_adv(dataset.X_test, batch_size= 128,
                        adv= attack_model, Adv_params= attack_params, sess=sess)
        if attack == 'fgsm':
            ord_my = np.inf
        elif attack == 'deepfool':
            ord_my = 2
        else:
            print('undefine attack')
            ord_my = 2
    else:
        x_adv = dataset.X_test

    _, adv_accuracy = model.evaluate(x_adv, dataset.Y_test, verbose=2)
    x_input = tf.placeholder(tf.float32, shape=dataset.X_test.shape)
    x_adv_input = tf.placeholder(tf.float32, shape=x_adv.shape)
    # must do that(tf.ph0, otherwise, slow due to figure larger)
    acc_all = []
    for dt in delta:
        print('Eps {0}/{1}'.format(dt, delta[-1]), end='\r')
        cut_adv1 = cut(x_adv_input, x_input, 
                    delta = dt, ord=ord_my, cal_delta=False)
        cut_adv = sess.run(cut_adv1, feed_dict={x_input: dataset.X_test, x_adv_input: x_adv})
        _, adv_accuracy = model.evaluate(cut_adv, dataset.Y_test, verbose=0)
        #acc_all.append(adv_accuracy)
        acc_all = acc_all + [adv_accuracy]
    print('\n')
    #print(acc_all)
    return acc_all


def main(argv=None):

    sparsity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]

    epsilon = [100]
    
    folder = '../data_result/BAT/'   #  './'
    data = 'mnist'
    train = 'bat5_2'
    attack = 'deepfool'
    addons = '_70'
    #addons = ''
    filename = str(addons) +'_model_2.h5'
    resultname = str(addons) + '_' + str(attack) + '_2.csv'
    

    delta = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]  # nt-c100-fgsm

    # Load Dataset
    dataset = DataSet(name= data, label_smoothing= 0.1)
    for p in sparsity:
        print('Sparsity {0}/{1}'.format(p, sparsity[-1]))
        at_fgm_acc = []
        # Load Model
        load =  folder + str(data) + '/' + str(train) + '/'+ str(p) + filename
        # TODO try catch:  
        try:
            model = keras.models.load_model(load)
        except Exception as e:
            print (Exception,":",e)
            continue
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        for eps in epsilon:
            #if attack == 'fgsm':
            #adv_acc = estimate_acc(dataset, model,
            #                       attack = attack, eps = eps, iterations = eps)
            #    at_fgm_acc.append(adv_acc)
            #if attack == 'deepfool':
            adv_acc = estimate_acc_df(dataset, model,
                                    attack = attack, eps = eps, iterations = eps, delta = delta)                       
            at_fgm_acc = adv_acc
        # try catch : otherwise, print screan, rand-name
        np.savetxt(folder + str(data) + '/' + str(train) + '/sparsity_' +str(p) + resultname,
                   at_fgm_acc, delimiter=',')


if __name__ == '__main__':
    tf.app.run()
