<<<<<<< HEAD:bap/adversarial/Adversarial.py
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

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import Noise
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import VirtualAdversarialMethod

class Adversarial_Model:
    def __init__(self, adv, 
                eps = 0.3, 
                iterations = 1,
                targeted = None,
                clip_min = 0.0, clip_max = 1.0):
        # adv = fgsm  noise  pgd  CW  deepfool  JSMA  VATM # MIM
        self.adv = adv
        if eps is None:
            self.eps = 0.3
        else:
            self.eps = eps
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max
        if iterations is None:
            self.iter = 10
        else:
            self.iter = iterations

        self.adv_model, self.adv_params = self.get_params()

    def get_params(self):

        if self.adv == 'fgsm':
            adv_model = FastGradientMethod
            adv_params = {'eps': self.eps,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}

        elif self.adv == 'pgd':
            adv_model = ProjectedGradientDescent
            adv_params = {"eps": self.eps, "eps_iter": 0.05, "ord": np.inf,
                          "clip_min":  self.clip_min, "clip_max": self.clip_max,
                          # "y":None, #"y_target": None,
                          "nb_iter": self.iter,
                          "rand_init": None,  # 默认是，可以false
                          "rand_minmax": 0.3}

        elif self.adv == 'noise':
            adv_model = Noise
            adv_params = {'eps': self.eps, "ord": np.inf,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}

        elif self.adv == 'CW':
            adv_model = CarliniWagnerL2
            adv_params =  {"clip_min": self.clip_min, "clip_max": self.clip_max,
                           #"y":None, #"y_target": None,
                           'batch_size':128, 'confidence':0, 'learning_rate':0.01,
                           'binary_search_steps': 1, 'max_iterations':self.iter, 'abort_early':True,
                           'initial_const':10}

        elif self.adv == 'deepfool':
            adv_model = DeepFool
            adv_params = {'max_iter': self.iter,
                          'nb_candidate': 10,
                          'overshoot': 0.02,
                          'clip_min': self.clip_min,
                          'clip_max': self.clip_max}
        # _logger.setLevel(logging.WARNING) # deep_fool.py line 19
        # line 218,219, 245-249


        elif self.adv == 'JSMA':
            adv_model = SaliencyMapMethod
            adv_params = {'theta': 0.2,  # Perturbation introduced to modified components
                          'gamma': self.eps,  # Maximum percentage of perturbed features
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}  # ,
                        # 'y_target': None}  #Target tensor if the attack is targeted

        elif self.adv == 'VATM':
            adv_model = VirtualAdversarialMethod
            adv_params = {'eps': self.eps,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max,
                          'nb_iter': self.iter,  # 1
                          'xi': 1e-6}  # the finite difference parameter

        return adv_model, adv_params


#import tensorflow.compat.v1 as tf
#import tensorflow as tf
# def generate_adv(x_input, batch_size, adv = None, adv_params = None, sess= None):
#
#     x = tf.placeholder(tf.float32, shape = x_input[0: batch_size].shape)
#     x_adv_batch = adv.generate(x_input, **adv_params)
#     # return x_adv_batch
#     x_adv = np.empty_like(x_input)
#
#     batches = np.ceil((x_input.shape[0] * 1.0)/ batch_size).astype(np.int32)
#
#     for batch in range(batches):
#         start = batch * batch_size
#         end = min(x_input.shape[0], (batch+1)*batch_size)
#
#         x_adv[start: end] = sess.run(x_adv_batch, feed = {x: x_input[start: end]})
#
#     return x_adv



if __name__ == '__main__':
    import tensorflow.compat.v1 as tf
    # import tensorflow as tf
    from tensorflow.compat.v1 import keras

    from cleverhans.utils_keras import KerasModelWrapper
    from data_load import DataSet
    from Model import model as Model
    from Adversarial_loss import get_adversarial_acc_metric, get_adversarial_loss
    dataset = DataSet(name= 'mnist', label_smoothing=0.1)
    model = Model(name= 'cnn_model', input_shape= dataset.input_shape, nb_classes=dataset.num_classes)
    model(model.input)

    sess = tf.Session()
    keras.backend.set_session(sess)

    # adv = fgsm  noise  pgd (0)  CW (0.5,feed) deepfool (2) JSMA (0) VATM # MIM
    wrap_model = KerasModelWrapper(model)
    attack = 'deepfool'
    eps = 0.3
    iterations=10
    at_model = Adversarial_Model(adv = attack, eps = eps, iterations = iterations).adv_model
    attack_params = Adversarial_Model(adv = attack, eps= eps, iterations = iterations).adv_params
    attack_model = at_model(wrap_model, sess = sess)

    adv_acc_metric = get_adversarial_acc_metric(model, attack_model, attack_params)
    model.compile(loss = get_adversarial_loss(model, attack_model, attack_params),
                    optimizer = keras.optimizers.Adam(),
                    metrics = ['accuracy'] + [adv_acc_metric])
    callbacks = []
    model.fit(dataset.X_train, dataset.Y_train,
                       batch_size= 128,
                       epochs= 10,
                       verbose=1,
                       callbacks= callbacks,
                       validation_data=(dataset.X_test, dataset.Y_test))

    acc_all = model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
    print(acc_all)
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

import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import Noise
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import VirtualAdversarialMethod

class Adversarial_Model:
    def __init__(self, adv, 
                eps = 0.3, 
                iterations = 1,
                targeted = None,
                clip_min = 0.0, clip_max = 1.0):
        # adv = fgsm  noise  pgd  CW  deepfool  JSMA  VATM # MIM
        self.adv = adv
        if eps is None:
            self.eps = 0.3
        else:
            self.eps = eps
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max
        if iterations is None:
            self.iter = 10
        else:
            self.iter = iterations

        self.adv_model, self.adv_params = self.get_params()

    def get_params(self):

        if self.adv == 'fgsm':
            adv_model = FastGradientMethod
            adv_params = {'eps': self.eps,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}

        elif self.adv == 'pgd':
            adv_model = ProjectedGradientDescent
            adv_params = {"eps": self.eps, "eps_iter": 0.05, "ord": np.inf,
                          "clip_min":  self.clip_min, "clip_max": self.clip_max,
                          # "y":None, #"y_target": None,
                          "nb_iter": self.iter,
                          "rand_init": None,  # 默认是，可以false
                          "rand_minmax": 0.3}

        elif self.adv == 'noise':
            adv_model = Noise
            adv_params = {'eps': self.eps, "ord": np.inf,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}

        elif self.adv == 'CW':
            adv_model = CarliniWagnerL2
            adv_params =  {"clip_min": self.clip_min, "clip_max": self.clip_max,
                           #"y":None, #"y_target": None,
                           'batch_size':128, 'confidence':0, 'learning_rate':0.01,
                           'binary_search_steps': 1, 'max_iterations':self.iter, 'abort_early':True,
                           'initial_const':10}

        elif self.adv == 'deepfool':
            adv_model = DeepFool
            adv_params = {'max_iter': self.iter,
                          'nb_candidate': 10,
                          'overshoot': 0.02,
                          'clip_min': self.clip_min,
                          'clip_max': self.clip_max}
        # _logger.setLevel(logging.WARNING) # deep_fool.py line 19
        # line 218,219, 245-249


        elif self.adv == 'JSMA':
            adv_model = SaliencyMapMethod
            adv_params = {'theta': 0.2,  # Perturbation introduced to modified components
                          'gamma': self.eps,  # Maximum percentage of perturbed features
                          'clip_min': self.clip_min, 'clip_max': self.clip_max}  # ,
                        # 'y_target': None}  #Target tensor if the attack is targeted

        elif self.adv == 'VATM':
            adv_model = VirtualAdversarialMethod
            adv_params = {'eps': self.eps,
                          'clip_min': self.clip_min, 'clip_max': self.clip_max,
                          'nb_iter': self.iter,  # 1
                          'xi': 1e-6}  # the finite difference parameter

        return adv_model, adv_params


#import tensorflow.compat.v1 as tf
#import tensorflow as tf
# def generate_adv(x_input, batch_size, adv = None, adv_params = None, sess= None):
#
#     x = tf.placeholder(tf.float32, shape = x_input[0: batch_size].shape)
#     x_adv_batch = adv.generate(x_input, **adv_params)
#     # return x_adv_batch
#     x_adv = np.empty_like(x_input)
#
#     batches = np.ceil((x_input.shape[0] * 1.0)/ batch_size).astype(np.int32)
#
#     for batch in range(batches):
#         start = batch * batch_size
#         end = min(x_input.shape[0], (batch+1)*batch_size)
#
#         x_adv[start: end] = sess.run(x_adv_batch, feed = {x: x_input[start: end]})
#
#     return x_adv

import tensorflow.compat.v1 as tf
#import tensorflow as tf
from tensorflow.compat.v1 import keras

from cleverhans.utils_keras import KerasModelWrapper
from data_load import DataSet
from Model import model as Model
from Adversarial_loss import get_adversarial_acc_metric, get_adversarial_loss
if __name__ == '__main__':
    dataset = DataSet(name= 'mnist', label_smoothing=0.1)
    model = Model(name= 'cnn_model', input_shape= dataset.input_shape, nb_classes=dataset.num_classes)
    model(model.input)

    sess = tf.Session()
    keras.backend.set_session(sess)

    # adv = fgsm  noise  pgd (0)  CW (0.5,feed) deepfool (2) JSMA (0) VATM # MIM
    wrap_model = KerasModelWrapper(model)
    attack = 'deepfool'
    eps = 0.3
    iterations=10
    at_model = Adversarial_Model(adv = attack, eps = eps, iterations = iterations).adv_model
    attack_params = Adversarial_Model(adv = attack, eps= eps, iterations = iterations).adv_params
    attack_model = at_model(wrap_model, sess = sess)

    adv_acc_metric = get_adversarial_acc_metric(model, attack_model, attack_params)
    model.compile(loss = get_adversarial_loss(model, attack_model, attack_params),
                    optimizer = keras.optimizers.Adam(),
                    metrics = ['accuracy'] + [adv_acc_metric])
    callbacks = []
    model.fit(dataset.X_train, dataset.Y_train,
                       batch_size= 128,
                       epochs= 10,
                       verbose=1,
                       callbacks= callbacks,
                       validation_data=(dataset.X_test, dataset.Y_test))

    acc_all = model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
    print(acc_all)
>>>>>>> 5952b40853b4ca9143ae5469167c8065a5523a59:Adversarial.py
