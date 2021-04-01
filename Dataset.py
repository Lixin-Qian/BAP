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
"""MNIST handwritten digits dataset.
"""

import gzip
import os

import numpy as np
from scipy.io import loadmat
#import tensorflow as tf
#from tensorflow import keras
import tensorflow.compat.v1 as tf
#from tensorflow.compat.v1 import keras
K = tf.keras.backend
#from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

cache_dir = './'
# cache_dir=None

def mnist_load_data(path='mnist.npz'):
    """Loads the MNIST dataset.

    Arguments:
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(
        path,
        cache_dir=cache_dir,
        origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
        file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def fashion_load_data():
    """Loads the Fashion-MNIST dataset.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(get_file(fname, cache_dir=cache_dir, origin=base + fname, cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def cifar10_load_data():
    """Loads CIFAR10 dataset.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, cache_dir=cache_dir, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def cifar100_load_data(label_mode='fine'):
    """Loads CIFAR100 dataset.

    Arguments:
        label_mode: one of "fine", "coarse".

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Raises:
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(dirname, cache_dir=cache_dir, origin=origin, untar=True)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

"""
ours
"""

def reformat(samples, labels):
    # change the size of data
    # (height，weight，channel，number)->(number, height，weight，channel)
    # labels to: one-hot encoding
    samples = np.transpose(samples, (3, 0, 1, 2)) #channels_last
    labels = np.array([x[0] for x in labels])

    for id in range(len(labels)):
        if labels[id] == 10:
            labels[id] = 0
    return samples, labels

def load_svhn():
    # traindata = loadmat("D:/mywork/datasets/SVHN/train_32x32.mat")
    # testdata = loadmat("D:/mywork/datasets/SVHN/test_32x32.mat")
    # cache_dir
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')  ######################### windows
    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, "datasets")
    datadir = os.path.join(datadir, "SVHN")
    traindata = loadmat(os.path.join(datadir, "train_32x32.mat"))
    testdata = loadmat(os.path.join(datadir, "test_32x32.mat"))

    train_samples = traindata['X']
    train_labels = traindata['y']
    test_samples = testdata['X']
    test_labels = testdata['y']

    x_train, y_train = reformat(train_samples, train_labels)
    x_test, y_test = reformat(test_samples, test_labels)

    return (x_train, y_train), (x_test, y_test)

def load_stl10():
    x_train=1.0
    y_train=1.0
    x_test=1.0
    y_test=1.0
    print('dataset stl10 un-done')
    return (x_train, y_train), (x_test, y_test)

def load_tcc():
    x_train=1.0
    y_train=1.0
    x_test=1.0
    y_test=1.0
    print('dataset tcc un-done')
    return (x_train, y_train), (x_test, y_test)